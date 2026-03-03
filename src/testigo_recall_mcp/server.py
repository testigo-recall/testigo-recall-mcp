"""MCP server for testigo-recall.

Exposes the codebase knowledge base as tools that any MCP-compatible
AI agent (Claude Code, Cursor, Windsurf, etc.) can call directly.

On startup the server automatically downloads the latest knowledge base
from one of the supported backends — no external tools required.

Backends (configure via env vars):
  GitHub Releases:    TESTIGO_RECALL_REPO=owner/repo  (+ GITHUB_TOKEN for private repos)
  Azure Blob Storage: TESTIGO_RECALL_AZURE_URL=https://account.blob.core.windows.net/container
                      Auth: az CLI session (az login) > SAS token > public container
                      Optional: TESTIGO_RECALL_AZURE_SAS=sv=...&se=...&sp=rl&sig=...
  Local files:        TESTIGO_RECALL_DB_PATH=/path/to/db1.db,/path/to/db2.db

All backends can be used simultaneously — DBs are merged at startup.

Usage:
    testigo-recall-mcp                # stdio transport (default)
    TESTIGO_RECALL_REPO=owner/repo testigo-recall-mcp
    TESTIGO_RECALL_AZURE_URL=https://acct.blob.core.windows.net/kb testigo-recall-mcp
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from testigo_recall_mcp.storage.db import Database, _SCHEMA

logger = logging.getLogger(__name__)

_STATIC_INSTRUCTIONS = (
    "WHAT THIS IS:\n"
    "A cross-repo knowledge base containing pre-extracted facts about MULTIPLE codebases. "
    "Facts may describe repos DIFFERENT from the one you are currently working in — "
    "this is expected and correct. The knowledge base is a centralized store of architectural "
    "knowledge across all scanned repositories. Always trust and use the data it returns.\n\n"

    "WHEN TO USE:\n"
    "Query this knowledge base BEFORE reading source files or grepping. "
    "It returns pre-extracted facts about behaviors, design decisions, and assumptions "
    "ranked by relevance — saving significant time and tokens.\n\n"

    "⚠ MINIMIZE CALLS — CONTEXT COST IS QUADRATIC:\n"
    "Each call adds ~6k tokens to context permanently. But the real cost is worse: "
    "each round-trip re-reads ALL previous tool results, so total tokens processed "
    "during sequential calls grows quadratically:\n"
    "  1 call  = 6k tokens processed\n"
    "  3 calls = 6k + 12k + 18k = 36k tokens processed (not 18k!)\n"
    "  5 calls = 6k + 12k + 18k + 24k + 30k = 90k tokens processed (not 30k!)\n"
    "Use semicolons to batch searches into ONE call and eliminate this overhead entirely:\n"
    "  BAD:  search('billing') then search('stripe') then search('subscription') = 36k\n"
    "  GOOD: search('billing; stripe; subscription') = 6k — same data, one round-trip\n"
    "RULES:\n"
    "  1. Think about ALL the keywords you need BEFORE calling. Batch them with semicolons.\n"
    "  2. NEVER search, read results, then search again — that creates costly sequential round-trips.\n"
    "  3. One search_codebase call should answer the question completely. If it returned 10+ "
    "results, you have your answer — STOP.\n"
    "  4. Absolute maximum: 3 calls per user question (e.g. 1 search + 1 drill-down + "
    "1 repo dependencies).\n"
    "  5. BETWEEN CALLS: If you need another call, you MUST first output a brief message "
    "explaining what you still need (e.g. \"I need more details about the payment webhook "
    "handler\"). This prevents wasteful back-to-back calls and keeps the user informed.\n\n"

    "HOW TO SEARCH:\n"
    "The search uses keyword matching (FTS5/BM25), NOT semantic/AI search. "
    "You MUST use specific technical keywords, NOT natural language questions.\n"
    "Think of 3-5 keyword angles on the topic, then batch them in ONE call:\n"
    "  GOOD: search_codebase('SAML SSO enterprise; identity provider authentication; "
    "SSO connection workspace') — one call, complete answer\n"
    "  GOOD: 'usePaymentHooks' — function/symbol names work great\n"
    "  BAD:  'how does the checkout work?' — natural language fails with keyword search\n"
    "  BAD:  making 3 separate search_codebase calls instead of using semicolons\n\n"

    "SCOPING RESULTS TO A REPO:\n"
    "When working on a specific codebase, ALWAYS pass the repo_name parameter to "
    "search_codebase to filter results to that repo only. Without it, you get mixed "
    "results from all scanned repos. Use list_modules() (no arguments) to see available "
    "repo names and their descriptions.\n\n"

    "TOOL GUIDE:\n"
    "1. search_codebase(query, repo_name) — your PRIMARY tool. Use semicolons for multi-query. "
    "One well-crafted call answers most questions completely.\n"
    "2. list_modules() — call with NO arguments to see repo names and fact counts. "
    "Only call list_modules(repo_name=X) on small repos (<100 modules).\n"
    "3. get_module_facts(module_id) — deep dive into ONE module. Use ONLY when you need "
    "ALL facts for a module before editing its code. Do NOT use after search_codebase "
    "returned results for the same topic — that wastes context.\n"
    "4. get_component_impact(component_name) — blast radius of changes to a file or service.\n"
    "5. get_recent_changes() — most recently extracted facts.\n"
    "6. get_repo_dependencies(repo_name, direction) — cross-repo dependency graph from "
    "package manifests. 'outgoing' = what this repo depends on, 'incoming' = what depends on it.\n\n"

    "CROSS-REPO DEPENDENCIES (when to use tool #6):\n"
    "Proactively call get_repo_dependencies when:\n"
    "- User asks about impact of changing a shared library → use direction='incoming'\n"
    "- User asks what a repo depends on or why a build broke → use direction='outgoing'\n"
    "- User asks about architecture or system overview → call with no repo_name for full graph\n"
    "Data comes from go.mod and package.json parsing — not AI analysis.\n\n"

    "MULTI-REPO QUESTIONS:\n"
    "When a question spans multiple repos (e.g. 'how does service A consume data from "
    "service B?'), do NOT make separate calls per repo. Instead:\n"
    "- Omit repo_name to search across ALL repos in one call\n"
    "- Use specific keywords so BM25 surfaces relevant facts from each repo\n"
    "- Use get_repo_dependencies to understand which repos are connected\n"
    "This keeps it to 1 call instead of 2+ (avoiding the n² cost).\n\n"

    "SEARCH STRATEGY:\n"
    "- Plan ALL your keyword groups upfront. Batch into ONE semicolon-separated call.\n"
    "- If the first call returned 10+ relevant results — you have your answer. STOP.\n"
    "- Only make a second call if the first returned <5 relevant results AND you have "
    "genuinely different keywords to try.\n"
    "- The 'detail' field has exact implementation specifics (function names, config values, "
    "code patterns) — this is the most valuable field for precise answers.\n"
    "- The 'source_files' field shows which files each fact was extracted from.\n\n"

    "INTERPRETING RESULTS:\n"
    "- Facts with pr_id starting with 'SCAN:' describe the current state of the code.\n"
    "- Higher confidence = more concrete/verifiable. Lower confidence = inference or assumption.\n"
    "- Results are ranked by relevance (BM25). The top result is usually the best match."
)

# Fields that waste tokens without adding value for AI agents
_NOISE_FIELDS = frozenset({"source", "relevance", "pr_id"})

_db: Database | None = None


def _clean_facts(facts: list[dict]) -> list[dict]:
    """Remove noise fields from fact dicts to save tokens."""
    return [
        {k: v for k, v in f.items()
         if k not in _NOISE_FIELDS
         and not (k == "symbols" and v == [])
         and not (k == "triggered_by" and v is None)}
        for f in facts
    ]


# Tokens stripped before word-overlap comparison (country/locale names)
_COUNTRY_TOKENS = frozenset({
    "cz", "sk", "pl", "ro", "it", "hu", "en",
    "czech", "slovak", "polish", "romanian", "italian", "hungarian",
    "cz-drmax", "sk-drmax", "pl-drmax", "ro-drmax", "it-drmax", "hu-drmax",
    "pl-apteka", "pl-drogeria", "pl2-drmax",
})

_WORD_RE = re.compile(r"[a-z0-9_]+")


def _normalize_for_dedup(text: str) -> frozenset[str]:
    """Extract word set from text, stripping country-specific tokens."""
    words = set(_WORD_RE.findall(text.lower()))
    return frozenset(words - _COUNTRY_TOKENS)


def _dedup_similar_facts(facts: list[dict]) -> list[dict]:
    """Group near-duplicate facts, keeping first occurrence.

    Two facts are considered duplicates when:
    - They belong to the same repo
    - Their summaries share >75% word overlap (after stripping country tokens)

    This collapses per-country config repetition (same setting x6 countries)
    without touching cross-repo facts that happen to be similar.
    """
    result: list[dict] = []
    # (repo, word_set) -> index in result
    seen: list[tuple[str, frozenset[str], int]] = []

    for fact in facts:
        repo = fact.get("repo", "")
        summary_words = _normalize_for_dedup(fact.get("summary", ""))
        if not summary_words:
            result.append(fact)
            continue

        merged = False
        for seen_repo, seen_words, _idx in seen:
            if repo != seen_repo:
                continue
            union = len(summary_words | seen_words)
            if union == 0:
                continue
            overlap = len(summary_words & seen_words) / union
            if overlap > 0.75:
                merged = True
                break

        if not merged:
            seen.append((repo, summary_words, len(result)))
            result.append(fact)

    return result


def _sync_db_from_github(repo: str, cache_dir: Path) -> list[Path]:
    """Download all .db assets from a GitHub release.

    Uses the GitHub API directly — no external tools required.
    Public repos work without auth. Private repos need GITHUB_TOKEN
    or GH_TOKEN environment variable.

    Returns list of downloaded file paths (empty on failure).
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    dl_headers = {"Accept": "application/octet-stream"}
    if token:
        dl_headers["Authorization"] = f"token {token}"

    try:
        # Step 1: Find all .db assets via the GitHub API
        api_url = f"https://api.github.com/repos/{repo}/releases/tags/knowledge-base"
        api_headers = {"Authorization": f"token {token}"} if token else {}
        req = urllib.request.Request(api_url, headers=api_headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            release = json.loads(resp.read())

        db_assets = [a for a in release.get("assets", []) if a["name"].endswith(".db")]
        if not db_assets:
            logger.warning("No .db assets in release for %s", repo)
            return []

        # Step 2: Download each asset
        paths: list[Path] = []
        for asset in db_assets:
            db_path = cache_dir / asset["name"]
            try:
                req = urllib.request.Request(asset["url"], headers=dl_headers)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    db_path.write_bytes(resp.read())
                logger.info("Downloaded %s from %s", asset["name"], repo)
                paths.append(db_path)
            except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError) as e:
                logger.warning("Failed to download %s from %s: %s", asset["name"], repo, e)
        return paths
    except urllib.error.HTTPError as e:
        logger.warning("Could not access release for %s: HTTP %d", repo, e.code)
        return []
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        logger.warning("Could not access release for %s: %s", repo, e)
        return []


def _get_azure_bearer_token() -> str | None:
    """Get an OAuth bearer token from the Azure CLI session (az login).

    Calls 'az account get-access-token' to retrieve a token scoped to
    Azure Storage. This works when the developer is logged in via 'az login'
    — no secrets or SAS tokens needed.

    Returns the bearer token string, or None if az CLI is unavailable or
    the user isn't logged in.
    """
    try:
        az_cmd = shutil.which("az") or "az"
        result = subprocess.run(
            [az_cmd, "account", "get-access-token", "--resource", "https://storage.azure.com", "--output", "json"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            logger.debug("az CLI auth failed (returncode %d): %s", result.returncode, result.stderr.strip())
            return None
        data = json.loads(result.stdout)
        token = data.get("accessToken")
        if token:
            logger.info("Using Azure CLI auth (az login)")
        return token
    except FileNotFoundError:
        logger.debug("az CLI not installed — skipping Azure CLI auth")
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        logger.debug("az CLI auth failed: %s", e)
        return None


def _sync_db_from_azure_blob(
    container_url: str,
    sas_token: str,
    cache_dir: Path,
    bearer_token: str | None = None,
) -> list[Path]:
    """Download all .db blobs from an Azure Blob Storage container.

    Uses the Azure Blob REST API directly — no Azure SDK required.

    Auth priority (first that works wins):
      1. SAS token (if provided)
      2. Bearer token (pre-fetched from az CLI, passed by caller)
      3. No auth (for public containers)

    Args:
        container_url: Full container URL, e.g. https://acct.blob.core.windows.net/container
        sas_token: SAS token string (can be empty for CLI auth or public containers)
        cache_dir: Local directory to store downloaded .db files
        bearer_token: Pre-fetched Azure CLI bearer token (avoids repeated az CLI calls)

    Returns list of downloaded file paths (empty on failure).
    """
    # Normalize: strip trailing slash from URL, strip leading '?' from SAS
    container_url = container_url.rstrip("/")
    sas_token = sas_token.lstrip("?")

    def _make_url(base_url: str) -> str:
        """Append SAS token to URL if available."""
        if sas_token:
            sep = "&" if "?" in base_url else "?"
            return f"{base_url}{sep}{sas_token}"
        return base_url

    def _make_request(url: str) -> urllib.request.Request:
        """Create a request with appropriate auth headers."""
        req = urllib.request.Request(url)
        if bearer_token:
            req.add_header("Authorization", f"Bearer {bearer_token}")
            req.add_header("x-ms-version", "2020-10-02")
        return req

    try:
        # Step 1: List all blobs in the container
        list_url = _make_url(f"{container_url}?restype=container&comp=list")
        req = _make_request(list_url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml_body = resp.read()

        root = ET.fromstring(xml_body)
        # Azure Blob List XML: <EnumerationResults><Blobs><Blob><Name>...</Name></Blob></Blobs></EnumerationResults>
        db_blobs: list[str] = []
        for blob_elem in root.iter("Blob"):
            name_elem = blob_elem.find("Name")
            if name_elem is not None and name_elem.text and name_elem.text.endswith(".db"):
                db_blobs.append(name_elem.text)

        if not db_blobs:
            logger.warning("No .db blobs in Azure container %s", container_url)
            return []

        # Step 2: Download each .db blob
        paths: list[Path] = []
        for blob_name in db_blobs:
            db_path = cache_dir / blob_name
            try:
                blob_url = _make_url(f"{container_url}/{blob_name}")
                req = _make_request(blob_url)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    db_path.write_bytes(resp.read())
                logger.info("Downloaded %s from Azure Blob Storage", blob_name)
                paths.append(db_path)
            except (urllib.error.HTTPError, urllib.error.URLError, OSError, TimeoutError) as e:
                logger.warning("Failed to download %s from Azure: %s", blob_name, e)
        return paths
    except urllib.error.HTTPError as e:
        logger.warning("Could not list Azure container %s: HTTP %d", container_url, e.code)
        return []
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        logger.warning("Could not access Azure container %s: %s", container_url, e)
        return []
    except ET.ParseError as e:
        logger.warning("Invalid XML response from Azure container %s: %s", container_url, e)
        return []


def _collect_sources() -> list[Path]:
    """Collect all DB sources from env vars.

    Supports comma-separated values for multiple sources:
      TESTIGO_RECALL_DB_PATH=local1.db,local2.db
      TESTIGO_RECALL_REPO=org/repo-a,org/repo-b
      TESTIGO_RECALL_AZURE_URL=https://acct.blob.core.windows.net/container
      TESTIGO_RECALL_AZURE_SAS=sv=...&se=...&sp=rl&sig=...

    All backends can be combined — DBs are merged at startup.
    """
    sources: list[Path] = []

    # Local DB paths (comma-separated)
    local = os.environ.get("TESTIGO_RECALL_DB_PATH") or os.environ.get("PR_IMPACT_DB_PATH")
    if local:
        for p in local.split(","):
            p = p.strip()
            if p:
                path = Path(p)
                if path.exists():
                    sources.append(path)
                else:
                    logger.warning("Local DB not found: %s", p)

    # GitHub repos (comma-separated)
    repos = os.environ.get("TESTIGO_RECALL_REPO") or os.environ.get("PR_IMPACT_REPO")
    if repos:
        for repo in repos.split(","):
            repo = repo.strip()
            if not repo:
                continue
            cache_dir = Path.home() / ".testigo-recall" / repo.replace("/", "--")
            cache_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Syncing knowledge base from %s ...", repo)
            downloaded = _sync_db_from_github(repo, cache_dir)
            if downloaded:
                sources.extend(downloaded)
            else:
                # Fallback: use any cached .db files in the directory
                cached = sorted(cache_dir.glob("*.db"))
                if cached:
                    logger.info("Using %d cached DB(s) from %s", len(cached), cache_dir)
                    sources.extend(cached)
                else:
                    logger.warning("No knowledge base available for %s", repo)

    # Azure Blob Storage (comma-separated URLs, shared SAS token)
    azure_urls = os.environ.get("TESTIGO_RECALL_AZURE_URL")
    azure_sas = os.environ.get("TESTIGO_RECALL_AZURE_SAS", "")
    if azure_urls:
        # Pre-fetch bearer token once for all Azure URLs (avoid repeated az CLI calls)
        azure_bearer: str | None = None
        if not azure_sas:
            azure_bearer = _get_azure_bearer_token()

        for url in azure_urls.split(","):
            url = url.strip()
            if not url:
                continue
            # Derive cache dir from URL: https://acct.blob.core.windows.net/container -> azure--acct--container
            try:
                parsed = urllib.parse.urlparse(url)
                account = parsed.hostname.split(".")[0] if parsed.hostname else "unknown"
                container = parsed.path.strip("/").split("/")[0] if parsed.path else "default"
                cache_key = f"azure--{account}--{container}"
            except Exception:
                # Stable fallback: use URL without scheme as cache key
                cache_key = f"azure--{url.replace('://', '--').replace('/', '--')}"

            cache_dir = Path.home() / ".testigo-recall" / cache_key
            cache_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Syncing knowledge base from Azure Blob Storage: %s ...", url)
            downloaded = _sync_db_from_azure_blob(url, azure_sas, cache_dir, bearer_token=azure_bearer)
            if downloaded:
                sources.extend(downloaded)
            else:
                # Fallback: use any cached .db files in the directory
                cached = sorted(cache_dir.glob("*.db"))
                if cached:
                    logger.info("Using %d cached DB(s) from %s", len(cached), cache_dir)
                    sources.extend(cached)
                else:
                    logger.warning("No knowledge base available from Azure: %s", url)

    return sources


def _migrate_connection(conn: sqlite3.Connection) -> None:
    """Bring a raw connection's schema up-to-date (symbols col, FTS, repo_dependencies)."""
    c = conn.cursor()

    # Add symbols column if missing
    cols = {r[1] for r in c.execute("PRAGMA table_info(facts)").fetchall()}
    if "symbols" not in cols:
        c.execute("ALTER TABLE facts ADD COLUMN symbols TEXT NOT NULL DEFAULT '[]'")

    # Rebuild FTS index if it's missing the symbols column
    row = c.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='facts_fts'"
    ).fetchone()
    if row and "symbols" not in (row[0] or ""):
        c.executescript("""
            DROP TRIGGER IF EXISTS facts_ai;
            DROP TRIGGER IF EXISTS facts_ad;
            DROP TABLE IF EXISTS facts_fts;
        """)
        conn.executescript(_SCHEMA)
        c.execute(
            "INSERT INTO facts_fts(rowid, summary, detail, symbols) "
            "SELECT id, summary, detail, symbols FROM facts"
        )

    # Add triggered_by column to pr_analyses if missing
    pa_cols = {r[1] for r in c.execute("PRAGMA table_info(pr_analyses)").fetchall()}
    if "triggered_by" not in pa_cols:
        c.execute("ALTER TABLE pr_analyses ADD COLUMN triggered_by TEXT DEFAULT NULL")

    # Ensure repo_dependencies table exists (base DB may predate this feature)
    c.execute(
        "CREATE TABLE IF NOT EXISTS repo_dependencies ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "from_repo TEXT NOT NULL, "
        "to_repo TEXT NOT NULL, "
        "manifest TEXT, "
        "raw_import TEXT, "
        "relation TEXT DEFAULT 'depends_on', "
        "UNIQUE(from_repo, to_repo, raw_import))"
    )

    # Ensure repo_summaries table exists
    c.execute(
        "CREATE TABLE IF NOT EXISTS repo_summaries ("
        "repo TEXT PRIMARY KEY, "
        "summary TEXT NOT NULL, "
        "updated_at TEXT NOT NULL)"
    )


def _merge_into(target: sqlite3.Connection, source_path: Path) -> int:
    """Merge all data from source DB into target. Returns facts merged."""
    target.execute("ATTACH DATABASE ? AS src", (str(source_path),))

    # Replace pr_analyses (natural PK handles conflicts)
    # Use explicit columns to handle schema mismatches (older DBs may lack triggered_by)
    src_pa_cols = {r[1] for r in target.execute("PRAGMA src.table_info(pr_analyses)").fetchall()}
    triggered_by_expr = "triggered_by" if "triggered_by" in src_pa_cols else "NULL"
    target.execute(
        "INSERT OR REPLACE INTO pr_analyses (pr_id, repo, timestamp, files, kinds, triggered_by) "
        f"SELECT pr_id, repo, timestamp, files, kinds, {triggered_by_expr} FROM src.pr_analyses"
    )

    # Clear facts/deps for modules we're importing (avoid duplicates)
    target.execute(
        "DELETE FROM facts WHERE EXISTS ("
        "  SELECT 1 FROM src.pr_analyses s "
        "  WHERE s.pr_id = facts.pr_id AND s.repo = facts.repo"
        ")"
    )
    target.execute(
        "DELETE FROM dependencies WHERE EXISTS ("
        "  SELECT 1 FROM src.pr_analyses s "
        "  WHERE s.pr_id = dependencies.pr_id AND s.repo = dependencies.repo"
        ")"
    )

    # Insert facts (skip id — autoincrement + FTS triggers handle it)
    # Check if source DB has symbols column (older DBs may not)
    src_cols = {r[1] for r in target.execute("PRAGMA src.table_info(facts)").fetchall()}
    symbols_expr = "COALESCE(symbols, '[]')" if "symbols" in src_cols else "'[]'"
    count = target.execute(
        "INSERT INTO facts (pr_id, repo, category, summary, detail, confidence, source, source_files, symbols) "
        "SELECT pr_id, repo, category, summary, detail, confidence, source, source_files, "
        f"{symbols_expr} FROM src.facts"
    ).rowcount

    # Insert dependencies
    target.execute(
        "INSERT INTO dependencies (pr_id, repo, from_component, to_component, relation) "
        "SELECT pr_id, repo, from_component, to_component, relation "
        "FROM src.dependencies"
    )

    # Merge repo summaries (newer wins)
    try:
        target.execute(
            "INSERT OR REPLACE INTO repo_summaries "
            "SELECT * FROM src.repo_summaries"
        )
    except sqlite3.OperationalError:
        pass  # Source DB may not have repo_summaries table yet

    # Merge repo dependencies
    try:
        target.execute(
            "INSERT OR IGNORE INTO repo_dependencies "
            "(from_repo, to_repo, manifest, raw_import, relation) "
            "SELECT from_repo, to_repo, manifest, raw_import, relation "
            "FROM src.repo_dependencies"
        )
    except sqlite3.OperationalError:
        pass  # Source DB may not have repo_dependencies table yet

    # Commit before detach — SQLite requires no open transactions
    target.commit()
    target.execute("DETACH DATABASE src")
    return count


def _resolve_db_path() -> str | None:
    """Determine the DB path, supporting multiple sources.

    If only one source exists, it's used directly.
    If multiple sources exist, they're merged into a temp DB.
    """
    sources = _collect_sources()

    if not sources:
        return None

    if len(sources) == 1:
        return str(sources[0])

    # Multiple sources — merge into a PID-scoped temp DB to avoid
    # conflicts when multiple MCP server instances run simultaneously
    merged_path = Path(tempfile.gettempdir()) / f"testigo-recall-merged-{os.getpid()}.db"

    # Clean up stale merged DBs from dead processes
    for stale in Path(tempfile.gettempdir()).glob("testigo-recall-merged-*.db"):
        if stale == merged_path:
            continue
        try:
            stale.unlink()
        except OSError:
            pass  # still in use by another live process

    # Copy first source as base (preserves schema + FTS)
    shutil.copy2(sources[0], merged_path)
    logger.info("Base DB: %s", sources[0])

    conn = sqlite3.connect(str(merged_path), timeout=10)
    conn.row_factory = sqlite3.Row

    # Migrate base DB schema — the copied DB may predate newer columns/tables.
    # Must run BEFORE merging so INSERT targets have all expected columns.
    _migrate_connection(conn)
    conn.commit()
    try:
        for src in sources[1:]:
            count = _merge_into(conn, src)
            logger.info("Merged %d facts from %s", count, src)
    finally:
        conn.close()

    logger.info("Merged %d sources into %s", len(sources), merged_path)
    return str(merged_path)


def _get_db() -> Database:
    """Lazy-init the database connection."""
    global _db
    if _db is None:
        _db = Database(_resolve_db_path())
    return _db


def _build_catalog() -> str:
    """Build repo catalog string from DB summaries.

    Called at module level to inject into FastMCP instructions.
    Returns empty string if no summaries exist yet.
    """
    try:
        db = _get_db()
        summaries = db.get_repo_summaries()
        if not summaries:
            return ""
        lines = []
        c = db._conn.cursor()
        for s in summaries:
            row = c.execute(
                "SELECT COUNT(*) as cnt FROM facts WHERE repo = ?",
                (s["repo"],),
            ).fetchone()
            count = row["cnt"] if row else 0
            lines.append(f"- {s['repo']} ({count} facts): {s['summary']}")
        return (
            "\n\nAVAILABLE REPOSITORIES:\n"
            + "\n".join(lines)
            + "\nUse this catalog to decide which repo to search for a given question."
        )
    except Exception:
        return ""


mcp = FastMCP(
    "testigo-recall",
    instructions=_STATIC_INSTRUCTIONS + _build_catalog(),
)


@mcp.tool()
def search_codebase(
    query: str,
    category: str | None = None,
    min_confidence: float = 0.0,
    limit: int = 20,
    repo_name: str | None = None,
) -> str:
    """Search the codebase knowledge base for facts about what the code does,
    how it's built, and what it assumes.

    Use this FIRST before reading source files. It returns pre-extracted facts
    ranked by relevance, saving significant time and tokens.

    MULTI-QUERY: Use semicolons to search multiple keyword groups in one call.
    Example: "payment gateway; checkout flow; stripe webhooks"
    This runs 3 searches, deduplicates, and returns combined results.
    ALWAYS batch related searches into one call — this is dramatically cheaper.

    Args:
        query: Search keywords (e.g. "authentication", "payment flow", "database connection")
            Use semicolons to batch multiple searches: "auth login; session JWT; middleware"
        category: Optional filter — "behavior" (what it does), "design" (how it's built), or "assumption" (what it expects)
        min_confidence: Minimum confidence threshold 0.0-1.0 (default: 0.0)
        limit: Max results per query (default: 20). With batched queries, total results can be up to limit × number of queries (max 70).
        repo_name: Optional filter to scope search to a specific repository
    """
    # Input validation
    limit = max(1, min(limit, 100))
    min_confidence = max(0.0, min(min_confidence, 1.0))

    db = _get_db()

    # Split on semicolons for multi-query support
    queries = [q.strip() for q in query.split(";") if q.strip()]
    if not queries:
        return "Empty query. Provide search keywords."

    if len(queries) == 1:
        # Single query — standard path
        results = db.search(queries[0], category=category, min_confidence=min_confidence, limit=limit, repo_name=repo_name)
    else:
        # Multi-query: run each, deduplicate, combine
        seen: set[tuple] = set()
        results: list[dict] = []
        for q in queries:
            hits = db.search(q, category=category, min_confidence=min_confidence, limit=limit, repo_name=repo_name)
            for fact in hits:
                key = (fact.get("pr_id"), fact.get("category"), fact.get("summary"))
                if key not in seen:
                    seen.add(key)
                    results.append(fact)
        # Cap total results — hard ceiling prevents output-too-large (~50KB)
        # regardless of how many semicolon queries are batched
        max_total = min(limit * len(queries), 70)
        results = results[:max_total]

    if not results:
        return f"No facts found for '{query}'. Try broader keywords or remove the category filter."

    # Collapse near-duplicate facts (e.g. same config across country layers)
    results = _dedup_similar_facts(results)

    payload = json.dumps(_clean_facts(results))
    # Nudge the agent to state what it still needs before calling again
    return payload + "\n\n⚠ Before making another call, tell the user what you still need to look up and why."


@mcp.tool()
def get_module_facts(module_id: str) -> str:
    """Get all extracted facts for a specific module.

    Module IDs look like "SCAN:backend/app/api".
    Use search_codebase first to discover module IDs, then use this
    for a deep dive into a specific module.

    Args:
        module_id: The module identifier (e.g. "SCAN:backend/app/api/simplified")
    """
    db = _get_db()
    facts = db.get_facts_by_module(module_id)
    if not facts:
        return f"No facts found for module '{module_id}'. Use search_codebase to find valid module IDs."
    payload = json.dumps(_clean_facts(facts))
    return payload + "\n\n⚠ Before making another call, tell the user what you still need to look up and why."


@mcp.tool()
def get_recent_changes(
    category: str | None = None,
    limit: int = 10,
) -> str:
    """Get the most recently extracted facts across the entire codebase.

    Useful for understanding what changed recently or getting an overview
    of the codebase.

    Args:
        category: Optional filter — "behavior", "design", or "assumption"
        limit: Number of recent facts to return (default: 10)
    """
    # Input validation
    limit = max(1, min(limit, 100))
    valid_categories = {"behavior", "design", "assumption"}
    if category and category not in valid_categories:
        return f"Invalid category '{category}'. Must be one of: {', '.join(sorted(valid_categories))}."

    db = _get_db()
    facts = db.get_recent_facts(category=category, limit=limit)
    if not facts:
        if category:
            return f"No facts found for category '{category}'."
        return "No facts in the knowledge base yet. Run 'testigo-recall scan' first."
    payload = json.dumps(_clean_facts(facts))
    return payload + "\n\n⚠ Before making another call, tell the user what you still need to look up and why."


@mcp.tool()
def get_component_impact(component_name: str) -> str:
    """Find all modules and PRs where a specific component (file/service) appears.

    Use this to understand the blast radius of changes to a component —
    what depends on it and what it depends on.

    Args:
        component_name: File path or service name (e.g. "api_service.py", "backend/app/auth")
    """
    # Input validation
    if not component_name or not component_name.strip():
        return "component_name is required. Provide a file path or service name (e.g. 'api_service.py', 'backend/app/auth')."

    db = _get_db()
    impacts = db.get_component_impact(component_name)
    if not impacts:
        return f"No dependency data found for '{component_name}'. Try a shorter name or file path."
    payload = json.dumps(impacts)
    return payload + "\n\n⚠ Before making another call, tell the user what you still need to look up and why."


@mcp.tool()
def list_modules(repo_name: str | None = None) -> str:
    """List scanned modules in the knowledge base.

    Without repo_name: returns a compact summary of repos with module/fact counts.
    With repo_name: returns the full list of modules for that specific repo.

    Always call without repo_name first to discover available repos, then call
    again with repo_name to get the module list for a specific repo.

    Args:
        repo_name: Repository name — pass this to get the full module list for one repo
    """
    db = _get_db()
    c = db._conn.cursor()

    if repo_name:
        rows = c.execute(
            "SELECT pr_id, repo, COUNT(*) as fact_count "
            "FROM facts WHERE repo = ? GROUP BY pr_id, repo ORDER BY pr_id",
            (repo_name,),
        ).fetchall()
        if not rows:
            return f"No modules found for repo '{repo_name}'."
        modules = [{"module_id": r["pr_id"], "repo": r["repo"], "fact_count": r["fact_count"]} for r in rows]
        return json.dumps(modules)

    # No filter — return compact repo summary instead of every module
    rows = c.execute(
        "SELECT repo, COUNT(DISTINCT pr_id) as modules, COUNT(*) as facts "
        "FROM facts GROUP BY repo ORDER BY repo",
    ).fetchall()
    if not rows:
        return "No modules found. Run 'testigo-recall scan' to populate the knowledge base."

    # Include repo summaries if available
    summaries_map: dict[str, str] = {}
    try:
        for s in db.get_repo_summaries():
            summaries_map[s["repo"]] = s["summary"]
    except Exception:
        pass

    repos = [
        {
            "repo": r["repo"],
            "modules": r["modules"],
            "facts": r["facts"],
            **({"summary": summaries_map[r["repo"]]} if r["repo"] in summaries_map else {}),
        }
        for r in rows
    ]
    return json.dumps(repos)


@mcp.tool()
def get_repo_dependencies(
    repo_name: str | None = None,
    direction: str = "both",
) -> str:
    """Get cross-repo dependency graph showing which repos depend on each other.

    Use this to understand the blast radius of changes across repositories.
    Data comes from package manifests (go.mod, package.json), not code analysis.

    Args:
        repo_name: Filter to a specific repo. Without this, returns entire graph.
        direction: "outgoing" (what this repo depends on), "incoming" (what depends on this repo), "both"
    """
    valid_directions = {"outgoing", "incoming", "both"}
    if direction not in valid_directions:
        return f"Invalid direction '{direction}'. Must be one of: {', '.join(sorted(valid_directions))}."

    db = _get_db()
    deps = db.get_repo_dependencies(repo_name=repo_name, direction=direction)
    if not deps:
        if repo_name:
            return f"No cross-repo dependencies found for '{repo_name}'."
        return "No cross-repo dependencies in the knowledge base. Run 'testigo-recall deps' to populate."

    return json.dumps(deps)


def main() -> None:
    """Entry point for the testigo-recall-mcp command."""
    mcp.run(transport="stdio")
