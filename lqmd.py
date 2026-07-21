#!/usr/bin/env python3
"""
lqmd - qmd-style local search over a Logseq DB-version graph.

Source of truth is the graph itself, read through the logseq CLI. No markdown
files are written or read. The index lives in one SQLite file holding an FTS5
keyword index and a sqlite-vec vector index. Embeddings are computed in-process
with fastembed (ONNX Runtime, CPU); no embedding daemon is required.

Pipeline (v1): keyword (BM25), vector (cosine), and hybrid (RRF fusion of both).
Query expansion and LLM reranking are deliberately left for a later layer.

Indexing is incremental by default: each run re-pulls the graph and re-chunks it,
but only embeds chunks whose content has changed since the last run. --full
forces a clean rebuild from scratch.

Commands:
    lqmd probe                 inspect what the graph extraction returns (run this first)
    lqmd index [--full]        pull the graph, chunk, embed changed chunks, build the index
    lqmd search  "<query>"     BM25 keyword search only (fast, no embedding model)
    lqmd vsearch "<query>"     vector semantic search only
    lqmd query   "<query>"     hybrid: BM25 + vector, fused by RRF (best quality)
    lqmd get <page|#docid>     print an indexed page's text
    lqmd status                index health

Common flags: -n <num>, --json, --graph <name>, --db <path>.
"""

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time

import sqlite3

try:
    import sqlite_vec
except ImportError:
    sys.exit("sqlite-vec is not installed. Run: pip install sqlite-vec")

GRAPH        = os.environ.get("LQMD_GRAPH", "vault")
LOGSEQ_ROOT  = os.environ.get("LQMD_ROOT_DIR", "")

DB_PATH      = os.environ.get("LQMD_DB", os.path.expanduser("~/.cache/lqmd/index.sqlite"))
LOGSEQ_BIN   = os.environ.get("LQMD_LOGSEQ", "logseq")
QUERY_TIMEOUT_MS = os.environ.get("LQMD_QUERY_TIMEOUT_MS", "120000")

EMBED_MODEL  = os.environ.get("LQMD_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
MODEL_CACHE  = os.environ.get("LQMD_MODEL_CACHE",
                              os.path.expanduser("~/.local/share/lqmd/models"))
QUERY_PREFIX = os.environ.get("LQMD_QUERY_PREFIX",
                              "Represent this sentence for searching relevant passages: ")
DOC_PREFIX   = os.environ.get("LQMD_DOC_PREFIX", "")

FUSE_K       = 60
CANDIDATES   = 150

EXCLUDE_PAGE_PREFIXES = ["$$$"]

BLOCKS_QUERY = """
[:find (pull ?b [:db/id :block/uuid :block/title :block/order :block/updated-at
                 {:block/page [:db/id :block/uuid :block/name :block/title
                               {:block/tags [:block/title]}]}
                 {:block/parent [:db/id]}])
 :where
 [?b :block/page ?pg]
 [?b :block/title ?bt]]
"""

UUID_TITLES_QUERY = """
[:find (pull ?e [:block/uuid :block/title :block/name])
 :where [?e :block/uuid _]]
"""

TAG_BLOCKS_QUERY = """
[:find ?u
 :in $ ?name
 :where
 [?t :block/name ?name]
 (or-join [?b ?t]
   [?b :block/tags ?t]
   (and [?b :block/page ?pg] [?pg :block/tags ?t]))
 [?b :block/uuid ?u]]
"""

REF_BLOCKS_QUERY = """
[:find ?u
 :in $ ?name
 :where
 [?p :block/name ?name]
 [?b :block/refs ?p]
 [?b :block/uuid ?u]]
"""

REF_DESC_QUERY = """
[:find ?u
 :in $ ?name
 :where
 [?p :block/name ?name]
 [?a :block/refs ?p]
 (or-join [?a ?b]
   [?b :block/parent ?a]
   (and [?b :block/parent ?c1] [?c1 :block/parent ?a])
   (and [?b :block/parent ?c1] [?c1 :block/parent ?c2] [?c2 :block/parent ?a]))
 [?b :block/uuid ?u]]
"""

def g(d, *names, default=None):
    """Return the first present key among candidate names."""
    if not isinstance(d, dict):
        return default
    for n in names:
        if n in d:
            return d[n]
    return default

def block_text(b):   return g(b, "block/title", ":block/title", "title", default="") or ""
def block_id(b):     return g(b, "db/id", ":db/id", "id")
def block_uuid(b):   return g(b, "block/uuid", ":block/uuid", "uuid")
def block_order(b):  return g(b, "block/order", ":block/order", "order", default="")
def block_updated(b):return g(b, "block/updated-at", ":block/updated-at", "updated-at")
def block_parent(b):
    p = g(b, "block/parent", ":block/parent", "parent")
    return block_id(p) if isinstance(p, dict) else None
def block_page(b):   return g(b, "block/page", ":block/page", "page", default={})

def page_uuid(p):    return g(p, "block/uuid", ":block/uuid", "uuid")
def page_name(p):    return g(p, "block/name", ":block/name", "name", default="")
def page_title(p):   return g(p, "block/title", ":block/title", "title", default="")
def page_id(p):      return g(p, "db/id", ":db/id", "id")
def page_tags(p):
    tags = g(p, "block/tags", ":block/tags", "tags", default=[]) or []
    out = []
    for t in tags:
        title = g(t, "block/title", ":block/title", "title")
        if title:
            out.append(title)
    return out

def run_logseq_query(edn, graph, inputs=None, fatal=True):
    """Run a Datalog query through the logseq CLI. fatal=True (indexing) exits
    the process on failure; fatal=False (query-time use inside the server) raises
    RuntimeError so the caller can degrade gracefully. inputs is an EDN string
    for the query's :in parameters."""
    cmd = [LOGSEQ_BIN, "query", "--query", edn, "--graph", graph,
           "--output", "json", "--timeout-ms", QUERY_TIMEOUT_MS]
    if inputs is not None:
        cmd.extend(["--inputs", inputs])
    if LOGSEQ_ROOT:
        cmd.extend(["--root-dir", LOGSEQ_ROOT])
    def fail(msg):
        if fatal:
            sys.exit(msg)
        raise RuntimeError(msg)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        return fail(f"Could not find the {LOGSEQ_BIN} binary. Set LQMD_LOGSEQ.")
    except PermissionError:
        return fail(f"Cannot execute {LOGSEQ_BIN} (permission denied). Set LQMD_LOGSEQ "
                    f"to the absolute path, and check the binary is executable and not on "
                    f"a noexec mount.")
    except subprocess.CalledProcessError as e:
        return fail(f"logseq query failed:\n{e.stderr or e.stdout}")
    except OSError as e:
        return fail(f"Could not run {LOGSEQ_BIN}: {e}")
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return fail(f"Could not parse logseq output as JSON. First 400 chars:\n{out.stdout[:400]}")
    if isinstance(data, dict):
        data = data.get("data", data)
        data = data.get("result", data) if isinstance(data, dict) else data
    return data

def fetch_blocks(graph):
    rows = run_logseq_query(BLOCKS_QUERY, graph)
    blocks = []
    for row in rows:
        b = row[0] if isinstance(row, list) and row else row
        if isinstance(b, dict):
            blocks.append(b)
    return blocks

_UUID = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
_REF_RE = re.compile(r"\[\[(%s)\]\]|\(\((%s)\)\)" % (_UUID, _UUID))

def make_resolver(uuid_titles):
    """Return a function that rewrites [[uuid]] / ((uuid)) tokens to the
    referenced entity's title. Unknown UUIDs are left untouched."""
    if not uuid_titles:
        return lambda s: s
    def resolve(s):
        if "[[" not in s and "((" not in s:
            return s
        return _REF_RE.sub(
            lambda m: uuid_titles.get((m.group(1) or m.group(2)).lower(), m.group(0)), s)
    return resolve

def fetch_uuid_titles(graph):
    """uuid -> display title for every titled entity, so references resolve.
    Titles that themselves contain references are resolved one level deep."""
    rows = run_logseq_query(UUID_TITLES_QUERY, graph)
    out = {}
    for row in rows:
        e = row[0] if isinstance(row, list) and row else row
        if not isinstance(e, dict):
            continue
        u = block_uuid(e)
        if not u:
            continue
        title = (g(e, "block/title", ":block/title", "title")
                 or g(e, "block/name", ":block/name", "name"))
        if title:
            out[str(u).lower()] = title
    resolve = make_resolver(out)
    return {u: resolve(t) for u, t in out.items()}

def excluded(name):
    n = (name or "").lower()
    return any(n.startswith(pfx.lower()) for pfx in EXCLUDE_PAGE_PREFIXES)

def build_pages(blocks):
    """Group blocks into pages keyed by page uuid; keep enough to rebuild the tree."""
    pages = {}
    for b in blocks:
        p = block_page(b)
        puuid = page_uuid(p)
        if not puuid:
            continue
        if excluded(page_name(p)):
            continue
        page = pages.get(puuid)
        if page is None:
            page = {
                "uuid": puuid,
                "name": page_name(p),
                "title": page_title(p) or page_name(p),
                "page_id": page_id(p),
                "tags": page_tags(p),
                "blocks": {},
                "updated": 0,
            }
            pages[puuid] = page
        bid = block_id(b)
        if bid is None:
            continue
        page["blocks"][bid] = b
        ts = block_updated(b)
        if isinstance(ts, (int, float)) and ts > page["updated"]:
            page["updated"] = int(ts)
    return pages

def _page_children(page):
    blocks = page["blocks"]
    children = {}
    for bid, b in blocks.items():
        children.setdefault(block_parent(b), []).append(bid)
    for sibs in children.values():
        sibs.sort(key=lambda i: (str(block_order(blocks[i])), i))
    roots = children.get(page["page_id"], [])
    if not roots:
        roots = [bid for bid, b in blocks.items() if block_parent(b) not in blocks]
        roots.sort(key=lambda i: (str(block_order(blocks[i])), i))
    return children, roots

def render_page(page):
    """Rebuild the page outline as indented text from parent/order, depth-first."""
    blocks = page["blocks"]
    children, roots = _page_children(page)
    lines, seen = [], set()
    def walk(bid, depth):
        if bid in seen:
            return
        seen.add(bid)
        text = block_text(blocks[bid]).strip()
        if text:
            lines.append(("  " * depth) + "- " + text)
        for c in children.get(bid, []):
            walk(c, depth + 1)
    for r in roots:
        walk(r, 0)
    return "\n".join(lines)

def iter_blocks(page, uuid_titles):
    """One record per block that carries text, in document order. Each block
    keeps its own text for display plus the parent-block trail for context, with
    references resolved to titles. Filtering by tags/references is done at query
    time against the live graph, not from the index."""
    blocks = page["blocks"]
    children, roots = _page_children(page)
    resolve = make_resolver(uuid_titles)
    out, seen = [], set()
    def walk(bid, depth, trail):
        if bid in seen:
            return
        seen.add(bid)
        b = blocks[bid]
        text = resolve(block_text(b).strip())
        next_trail = trail
        if text:
            buid = block_uuid(b) or f"id:{block_id(b)}"
            out.append({
                "block_uuid": buid,
                "depth": depth,
                "breadcrumb": " ▸ ".join(c[:60] for c in trail[-3:]),
                "body": text,
            })
            next_trail = trail + [text]
        for c in children.get(bid, []):
            walk(c, depth + 1, next_trail)
    for r in roots:
        walk(r, 0, [])
    return out

def docid_for(page_uuid, block_uuid):
    h = hashlib.sha1(f"{page_uuid}\x00{block_uuid}".encode()).hexdigest()
    return h[:6]

def content_hash(page_uuid, block_uuid, page_title, breadcrumb, body):
    """Stable identity of an indexed block. The hash changes (forcing the row to
    be rewritten) iff the block's text, parent trail, or embedded page title
    change; an unchanged block keeps its hash across runs."""
    parts = "\x00".join([page_uuid or "", block_uuid or "", page_title or "",
                         breadcrumb or "", body or ""])
    return hashlib.sha1(parts.encode()).hexdigest()[:16]

def block_embed_text(rec):
    """Text actually embedded and keyword-indexed for a block: the page title
    and parent trail give a short block the context it needs to be found."""
    parts = [f"title: {rec['page_title']}"]
    if rec["breadcrumb"]:
        parts.append(rec["breadcrumb"])
    parts.append(rec["body"])
    return "\n".join(parts)

_EMBEDDER = None

def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from fastembed import TextEmbedding
        except ImportError:
            sys.exit("fastembed is not installed. Run: pip install fastembed")
        os.makedirs(MODEL_CACHE, exist_ok=True)
        _EMBEDDER = TextEmbedding(model_name=EMBED_MODEL, cache_dir=MODEL_CACHE)
    return _EMBEDDER

def embed_texts(texts, is_query=False):
    prefix = QUERY_PREFIX if is_query else DOC_PREFIX
    model = _get_embedder()
    return [v.tolist() for v in model.embed([prefix + t for t in texts])]

def vec_blob(v):
    return struct.pack(f"{len(v)}f", *v)

def connect(db_path, read_only=False, check_same_thread=True):
    if read_only:
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True,
                             check_same_thread=check_same_thread)
    else:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        db = sqlite3.connect(db_path, check_same_thread=check_same_thread)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    if not read_only:
        db.execute("PRAGMA journal_mode=WAL")
    return db

def meta_get(db, key, default=None):
    if not table_exists(db, "meta"):
        return default
    row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else default

def meta_set(db, key, value):
    db.execute("INSERT INTO meta(key,value) VALUES(?,?) "
               "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))

def init_schema(db, dim):
    db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
    db.execute("""CREATE TABLE IF NOT EXISTS docs(
        rowid INTEGER PRIMARY KEY,
        docid TEXT, chash TEXT, page_uuid TEXT, page_name TEXT, page_title TEXT,
        block_uuid TEXT, breadcrumb TEXT, depth INTEGER, seq INTEGER,
        context TEXT, body TEXT)""")
    db.execute("CREATE INDEX IF NOT EXISTS docs_chash ON docs(chash)")
    db.execute("CREATE INDEX IF NOT EXISTS docs_page ON docs(page_uuid)")
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(text, "
               "tokenize='porter unicode61')")
    db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec USING "
               f"vec0(rowid INTEGER PRIMARY KEY, emb float[{dim}] distance_metric=cosine)")

def table_exists(db, name):
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?",
        (name,)).fetchone() is not None

def column_exists(db, table, col):
    if not table_exists(db, table):
        return False
    return any(r[1] == col for r in db.execute(f"PRAGMA table_info({table})"))

def existing_chunks(db):
    """Map content hash -> rowid for what is already indexed. Empty if the index
    is absent or predates the chash column."""
    if not column_exists(db, "docs", "chash"):
        return {}
    return {ch: rid for ch, rid in
            db.execute("SELECT chash, rowid FROM docs WHERE chash IS NOT NULL")}

def drop_index_tables(db):
    for t in ("docs", "docs_fts", "vec"):
        if table_exists(db, t):
            db.execute(f"DROP TABLE {t}")
    db.commit()

def cmd_probe(args):
    blocks = fetch_blocks(args.graph)
    print(f"blocks returned: {len(blocks)}")
    if not blocks:
        print("No blocks. Check the graph name and the Datalog in BLOCKS_QUERY.")
        return
    print("\n--- raw JSON of first block (confirm the key names) ---")
    print(json.dumps(blocks[0], indent=2, ensure_ascii=False)[:1200])
    has_updated = any(block_updated(b) is not None for b in blocks[:200])
    print(f"\n:block/updated-at present in sample: {has_updated} "
          f"(not required; incremental indexing keys on content hash, not timestamps)")
    pages = build_pages(blocks)
    print(f"pages after grouping/exclusion: {len(pages)}")
    sample = list(pages.values())[:8]
    print("\n--- sample page names ---")
    for p in sample:
        print(f"  {p['name']!r}  title={p['title']!r}  blocks={len(p['blocks'])}  tags={p['tags']}")
    if sample:
        first = sample[0]
        print(f"\n--- rendered text of page {first['name']!r} ---")
        print(render_page(first)[:1000])

def cmd_index(args):
    blocks = fetch_blocks(args.graph)
    pages = build_pages(blocks)
    if not pages:
        sys.exit("Nothing to index. Run lqmd probe to see what the graph returns.")

    uuid_titles = fetch_uuid_titles(args.graph)
    desired = {}
    for p in pages.values():
        context = ", ".join(p["tags"]) if p["tags"] else ""
        for seq, blk in enumerate(iter_blocks(p, uuid_titles)):
            ch = content_hash(p["uuid"], blk["block_uuid"], p["title"],
                              blk["breadcrumb"], blk["body"])
            desired[ch] = {
                "docid": docid_for(p["uuid"], blk["block_uuid"]),
                "chash": ch, "page_uuid": p["uuid"], "page_name": p["name"],
                "page_title": p["title"], "block_uuid": blk["block_uuid"],
                "breadcrumb": blk["breadcrumb"], "depth": blk["depth"], "seq": seq,
                "context": context, "body": blk["body"],
            }
    if not desired:
        sys.exit("Pages found but no text to index.")

    db = connect(args.db)

    if args.full:
        drop_index_tables(db)
        existing = {}
    elif table_exists(db, "docs") and not column_exists(db, "docs", "block_uuid"):
        print("existing index predates block-level support; rebuilding in full.")
        drop_index_tables(db)
        existing = {}
    else:
        existing = existing_chunks(db)

    desired_hashes  = set(desired)
    existing_hashes = set(existing)
    to_add    = [desired[ch] for ch in desired_hashes - existing_hashes]
    to_delete = [existing[ch] for ch in existing_hashes - desired_hashes]
    kept      = len(desired_hashes & existing_hashes)

    new_embeddings, dim = [], None
    if to_add:
        print(f"new/changed blocks: {len(to_add)}  (unchanged {kept}, removing {len(to_delete)})  "
              f"embedding via {EMBED_MODEL} ...")
        texts = [block_embed_text(r) for r in to_add]
        BATCH = 64
        for i in range(0, len(texts), BATCH):
            new_embeddings.extend(embed_texts(texts[i:i+BATCH]))
            print(f"  embedded {min(i+BATCH, len(texts))}/{len(texts)}", end="\r")
        print()
        dim = len(new_embeddings[0])
    else:
        print(f"no new or changed blocks (unchanged {kept}, removing {len(to_delete)})")

    stored_dim = meta_get(db, "dim")
    if dim is None:
        if stored_dim is None:
            print("nothing to index.")
            return
        dim = int(stored_dim)
    elif stored_dim is not None and not args.full and int(stored_dim) != dim:
        sys.exit(f"Embedding dim changed ({stored_dim} -> {dim}). "
                 f"Vectors are not cross-compatible; rebuild with --full.")

    init_schema(db, dim)

    if to_delete:
        del_rows = [(rid,) for rid in to_delete]
        db.executemany("DELETE FROM docs WHERE rowid=?", del_rows)
        db.executemany("DELETE FROM docs_fts WHERE rowid=?", del_rows)
        db.executemany("DELETE FROM vec WHERE rowid=?", del_rows)

    (next_rowid,) = db.execute("SELECT COALESCE(MAX(rowid),0)+1 FROM docs").fetchone()
    doc_rows, fts_rows, vec_rows = [], [], []
    for rec, emb in zip(to_add, new_embeddings):
        rid = next_rowid
        next_rowid += 1
        doc_rows.append((rid, rec["docid"], rec["chash"], rec["page_uuid"], rec["page_name"],
                         rec["page_title"], rec["block_uuid"], rec["breadcrumb"], rec["depth"],
                         rec["seq"], rec["context"], rec["body"]))
        fts_rows.append((rid, block_embed_text(rec)))
        vec_rows.append((rid, vec_blob(emb)))
    db.executemany("INSERT INTO docs(rowid,docid,chash,page_uuid,page_name,page_title,"
                   "block_uuid,breadcrumb,depth,seq,context,body) "
                   "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", doc_rows)
    db.executemany("INSERT INTO docs_fts(rowid, text) VALUES(?,?)", fts_rows)
    db.executemany("INSERT INTO vec(rowid, emb) VALUES(?,?)", vec_rows)

    meta_set(db, "dim", dim)
    meta_set(db, "embed_model", EMBED_MODEL)
    meta_set(db, "graph", args.graph)
    meta_set(db, "indexed_at", int(time.time()))
    meta_set(db, "page_count", len(pages))
    db.commit()
    (total,) = db.execute("SELECT count(*) FROM docs").fetchone()
    print(f"index now holds {total} blocks across {len(pages)} pages "
          f"(+{len(to_add)} -{len(to_delete)}) in {args.db}")

def fts_match_strings(query):
    terms = re.findall(r"\w+", query, flags=re.UNICODE)
    if not terms:
        return [f'"{query}"']
    quoted = [f'"{t}"' for t in terms]
    variants = [" AND ".join(quoted)]
    if len(quoted) > 1:
        variants.append(" OR ".join(quoted))
    return variants

def search_lex(db, query, limit):
    out, seen = [], set()
    for m in fts_match_strings(query):
        if len(out) >= limit:
            break
        for (rid,) in db.execute(
                "SELECT rowid FROM docs_fts WHERE docs_fts MATCH ? "
                "ORDER BY bm25(docs_fts) LIMIT ?", (m, limit)):
            if rid not in seen:
                seen.add(rid)
                out.append(rid)
                if len(out) >= limit:
                    break
    return out

def search_vec(db, query, limit):
    emb = embed_texts([query], is_query=True)[0]
    return [rid for (rid, _d) in db.execute(
        "SELECT rowid, distance FROM vec WHERE emb MATCH ? AND k=? ORDER BY distance",
        (vec_blob(emb), limit)).fetchall()]

def rrf(lists, k=FUSE_K):
    scores = {}
    for lst in lists:
        for rank, rid in enumerate(lst):
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (k + rank + 1)
    return scores

def rank_hits(scored):
    """All scored hits as (score, rowid), highest first, no per-page collapse so
    several blocks from one page can each surface as their own result."""
    return sorted(((sc, rid) for rid, sc in scored.items()), key=lambda x: -x[0])

_TAG_RE = re.compile(r"#\[\[([^\]]+)\]\]|#([^\s#\[\]()]+)")
_PAGE_RE = re.compile(r"\[\[([^\]]+)\]\]")

def parse_constraints(query):
    """Split a query into free text plus #Tag and [[Page]] constraints. Tags are
    stripped first so that #[[Foo]] is read as a tag, not a page reference.
    Names are lowercased to match the graph's :block/name form."""
    tags = [(m.group(1) or m.group(2)).strip().lower()
            for m in _TAG_RE.finditer(query)]
    without_tags = _TAG_RE.sub(" ", query)
    pages = [m.group(1).strip().lower() for m in _PAGE_RE.finditer(without_tags)]
    text = " ".join(_PAGE_RE.sub(" ", without_tags).split()).strip()
    return text, [t for t in tags if t], [p for p in pages if p]

def _block_uuids(query_edn, graph, name):
    rows = run_logseq_query(query_edn, graph, inputs=json.dumps([name]), fatal=False)
    out = set()
    for row in rows or []:
        u = row[0] if isinstance(row, list) and row else row
        if u:
            out.add(str(u).lower())
    return out

def blocks_for_ref(graph, name):
    """Blocks that reference the page, plus blocks nested beneath them (to a few
    levels), so data captured under a "[[Person]] ..." block comes along with
    the referencing block."""
    hits = _block_uuids(REF_BLOCKS_QUERY, graph, name)
    try:
        hits |= _block_uuids(REF_DESC_QUERY, graph, name)
    except RuntimeError:
        pass
    return hits

def constraint_rowids(db, graph, tags, pages):
    """Resolve #Tag / [[Page]] constraints against the live graph via Datalog,
    intersect them (AND), then map the matching block uuids to index rowids.
    Returns the allowed rowid set (empty if nothing matches)."""
    allowed_uuids = None
    for name in tags:
        hits = _block_uuids(TAG_BLOCKS_QUERY, graph, name)
        allowed_uuids = hits if allowed_uuids is None else (allowed_uuids & hits)
    for name in pages:
        hits = blocks_for_ref(graph, name)
        allowed_uuids = hits if allowed_uuids is None else (allowed_uuids & hits)
    if not allowed_uuids:
        return set()
    rowids, uuids = set(), list(allowed_uuids)
    for i in range(0, len(uuids), 800):
        chunk = uuids[i:i + 800]
        placeholders = ",".join("?" * len(chunk))
        rowids.update(rid for (rid,) in db.execute(
            f"SELECT rowid FROM docs WHERE lower(block_uuid) IN ({placeholders})", chunk))
    return rowids

def snippet(body, query, width=240):
    terms = re.findall(r"\w+", query.lower())
    low = body.lower()
    pos = next((low.find(t) for t in terms if low.find(t) >= 0), -1)
    if pos < 0:
        return body[:width].strip()
    start = max(0, pos - width // 3)
    return ("…" if start > 0 else "") + body[start:start + width].strip() + "…"

def shape_rows(db, ranked, query, n):
    """Shape ranked (score, rowid) hits into a list of result dicts. Pure: no
    printing, no I/O beyond the index reads. Shared by the CLI's emit and the
    HTTP server so both surface identical fields from one code path."""
    rows = []
    raw = [sc for sc, _ in ranked[:n]]
    lo, hi = (min(raw), max(raw)) if raw else (0, 1)
    span = hi - lo
    for sc, rid in ranked[:n]:
        docid, pname, ptitle, puuid, buid, crumb, context, body = db.execute(
            "SELECT docid,page_name,page_title,page_uuid,block_uuid,breadcrumb,context,body "
            "FROM docs WHERE rowid=?", (rid,)).fetchone()
        norm = 1.0 if span == 0 else (sc - lo) / span
        rows.append({
            "docid": docid, "page": pname, "title": ptitle, "context": context,
            "page_uuid": puuid, "block_uuid": buid, "breadcrumb": crumb,
            "score": round(norm, 3), "raw": sc,
            "text": body, "snippet": snippet(body, query),
        })
    return rows

def run_search(db, query, n, mode="hybrid", graph=None):
    """Full search path shared by the CLI and the HTTP server. #Tag / [[Page]]
    constraints are resolved against the live graph first (Datalog), then the
    hybrid/keyword/vector search runs over the remaining free text restricted to
    the constrained blocks. With no constraints it is a plain search; with
    constraints but no free text, the constraints alone select the blocks."""
    text, tags, pages = parse_constraints(query)
    result = {"text": text, "tags": tags, "pages": pages, "rows": []}

    allowed = None
    if tags or pages:
        if graph is None:
            graph = meta_get(db, "graph") or GRAPH
        try:
            allowed = constraint_rowids(db, graph, tags, pages)
        except RuntimeError as exc:
            result["error"] = str(exc)
            return result

    if text:
        limit = CANDIDATES if allowed is None else max(CANDIDATES, 400)
        lists = []
        if mode in ("hybrid", "keyword"):
            lists.append(search_lex(db, text, limit))
        if mode in ("hybrid", "vector"):
            lists.append(search_vec(db, text, limit))
        scored = rrf(lists)
        if allowed is not None:
            scored = {rid: sc for rid, sc in scored.items() if rid in allowed}
        ranked = rank_hits(scored)
    elif allowed is not None:
        ranked = [(1.0, rid) for rid in sorted(allowed)]
    else:
        ranked = []

    result["rows"] = shape_rows(db, ranked, text or query, n)
    return result

def emit_rows(result, as_json):
    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if result.get("error"):
        print(f"constraint query failed: {result['error']}")
        return
    rows = result["rows"]
    if not rows:
        print("no results")
        return
    filt = []
    if result["tags"]:
        filt.append(" ".join("#" + t for t in result["tags"]))
    if result["pages"]:
        filt.append(" ".join(f"[[{p}]]" for p in result["pages"]))
    if filt:
        print("filter: " + "  ".join(filt))
    for r in rows:
        crumb = f"  ({r['breadcrumb']})" if r["breadcrumb"] else ""
        print(f"\n{r['page']}{crumb}  #{r['docid']}   score {r['score']:.0%}")
        print(r["snippet"])

def open_index(args):
    if not os.path.exists(args.db):
        sys.exit(f"No index at {args.db}. Run lqmd index first.")
    return connect(args.db)

def cmd_search(args):
    db = open_index(args)
    emit_rows(run_search(db, args.query, args.n, mode="keyword", graph=args.graph), args.json)

def cmd_vsearch(args):
    db = open_index(args)
    emit_rows(run_search(db, args.query, args.n, mode="vector", graph=args.graph), args.json)

def cmd_query(args):
    db = open_index(args)
    emit_rows(run_search(db, args.query, args.n, mode="hybrid", graph=args.graph), args.json)

def get_page_blocks(db, key=None, page_uuid=None):
    """A page's blocks in document order as {depth, text} dicts. Select the page
    by page_uuid, by page name/title, or by '#docid' of any block on it. Returns
    None if nothing matches. The structured form lets the client render the
    outline as a bullet tree."""
    if page_uuid is None and key and key.startswith("#"):
        row = db.execute("SELECT page_uuid FROM docs WHERE docid=? LIMIT 1",
                         (key[1:],)).fetchone()
        if not row:
            return None
        page_uuid = row[0]
    if page_uuid is not None:
        rows = db.execute("SELECT body, depth FROM docs WHERE page_uuid=? ORDER BY seq",
                          (page_uuid,)).fetchall()
    elif key:
        rows = db.execute("SELECT body, depth FROM docs WHERE page_name=? OR page_title=? "
                          "ORDER BY seq", (key, key)).fetchall()
    else:
        return None
    if not rows:
        return None
    return [{"depth": int(d or 0), "text": b} for b, d in rows]

def get_page_text(db, key=None, page_uuid=None):
    """The page as an indented plaintext outline, or None. Shared by the CLI's
    cmd_get and as a fallback rendering of the HTTP /get endpoint."""
    blocks = get_page_blocks(db, key=key, page_uuid=page_uuid)
    if blocks is None:
        return None
    return "\n".join(("  " * b["depth"]) + "- " + b["text"] for b in blocks)

def cmd_get(args):
    db = open_index(args)
    text = get_page_text(db, args.page)
    if text is None:
        sys.exit(f"Not found: {args.page}")
    print(text)

def cmd_status(args):
    db = open_index(args)
    print(f"db:          {args.db}")
    print(f"graph:       {meta_get(db,'graph')}")
    print(f"embed model: {meta_get(db,'embed_model')}  dim {meta_get(db,'dim')}")
    print(f"pages:       {meta_get(db,'page_count')}")
    (blocks,) = db.execute("SELECT count(*) FROM docs").fetchone()
    print(f"blocks:      {blocks}")
    ts = meta_get(db, "indexed_at")
    if ts:
        print(f"indexed at:  {time.strftime('%Y-%m-%d %H:%M', time.localtime(int(ts)))}")

def main():
    ap = argparse.ArgumentParser(prog="lqmd", description="qmd-style search over a Logseq DB graph")
    ap.add_argument("--graph", default=GRAPH)
    ap.add_argument("--db", default=DB_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe").set_defaults(func=cmd_probe)

    pi = sub.add_parser("index"); pi.add_argument("--full", action="store_true")
    pi.set_defaults(func=cmd_index)

    for name, fn in (("search", cmd_search), ("vsearch", cmd_vsearch), ("query", cmd_query)):
        p = sub.add_parser(name)
        p.add_argument("query")
        p.add_argument("-n", type=int, default=5)
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=fn)

    pg = sub.add_parser("get"); pg.add_argument("page"); pg.set_defaults(func=cmd_get)
    sub.add_parser("status").set_defaults(func=cmd_status)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
