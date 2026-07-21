# lqmd

Local hybrid search over a Logseq DB-version graph, with a small read-only
HTTP server and PWA on top.

The graph itself is the source of truth: content is pulled through the
`logseq` CLI as Datalog queries — no markdown files are read or written. The
index lives in a single SQLite file combining an FTS5 keyword index and a
sqlite-vec vector index. Embeddings are computed in-process with
[fastembed](https://github.com/qdrant/fastembed) (ONNX Runtime, CPU); no
embedding daemon is required.

## Components

| Path | Purpose |
|---|---|
| `lqmd.py` | CLI: indexes the graph and searches it (keyword, vector, hybrid) |
| `server.py` | FastAPI wrapper exposing the same search read-only over HTTP |
| `static/` | Installable PWA search UI served by the server |
| `lqmd-server.service` | systemd user unit for running the server |

## How search works

- **One record per block.** Each block is indexed with its page title and
  parent-block breadcrumb prepended, so short blocks carry enough context to
  be found. `[[uuid]]` and `((uuid))` references are resolved to titles.
- **Keyword** (`search`): SQLite FTS5 with the `porter unicode61` tokenizer
  (so plurals and word forms match), ranked by BM25. The all-terms (AND)
  match runs first; any-term (OR) matches only fill remaining candidate room.
- **Vector** (`vsearch`): cosine similarity over `BAAI/bge-base-en-v1.5`
  embeddings (768-dim) stored in sqlite-vec.
- **Hybrid** (`query`): both lists fused with Reciprocal Rank Fusion — the
  best default.
- **Constraints:** queries may include `#tag` and `[[page]]` filters. These
  are resolved against the *live* graph via Datalog at query time and
  intersected (AND) with the free-text results. A constraint-only query
  returns all matching blocks.
- **Incremental indexing:** every run re-pulls and re-chunks the graph but
  only embeds blocks whose content hash changed. `--full` forces a clean
  rebuild, and is required whenever the embedding model (dimension) or FTS
  tokenizer changes.

## CLI

```
pip install sqlite-vec fastembed

lqmd probe                 inspect what the graph extraction returns (run this first)
lqmd index [--full]        pull the graph, chunk, embed changed blocks, build the index
lqmd search  "<query>"     keyword (BM25) only — fast, no embedding model
lqmd vsearch "<query>"     vector semantic search only
lqmd query   "<query>"     hybrid: BM25 + vector fused by RRF
lqmd get <page|#docid>     print an indexed page as an outline
lqmd status                index health
```

Common flags: `-n <num>`, `--json`, `--graph <name>`, `--db <path>`.

## Server

```
pip install fastapi uvicorn      # in addition to the CLI deps
python3 server.py                # binds 0.0.0.0:8765 by default
```

The server imports `lqmd.py` directly so query vectors and stored document
vectors come from one embedding path. It never writes to the graph or the
index; build the index with `lqmd index` before starting it.

| Endpoint | Description |
|---|---|
| `GET /search?q=...&n=5&mode=hybrid\|keyword\|vector` | ranked results (JSON) |
| `GET /get?page=NAME` / `?docid=XXXXXX` / `?page_uuid=UUID` | full indexed page |
| `GET /status` | index metadata and health |
| `GET /health` | liveness (200 once warm) |
| `GET /` | the PWA |

## Configuration

All configuration is via environment variables (the systemd unit is the
canonical place to set them):

| Variable | Default | Purpose |
|---|---|---|
| `LQMD_GRAPH` | `vault` | Logseq graph name |
| `LQMD_ROOT_DIR` | *(logseq default)* | Logseq root directory |
| `LQMD_DB` | `~/.cache/lqmd/index.sqlite` | index location |
| `LQMD_LOGSEQ` | `logseq` | path to the logseq CLI binary |
| `LQMD_QUERY_TIMEOUT_MS` | `120000` | Datalog query timeout |
| `LQMD_EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | fastembed model id |
| `LQMD_MODEL_CACHE` | `~/.local/share/lqmd/models` | model download cache |
| `LQMD_QUERY_PREFIX` | bge query prefix | prepended to query embeddings |
| `LQMD_DOC_PREFIX` | *(empty)* | prepended to document embeddings |
| `LQMD_SERVER_HOST` | `0.0.0.0` | server bind address |
| `LQMD_SERVER_PORT` | `8765` | server port |

Changing `LQMD_EMBED_MODEL` to a model with a different dimension requires
`lqmd index --full`; the indexer refuses mismatched dimensions otherwise.

## Deployment

`lqmd-server.service` is a systemd user unit. Install and run:

```
cp lqmd-server.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now lqmd-server
```

After deploying code changes, `systemctl --user restart lqmd-server`
(the unit has no `ExecReload`).

## Security

The server has **no authentication** by design. It is meant for a trusted
home LAN, single user. For access from outside, put an authenticating layer
in front of it — this deployment uses Cloudflare Zero Trust; an
authenticating reverse proxy works equally well. Do not expose the port
directly to the internet.
