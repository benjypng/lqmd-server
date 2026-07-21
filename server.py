#!/usr/bin/env python3
"""
lqmd server - read-only HTTP access to lqmd's hybrid search over a Logseq graph.

A thin wrapper over lqmd.py. It imports lqmd and calls its retrieval and
embedding functions directly, so query vectors and the stored document vectors
come from one embedding path. The server never writes to the graph or the index;
it only reads the prebuilt SQLite index that `lqmd index` produces.

Security: there is NO authentication here. The server is meant to run on a home
LAN for a single user. Exposing it beyond the LAN would require adding a token
(or putting an authenticating reverse proxy in front of it); that is deliberately
not built. Bind it to a trusted interface only.

Run (on the host where lqmd's index, model cache, and deps live):

    pip install fastapi uvicorn          # in addition to lqmd's own deps
    python3 server.py                    # binds 0.0.0.0:8765 by default

Override the bind with LQMD_SERVER_HOST / LQMD_SERVER_PORT. The index path,
graph, embed model, and prefixes are whatever lqmd resolves from its own
environment (LQMD_DB, LQMD_GRAPH, ...).

Endpoints:
    GET /search?q=...&n=5&mode=hybrid|keyword|vector   ranked results (JSON)
    GET /get?page=NAME   |  /get?docid=XXXXXX           full indexed page text
    GET /status                                         index health
    GET /health                                         liveness (ok once warm)
    /                                                   static PWA (static/)
"""

import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import lqmd

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
HOST = os.environ.get("LQMD_SERVER_HOST", "0.0.0.0")
PORT = int(os.environ.get("LQMD_SERVER_PORT", "8765"))

state = {
    "db": None,
    "lock": threading.Lock(),
    "ready": False,
    "wal_mode": None,
}


def _ensure_wal(db_path):
    """Best-effort: open the index writable once so lqmd.connect sets WAL, then
    drop the writable handle. Lets a reindex (a separate writer) and the
    server's read-only handle coexist without blocking each other. Never fatal:
    if a reindex currently holds the file we just report the failure and serve
    read-only anyway."""
    try:
        writer = lqmd.connect(db_path)
        mode = writer.execute("PRAGMA journal_mode").fetchone()[0]
        writer.close()
        return mode
    except Exception as exc:
        return f"unknown ({exc})"


@asynccontextmanager
async def lifespan(app):
    db_path = lqmd.DB_PATH
    if not os.path.exists(db_path):
        raise RuntimeError(
            f"No index at {db_path}. Build it with `lqmd index` before starting the server.")

    state["wal_mode"] = _ensure_wal(db_path)

    lqmd.embed_texts(["warm up the embedding model"], is_query=True)

    state["db"] = lqmd.connect(db_path, read_only=True, check_same_thread=False)
    state["ready"] = True

    try:
        yield
    finally:
        db = state.get("db")
        if db is not None:
            db.close()


app = FastAPI(title="lqmd server", lifespan=lifespan)


@app.get("/health")
def health():
    if state["ready"] and state["db"] is not None:
        return {"status": "ok"}
    return JSONResponse({"status": "starting"}, status_code=503)


@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    n: int = Query(5, ge=1, le=100),
    mode: str = Query("hybrid"),
):
    mode = mode.lower()
    if mode not in ("hybrid", "keyword", "vector"):
        raise HTTPException(status_code=400, detail="mode must be hybrid, keyword, or vector")

    db = state["db"]
    with state["lock"]:
        result = lqmd.run_search(db, q, n, mode=mode)

    if result.get("error"):
        raise HTTPException(status_code=502, detail=f"constraint query failed: {result['error']}")

    rows = result["rows"]
    return {
        "query": q, "mode": mode, "n": n, "count": len(rows),
        "text": result["text"], "tags": result["tags"], "pages": result["pages"],
        "results": rows,
    }


@app.get("/get")
def get(
    page: Optional[str] = Query(None),
    docid: Optional[str] = Query(None),
    page_uuid: Optional[str] = Query(None),
):
    if not page and not docid and not page_uuid:
        raise HTTPException(status_code=400, detail="provide page, docid, or page_uuid")

    db = state["db"]
    with state["lock"]:
        if page_uuid:
            blocks = lqmd.get_page_blocks(db, page_uuid=page_uuid)
        else:
            blocks = lqmd.get_page_blocks(db, f"#{docid}" if docid else page)

    if blocks is None:
        raise HTTPException(status_code=404, detail="not found")
    text = "\n".join(("  " * b["depth"]) + "- " + b["text"] for b in blocks)
    return {"page": page, "docid": docid, "page_uuid": page_uuid,
            "blocks": blocks, "text": text}


@app.get("/status")
def status():
    db = state["db"]
    with state["lock"]:
        (chunks,) = db.execute("SELECT count(*) FROM docs").fetchone()
        meta = {k: lqmd.meta_get(db, k)
                for k in ("graph", "embed_model", "dim", "page_count", "indexed_at")}

    indexed_at = int(meta["indexed_at"]) if meta["indexed_at"] else None
    return {
        "db": lqmd.DB_PATH,
        "graph": meta["graph"],
        "embed_model": meta["embed_model"],
        "dim": int(meta["dim"]) if meta["dim"] else None,
        "pages": int(meta["page_count"]) if meta["page_count"] else None,
        "chunks": chunks,
        "indexed_at": indexed_at,
        "indexed_at_iso": (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(indexed_at))
                           if indexed_at else None),
        "wal_mode": state["wal_mode"],
        "ready": state["ready"],
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, workers=1)
