"""Tiny TigerGraph Savanna REST client (global-secret auth).

Kept dependency-free (requests only) and reproducible: every call goes through
here so host/secret live in exactly one place (.env). No pyTigerGraph needed.
"""
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from dotenv import dotenv_values  # noqa: E402

_ENV = dotenv_values(Path(__file__).parent.parent / ".env")
HOST = _ENV["TG_HOST"].rstrip("/")
SECRET = _ENV["TG_SECRET"]
GRAPH = _ENV.get("TG_GRAPH", "sp100")

_session = requests.Session()
_token_cache: dict = {}


def _is_suspended(text: str) -> bool:
    t = (text or "").lower()
    return "auto start is not enabled" in t or "failed to start workspace" in t


def token(graph: str | None = None, _retries: int = 6) -> str:
    """Fetch (and cache) an auth token. A Savanna workspace that auto-suspended and is
    resuming returns 5xx for ~30-60s; retry with backoff instead of failing the run."""
    key = graph or "__global__"
    if key in _token_cache:
        return _token_cache[key]
    body = {"secret": SECRET}
    if graph:
        body["graph"] = graph
    last = ""
    for attempt in range(_retries):
        try:
            r = _session.post(f"{HOST}/gsql/v1/tokens", json=body, timeout=25)
            if r.status_code == 200:
                tok = r.json()["token"]
                _token_cache[key] = tok
                return tok
            last = r.text
            if _is_suspended(r.text):
                raise RuntimeError(
                    "TigerGraph workspace is SUSPENDED — Resume MyWorkspace in the Savanna UI "
                    "(and enable Auto Start to avoid mid-run suspends).")
        except RuntimeError:
            raise
        except Exception as e:
            last = str(e)
        wait = min(30, 5 * (attempt + 1))
        print(f"  [tg] token attempt {attempt+1}/{_retries} failed; retrying in {wait}s…", flush=True)
        time.sleep(wait)
    raise requests.HTTPError(f"TigerGraph token failed after {_retries} tries: {last[:200]}")


def invalidate_token(graph: str | None = None) -> None:
    _token_cache.pop(graph or "__global__", None)


def gsql(statement: str, graph: str | None = None, timeout: int = 120) -> str:
    """Run a GSQL statement (or batch) via /gsql/v1/statements. graph=None -> global.
    Re-auths once on a 401/403 (token invalidated by a suspend/resume cycle)."""
    for attempt in range(2):
        H = {"Authorization": f"Bearer {token(graph)}", "Content-Type": "text/plain"}
        r = _session.post(f"{HOST}/gsql/v1/statements", headers=H,
                          data=statement.encode("utf-8"), timeout=timeout)
        if r.status_code in (401, 403) and attempt == 0:
            invalidate_token(graph)
            continue
        r.raise_for_status()
        return r.text
    r.raise_for_status()
    return r.text


def restpp(path: str, method: str = "GET", graph: str | None = None, **kw) -> requests.Response:
    """Call a REST++ data endpoint (e.g. /restpp/graph, /restpp/query)."""
    g = graph or GRAPH
    H = {"Authorization": f"Bearer {token(g)}"}
    H.update(kw.pop("headers", {}) or {})
    return _session.request(method, f"{HOST}{path}", headers=H, timeout=kw.pop("timeout", 120), **kw)


def graphs() -> list[str]:
    import re
    out = gsql("SHOW GRAPH *")
    return re.findall(r"Graph (\w+)\(", out)
