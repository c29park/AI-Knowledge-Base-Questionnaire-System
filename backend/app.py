from __future__ import annotations
from fastapi import FastAPI, Request, HTTPException, Query
import sys
import atexit, logging
import subprocess
import tempfile, uuid, threading, re
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path 
from typing import Any, List, Dict, Optional, Set
import json
import os
from datetime import datetime, timezone
app = FastAPI(title="KB + Questions API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], #Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = (Path(__file__).parent / "data").resolve()
DATA_DIR.mkdir(exist_ok=True)
KB_PATH = DATA_DIR / "kb.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"
ANSWERS_PATH = DATA_DIR / "answers.json"
SCRIPT_PATH = Path(__file__).parent / "basic_rag_evaluation_langgraph.py"

KB_KEYS = {"id", "category", "question", "answer"}
Q_KEYS = {"id", "text"}

# ---------- File helpers (atomic, threadsafe) ----------
_file_lock = threading.Lock()


def _unlink_quiet(p: Path) -> bool:
    try:
        p.unlink(missing_ok=True)   # py3.8+
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        logging.warning("Failed to delete %s: %s", p, e)
        return False


# also register in case the process exits without a clean shutdown
atexit.register(lambda: _unlink_quiet(ANSWERS_PATH))

def _atomic_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _read_list(path: Path) -> list:
    if not path.exists():
        _atomic_write(path, [])
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _write_list(path: Path, rows: list) -> None:
    with _file_lock:
        _atomic_write(path, rows)

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# ---------- ID + canonicalization helpers ----------
KB_ID_RE = re.compile(r"^kb[_-](\d+)$", re.I)  # accept old kb-<n>, normalize to kb_<n>

def _max_n_from_ids(ids: Set[str]) -> int:
    mx = 0
    for sid in ids:
        m = KB_ID_RE.match(str(sid))
        if m:
            try:
                n = int(m.group(1))
                if n > mx: mx = n
            except ValueError:
                pass
    return mx

def _next_kb_id(existing_ids: Set[str]) -> str:
    n = _max_n_from_ids(existing_ids) + 1
    nid = f"kb_{n}"
    existing_ids.add(nid)
    return nid

def _canon_question(q: str) -> str:
    """Lowercase + collapse whitespace; used for de-duplication by question text."""
    return re.sub(r"\s+", " ", (q or "").strip().lower())

def _ensure_array(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise HTTPException(400, "Expected a non-empty JSON array at the top level")
    return payload

def _validate_kb(arr: List[Dict[str, Any]]) -> None:
    for i, item in enumerate(arr):
        if not isinstance(item, dict):
            raise HTTPException(400, f"Item {i} is not an object")
        # id is optional for KB (server can assign)
        required = {"category", "question", "answer"}
        missing = [k for k in required if k not in item]
        if missing:
            raise HTTPException(400, f"KB item {i} missing keys: {missing}")
        wrong_types = [k for k in (set(item.keys()) & (KB_KEYS)) if not isinstance(item[k], str)]
        if wrong_types:
            raise HTTPException(400, f"KB item {i} keys not strings: {wrong_types}")

def _validate_questions(arr: List[Dict[str, Any]]) -> None:
    for i, item in enumerate(arr):
        if not isinstance(item, dict):
            raise HTTPException(400, f"Item {i} is not an object")
        missing = [k for k in Q_KEYS if k not in item]
        if missing:
            raise HTTPException(400, f"Question item {i} missing keys: {missing}")
        wrong_types = [k for k in Q_KEYS if not isinstance(item[k], str)]
        if wrong_types:
            raise HTTPException(400, f"Question item {i} keys not strings: {wrong_types}")
        if "answer" in item:
            raise HTTPException(400, f"Question item {i} must not include 'answer'")

# ---------- Answer generation jobs ----------
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()

def _start_job_for_script(script_path: Path, cwd: Path, extra_env: dict | None = None) -> str:
    job_id = uuid.uuid4().hex
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    log_path = DATA_DIR / f"answers_{job_id}.log"
    log_f = open(log_path, "w", encoding="utf-8")

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(cwd),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
    )

    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "running",
            "pid": proc.pid,
            "log": str(log_path),
            "started_at": _now_iso(),
            "ended_at": None,
            "returncode": None,
        }

    def _waiter():
        rc = proc.wait()
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "done" if rc == 0 else "error"
            _JOBS[job_id]["returncode"] = rc
            _JOBS[job_id]["ended_at"] = _now_iso()
        try:
            log_f.close()
        except Exception:
            pass

    threading.Thread(target=_waiter, daemon=True).start()
    return job_id



# ---------- Upload (replace/append) ----------
@app.post("/api/upload/kb")
async def upload_kb(req: Request, append: int = Query(0, description="append=1 to append instead of replace")):
    payload = await req.json()
    arr = _ensure_array(payload)
    _validate_kb(arr)

    current = _read_list(KB_PATH)
    existing_ids = {str(r.get("id", "")) for r in current}
    # Build a canonical question set from the current file to dedupe by question text
    qset = {_canon_question(r.get("question", "")) for r in current}

    inserted = 0
    skipped = 0
    now = _now_iso()

    if append == 1:
        # Append: assign next ids continuing from max; skip duplicates by question text
        new_ros: list[dict] = []
        for it in arr:
            cq = _canon_question(it["question"])
            if cq in qset:
                skipped += 1
                continue
            nid = _next_kb_id(existing_ids)
            new_rows.append({
                "id": nid,
                "category": str(it["category"]),
                "question": str(it["question"]),
                "answer": str(it["answer"]),
                "created_at": now,
                "updated_at": now,
            })
            qset.add(cq)
            inserted += 1

        out = current + new_rows
        _write_list(KB_PATH, out)
        return {"type": "kb", "mode": "append", "inserted": inserted, "skipped_duplicates": skipped}
    else:
        # Replace: dedupe within the provided array, then renumber from 1
        seen = set()
        uniques = []
        for it in arr:
            cq = _canon_question(it["question"])
            if cq in seen:
                skipped += 1
                continue
            seen.add(cq)
            uniques.append({
                "category": str(it["category"]),
                "question": str(it["question"]),
                "answer": str(it["answer"]),
            })

        # Assign kb_1..kb_n
        out = []
        now = _now_iso()
        for i, it in enumerate(uniques, start=1):
            out.append({
                "id": f"kb_{i}",
                "category": it["category"],
                "question": it["question"],
                "answer": it["answer"],
                "created_at": now,
                "updated_at": now,
            })
        _write_list(KB_PATH, out)
        return {"type": "kb", "mode": "replace", "replaced": len(out), "skipped_duplicates": skipped}

@app.post("/api/upload/questions")
async def upload_questions(req: Request):
    payload = await req.json()
    arr = _ensure_array(payload)
    _validate_questions(arr)
    _write_list(QUESTIONS_PATH, arr)
    return {"type": "questions", "inserted": len(arr)}

# ---------- KB CRUD used by the UI ----------
@app.post("/api/kb")
async def kb_create(req: Request):
    """
    Create one OR many KB items (append). Server assigns sequential kb_<n> ids.
    Skips duplicates by canonicalized question text.
    Body can be an object or an array of objects.
    """
    body = await req.json()
    rows = _read_list(KB_PATH)
    existing_ids = {str(r.get("id", "")) for r in rows}
    qset = {_canon_question(r.get("question", "")) for r in rows}

    def norm_one(it: Dict[str, Any]) -> Optional[Dict[str, str]]:
        for k in ("category", "question", "answer"):
            if k not in it:
                raise HTTPException(400, f"Missing '{k}'")
        cq = _canon_question(it["question"])
        if cq in qset:
            return None  # skip duplicate question
        nid = _next_kb_id(existing_ids)
        now = _now_iso()
        qset.add(cq)
        return {
            "id": nid,
            "category": str(it["category"]),
            "question": str(it["question"]),
            "answer": str(it["answer"]),
            "created_at": now,
            "updated_at": now,
        }

    if isinstance(body, list):
        items = [x for x in (norm_one(it) for it in body) if x is not None]
        if not items:
            return {"inserted": 0, "skipped_duplicates": True}
        rows.extend(items)
        _write_list(KB_PATH, rows)
        return {"inserted": len(items), "items": items}
    elif isinstance(body, dict):
        item = norm_one(body)
        if item is None:
            return {"inserted": 0, "skipped_duplicates": True}
        rows.append(item)
        _write_list(KB_PATH, rows)
        return item
    else:
        raise HTTPException(400, "Body must be object or array")

@app.put("/api/kb/{item_id}")
async def kb_update(item_id: str, req: Request):
    """
    Update fields of one KB item by id. 'id' is immutable.
    Preserves created_at, refreshes updated_at.
    """
    body = await req.json()
    rows = _read_list(KB_PATH)

    for i, r in enumerate(rows):
        if r.get("id") == item_id:
            rows[i] = {
                "id": item_id,
                "category": str(body.get("category", r.get("category", ""))),
                "question": str(body.get("question", r.get("question", ""))),
                "answer": str(body.get("answer", r.get("answer", ""))),
                "created_at": r.get("created_at") or _now_iso(),
                "updated_at": _now_iso(),
            }
            _write_list(KB_PATH, rows)
            return rows[i]
    raise HTTPException(404, f"KB item '{item_id}' not found")

@app.delete("/api/kb")
async def kb_delete(req: Request):
    """
    Delete many by ids. Body: { "ids": ["kb_1", "kb_2", ...] }
    """
    body = await req.json()
    ids_list = body.get("ids") or []
    if not isinstance(ids_list, list) or not ids_list:
        raise HTTPException(400, "Provide non-empty 'ids' array")

    ids: Set[str] = set(map(str, ids_list))
    rows = _read_list(KB_PATH)
    before = len(rows)
    rows = [r for r in rows if str(r.get("id")) not in ids]
    _write_list(KB_PATH, rows)
    return {"deleted": before - len(rows)}

@app.post("/api/kb/delete")
async def kb_delete_post(req: Request):
    body = await req.json()
    ids_list = body.get("ids") or []
    if not isinstance(ids_list, list) or not ids_list:
        raise HTTPException(400, "Provide non-empty 'ids' array")
    ids: Set[str] = set(map(str, ids_list))
    rows = _read_list(KB_PATH)
    before = len(rows)
    rows = [r for r in rows if str(r.get("id")) not in ids]
    _write_list(KB_PATH, rows)
    return {"deleted": before - len(rows)}

@app.post("/api/answers/generate")
def start_generate_answers():
    if not SCRIPT_PATH.exists():
        raise HTTPException(500, f"Script not found: {SCRIPT_PATH.name}")
    
    job_id = _start_job_for_script(
        SCRIPT_PATH,
        cwd=DATA_DIR,
        extra_env={"DUMP_ANSWERS": "1", "PYTHONUNBUFFERED": "1"},
    )

    return {"job_id": job_id, "status": "started"}

@app.post("/api/answers/clear")
def clear_answers():
    """Remove answers.json if present."""
    ok = _unlink_quiet(ANSWERS_PATH)
    return {"ok": True, "deleted": ok}

@app.on_event("shutdown")
def _cleanup_answers_on_shutdown():
    _unlink_quiet(ANSWERS_PATH)

# ---------- Lists ----------
@app.get("/api/kb")
def list_kb():
    return _read_list(KB_PATH)

@app.get("/api/questions")
def list_questions():
    return _read_list(QUESTIONS_PATH)

@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with _JOBS_LOCK:
        info = _JOBS.get(job_id)
    if not info:
        raise HTTPException(404, "job not found")
    return info

@app.get("/api/answers")
def list_answers():
    if not ANSWERS_PATH.exists():
        return []
    try:
        data = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception as e:
        raise HTTPException(500, f"Failed to read answers.json: {e}")

@app.get("/api/health")
def health():
    return {"ok": True}
