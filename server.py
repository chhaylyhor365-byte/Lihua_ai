"""Lihua AI backend: serves the chat UI and streams answers from a local Qwen3 model via Ollama."""
import json
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from rag import Store

app = FastAPI(title="Lihua AI")
store = Store()
STATIC = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    messages: list[dict]      # [{"role": "user"|"assistant", "content": "..."}]
    use_rag: bool = True
    think: bool = False


def line(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


def trim(messages: list[dict], budget_chars: int) -> list[dict]:
    """Keep the newest messages that fit in the context window (about 3 chars per token)."""
    kept, used = [], 0
    for m in reversed(messages):
        used += len(m.get("content", ""))
        if used > budget_chars and kept:
            break
        kept.append({"role": m["role"], "content": m.get("content", "")})
    return list(reversed(kept))


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/status")
async def status():
    info = {"ollama": False, "model": config.CHAT_MODEL, "model_ready": False,
            "embed_model": config.EMBED_MODEL, "embed_ready": False, **store.stats()}
    try:
        async with httpx.AsyncClient(timeout=3) as c:
            r = await c.get(f"{config.OLLAMA_URL}/api/tags")
        names = [m["name"] for m in r.json().get("models", [])]
        has = lambda want: any(n == want or n.split(":")[0] == want or n == want + ":latest" for n in names)
        info.update(ollama=True, model_ready=has(config.CHAT_MODEL), embed_ready=has(config.EMBED_MODEL))
    except Exception:
        pass
    return info


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.messages or req.messages[-1].get("role") != "user":
        raise HTTPException(400, "Last message must come from the user.")
    question = req.messages[-1]["content"]

    # NEW: long answer only when the user asks to explain
    explain = any(w in question.lower() for w in
                  ["explain", "why", "in detail", "how does", "how do"])
    if req.think:
        max_tokens = 2000          # thinking uses tokens too, so allow more
    elif explain:
        max_tokens = config.NUM_PREDICT_EXPLAIN
    else:
        max_tokens = config.NUM_PREDICT_SHORT

    async def stream():
        hits = []
        if req.use_rag and store.chunks:
            try:
                hits = await run_in_threadpool(store.search, question)
            except Exception as e:
                yield line({"type": "notice", "text": f"Knowledge search skipped: {e}"})
        system = config.SYSTEM_PROMPT
        if hits:
            refs = "\n\n".join(f"[{i + 1}] {h['source']}\n{h['text']}" for i, h in enumerate(hits))
            system += f"\n\nReference material from the user's knowledge base:\n\n{refs}"
            yield line({"type": "sources", "items": [
                {"source": h["source"], "score": round(h["score"], 2)} for h in hits]})

        budget = int(config.NUM_CTX * 3) - len(system)
        payload = {
            "model": config.CHAT_MODEL,
            "messages": [{"role": "system", "content": system}] + trim(req.messages, max(budget, 2000)),
            "stream": True,
            "think": req.think,
            "keep_alive": config.KEEP_ALIVE,
            "options": {
                "num_ctx": config.NUM_CTX,
                "temperature": config.TEMPERATURE,
                "num_predict": max_tokens,      # NEW
            },
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(None, connect=5)) as c:
                async with c.stream("POST", f"{config.OLLAMA_URL}/api/chat", json=payload) as r:
                    if r.status_code != 200:
                        body = (await r.aread()).decode("utf-8", "ignore")
                        yield line({"type": "error", "text": f"Ollama error {r.status_code}: {body[:300]}"})
                        return
                    async for raw in r.aiter_lines():
                        if not raw:
                            continue
                        d = json.loads(raw)
                        msg = d.get("message", {})
                        if msg.get("thinking"):
                            yield line({"type": "thinking", "text": msg["thinking"]})
                        if msg.get("content"):
                            yield line({"type": "token", "text": msg["content"]})
                        if d.get("done"):
                            yield line({"type": "done"})
        except httpx.ConnectError:
            yield line({"type": "error", "text": "Can't reach Ollama. Start it, then try again."})
        except Exception as e:
            yield line({"type": "error", "text": str(e)})

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/ingest")
def ingest():
    try:
        return store.ingest()
    except httpx.HTTPError as e:
        raise HTTPException(502, f"Embedding failed. Is Ollama running with '{config.EMBED_MODEL}' installed? ({e})")


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    saved = []
    for f in files:
        name = Path(f.filename or "").name
        if Path(name).suffix.lower() not in config.EXTENSIONS:
            continue
        (config.KNOWLEDGE_DIR / name).write_bytes(await f.read())
        saved.append(name)
    if not saved:
        raise HTTPException(400, "No supported files. Use text or code files such as .md, .txt, .py, .js.")
    result = await run_in_threadpool(ingest)
    return {"saved": saved, **result}


app.mount("/static", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    print(f"\n  Lihua AI  →  http://{config.HOST}:{config.PORT}\n  Model: {config.CHAT_MODEL}\n")
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")
