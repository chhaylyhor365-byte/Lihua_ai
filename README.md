# Lihua AI

A coding assistant that runs 100% on your laptop. ChatGPT-style chat UI, **Qwen3** as the model, **local RAG** over your own files.

```
Browser UI  ->  server.py (FastAPI)  ->  Ollama (Qwen3 on your GPU)
                      |
                      +-> rag.py: your files in ./knowledge -> embeddings -> search
```

## 1. Install (once)

1. Install **Ollama**: https://ollama.com/download
2. Pull the models (open a terminal):
   ```
   ollama pull qwen3:8b
   ollama pull nomic-embed-text
   ```
3. Install **Python 3.10+** and **VS Code** (with the Python extension).

## 2. Run in VS Code

1. `File > Open Folder...` and choose the `lihua-ai` folder.
2. Open a terminal in VS Code (`Ctrl+``) and run:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python server.py
   ```
   (Or press **F5** and choose "Lihua AI" after selecting the `.venv` interpreter. On Windows you can also double-click `start.bat`.)
3. Open http://127.0.0.1:8000

## 3. Use your own knowledge (RAG)

- Click **Add files** in the sidebar, or copy files into the `knowledge/` folder and click **Re-index**.
- Supported: .md .txt .py .js .ts .java .c .cpp .cs .go .rs .php .sql .html .css .json .yaml .ipynb and more.
- Keep the **Knowledge** toggle on. Answers show which files were used.
- Only new or changed files are re-embedded. The index lives in `index/`.

The store is built for 100,000+ chunks: about 300 MB of vectors in RAM, and a search takes milliseconds.

## 4. Settings (`config.py`)

| Setting | What it does |
|---|---|
| `CHAT_MODEL` | `qwen3:8b` by default. Use `qwen3:4b` if it feels slow, `qwen3:14b` if your GPU has 12 GB. Newer Qwen3.x tags work too: check https://ollama.com/library |
| `NUM_CTX` | Context window in tokens (default 8192). More context uses more VRAM. |
| `TOP_K`, `MIN_SCORE` | How many chunks the model sees, and how strict matching is |

You can also set them with environment variables: `LIHUA_MODEL`, `LIHUA_CTX`, `LIHUA_PORT`.

## Your hardware (16 GB RAM, RTX 3060)

Run `nvidia-smi` and check the VRAM. A **laptop** RTX 3060 has **6 GB**, a desktop one has 12 GB.

| VRAM | Recommended model | Context |
|---|---|---|
| 6 GB | `qwen3:4b` (fast), or `qwen3:8b` (smarter, partly on CPU) | 8k |
| 12 GB | `qwen3:8b` or `qwen3:14b` | 8k to 16k |

To save VRAM, set these Windows environment variables, then restart Ollama:
`OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0`.

**About the 100,000:** a true 100k-token context window needs far more VRAM than a 3060 has (the KV cache alone is ~15 GB for Qwen3-8B at fp16). RAG solves this: Lihua searches your 100,000+ chunks and sends only the best few to the model.

## Files

- `server.py` web server and chat streaming
- `rag.py` chunking, embeddings, search
- `config.py` all settings
- `static/` the chat UI (`index.html`, `style.css`, `app.js`, offline libraries in `vendor/`)
- `knowledge/` your documents
