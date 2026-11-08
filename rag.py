"""Local RAG: split files into chunks, embed them with Ollama, search with NumPy.
Brute-force cosine search is fast enough for 100,000+ chunks (about 300 MB of vectors)."""
import hashlib
import json
import threading

import httpx
import numpy as np

import config


def _prefix(kind: str) -> str:
    if "nomic" in config.EMBED_MODEL:
        return "search_query: " if kind == "query" else "search_document: "
    return ""


def embed(texts: list[str], kind: str = "doc") -> np.ndarray:
    pre = _prefix("query" if kind == "query" else "doc")
    out: list[list[float]] = []
    with httpx.Client(timeout=300) as c:
        for i in range(0, len(texts), config.EMBED_BATCH):
            batch = [pre + t for t in texts[i:i + config.EMBED_BATCH]]
            r = c.post(f"{config.OLLAMA_URL}/api/embed",
                       json={"model": config.EMBED_MODEL, "input": batch, "keep_alive": config.KEEP_ALIVE})
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
    a = np.asarray(out, dtype=np.float32)
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-9
    return a


def chunk_text(text: str) -> list[str]:
    size, overlap = config.CHUNK_CHARS, config.CHUNK_OVERLAP
    text = text.replace("\r\n", "\n")
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:  # prefer to cut at a blank line, then at a line break
            cut = text.rfind("\n\n", start + size // 2, end)
            if cut == -1:
                cut = text.rfind("\n", start + size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


class Store:
    def __init__(self):
        config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        config.KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        self.vec_path = config.INDEX_DIR / "vectors.npy"
        self.meta_path = config.INDEX_DIR / "meta.json"
        self.lock = threading.Lock()
        self.vectors = np.zeros((0, 0), np.float32)
        self.chunks: list[dict] = []      # {"source": str, "text": str}
        self.files: dict[str, str] = {}   # source -> sha1
        self._load()

    def _load(self):
        if self.vec_path.exists() and self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if meta.get("embed_model") == config.EMBED_MODEL:   # different model = incompatible vectors
                self.vectors = np.load(self.vec_path)
                self.chunks, self.files = meta["chunks"], meta["files"]

    def _save(self):
        np.save(self.vec_path, self.vectors)
        self.meta_path.write_text(json.dumps(
            {"embed_model": config.EMBED_MODEL, "chunks": self.chunks, "files": self.files},
            ensure_ascii=False), encoding="utf-8")

    def stats(self) -> dict:
        return {"chunks": len(self.chunks), "files": len(self.files)}

    @staticmethod
    def _read(path) -> str:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() == ".ipynb":  # keep only the notebook's code and text cells
            try:
                nb = json.loads(raw)
                return "\n\n".join("".join(c.get("source", [])) for c in nb.get("cells", []))
            except Exception:
                pass
        return raw

    def ingest(self) -> dict:
        """Index new/changed files and drop deleted ones. Safe to run repeatedly."""
        with self.lock:
            found: dict[str, tuple] = {}
            for p in config.KNOWLEDGE_DIR.rglob("*"):
                if (p.is_file() and p.suffix.lower() in config.EXTENSIONS
                        and p.stat().st_size <= config.MAX_FILE_MB * 1024 * 1024):
                    rel = p.relative_to(config.KNOWLEDGE_DIR).as_posix()
                    found[rel] = (p, hashlib.sha1(p.read_bytes()).hexdigest())

            stale = {s for s in self.files if s not in found or self.files[s] != found[s][1]}
            keep = [i for i, c in enumerate(self.chunks) if c["source"] not in stale]
            chunks = [self.chunks[i] for i in keep]
            vectors = self.vectors[keep] if len(keep) else np.zeros((0, 0), np.float32)
            files = {s: h for s, h in self.files.items() if s not in stale}

            new_chunks = []
            for rel, (p, sha) in found.items():
                if rel in files:
                    continue
                for piece in chunk_text(self._read(p)):
                    new_chunks.append({"source": rel, "text": piece})
                files[rel] = sha

            if new_chunks:
                new_vecs = embed([f"{c['source']}\n{c['text']}" for c in new_chunks])
                vectors = new_vecs if vectors.size == 0 else np.vstack([vectors, new_vecs])
                chunks += new_chunks

            self.vectors, self.chunks, self.files = vectors, chunks, files
            self._save()
            return {"added_chunks": len(new_chunks), "removed_files": len(stale - set(found)), **self.stats()}

    def search(self, query: str, k: int | None = None) -> list[dict]:
        k = k or config.TOP_K
        if not self.chunks:
            return []
        q = embed([query], "query")[0]
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top if scores[i] >= config.MIN_SCORE]
