"""Lihua AI settings. Change values here, or override with environment variables."""
import os
from pathlib import Path

BASE = Path(__file__).parent

# --- Ollama (runs the models locally on your GPU) ---
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
CHAT_MODEL = os.getenv("LIHUA_MODEL", "lihua-coder")
EMBED_MODEL = os.getenv("LIHUA_EMBED", "nomic-embed-text")

# --- Generation ---
NUM_CTX = int(os.getenv("LIHUA_CTX", "4096"))
TEMPERATURE = 0.3                # CHANGED: was 0.6
KEEP_ALIVE = "30m"
NUM_PREDICT_SHORT = 900          # NEW: max tokens for normal answers
NUM_PREDICT_EXPLAIN = 800        # NEW: max tokens when user asks to explain
THINK = False                    # NEW: turn off qwen3 thinking mode

# --- RAG (your local knowledge base) ---
KNOWLEDGE_DIR = BASE / "knowledge"
INDEX_DIR = BASE / "index"
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150
TOP_K = 5
MIN_SCORE = 0.30
EMBED_BATCH = 32
MAX_FILE_MB = 5
EXTENSIONS = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".h", ".cpp",
    ".hpp", ".cs", ".go", ".rs", ".php", ".rb", ".kt", ".swift", ".sql", ".html", ".css",
    ".json", ".yaml", ".yml", ".toml", ".sh", ".bat", ".ps1", ".ipynb",
}

# --- Web server ---
HOST = "127.0.0.1"
PORT = int(os.getenv("LIHUA_PORT", "8000"))

SYSTEM_PROMPT = """You are Lihua AI, a friendly programming assistant.

ANSWER STYLE (always):
- Start with ONE short sentence that says what the code does.
- Then show the code in a fenced block with the language name, like ```cpp or ```python.
- After the code, write 2 to 4 short sentences or a short list of key points that explain how it works.
- Use simple, clear words.
- No long introduction, no repeating the question, no "Hope this helps".

EXPLAIN MODE (when the user says "explain", "why", "how does it work", or "in detail"):
- Explain step by step in simple words, with a code example.
- Keep it under 250 words.

CODE FORMAT (very important):
- ALWAYS put code inside a fenced block with the language name.
- Never write code as plain text.
- Inline names like `sum` or `main()` use single backticks.

OTHER RULES:
- Reply in the same language the user writes in.
- Never invent functions or libraries. If unsure, say so in one sentence.
- If knowledge-base material is provided, use it and name the file."""
