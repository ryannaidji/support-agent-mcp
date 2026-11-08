import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
DB_PATH = Path(os.getenv("SUPPORT_DB", ROOT / "data" / "support.db"))
KB_DIR = Path(os.getenv("KB_DIR", ROOT / "kb"))
KB_INDEX = Path(os.getenv("KB_INDEX", ROOT / "data" / "kb_index.npz"))
