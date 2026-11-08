"""Tiny RAG: markdown KB embedded with nomic-embed-text (Ollama), cosine search in NumPy."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .config import EMBED_MODEL, KB_DIR, KB_INDEX, LLM_BASE_URL

Embedder = Callable[[Sequence[str]], np.ndarray]


def ollama_embedder(base_url: str = LLM_BASE_URL, model: str = EMBED_MODEL) -> Embedder:
    import httpx

    def embed(texts: Sequence[str]) -> np.ndarray:
        r = httpx.post(f"{base_url.rstrip('/')}/embeddings",
                       json={"model": model, "input": list(texts)}, timeout=120)
        r.raise_for_status()
        return np.array([d["embedding"] for d in r.json()["data"]], dtype=np.float32)

    return embed


def hashing_embedder(dim: int = 256) -> Embedder:
    """Deterministic bag-of-words embedder for tests (no network)."""
    def embed(texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
        return out
    return embed


def load_articles(kb_dir: Path = KB_DIR) -> list[dict]:
    arts = []
    for p in sorted(Path(kb_dir).glob("*.md")):
        text = p.read_text(encoding="utf-8")
        title = text.splitlines()[0].lstrip("# ").strip() if text else p.stem
        arts.append({"id": p.stem, "title": title, "text": text})
    return arts


def _norm(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)


class KnowledgeBase:
    def __init__(self, articles: list[dict], embedder: Embedder, matrix: np.ndarray | None = None):
        self.articles = articles
        self.embedder = embedder
        self.matrix = _norm(matrix if matrix is not None
                            else embedder([a["text"] for a in articles]))

    @classmethod
    def build(cls, kb_dir: Path = KB_DIR, embedder: Embedder | None = None,
              cache: Path | None = KB_INDEX) -> "KnowledgeBase":
        embedder = embedder or ollama_embedder()
        arts = load_articles(kb_dir)
        sig = hashlib.md5("".join(a["text"] for a in arts).encode()).hexdigest()
        if cache and Path(cache).exists():
            z = np.load(cache, allow_pickle=False)
            if str(z["sig"]) == sig:
                return cls(arts, embedder, z["matrix"])
        kb = cls(arts, embedder)
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, matrix=kb.matrix, sig=sig)
        return kb

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = _norm(self.embedder([query]))[0]
        scores = self.matrix @ q
        idx = np.argsort(-scores)[:k]
        return [{"id": self.articles[i]["id"], "title": self.articles[i]["title"],
                 "score": round(float(scores[i]), 4), "text": self.articles[i]["text"]}
                for i in idx]
