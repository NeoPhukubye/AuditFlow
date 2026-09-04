"""
Knowledge Base (RAG) module for AuditFlow.

Provides Retrieval-Augmented Generation capabilities:

- ``EmbeddingProvider`` abstraction with two implementations:
    * ``GeminiEmbeddingProvider``  -- dense embeddings via the official Google
      GenAI SDK (``text-embedding-004``). Used in production when an API key is
      present.
    * ``LocalEmbeddingProvider``   -- TF-IDF + hashing fallback that runs fully
      offline with scikit-learn. Lets the pipeline be imported, tested and the
      retrieval path exercised without any cloud credentials.

- ``VectorStore`` -- lightweight in-memory cosine-similarity vector store
  backed by numpy. For larger deployments this can be swapped for FAISS,
  Qdrant or a managed vector DB without changing the public API.

- ``KnowledgeBase`` -- loads articles from a directory, chunks them, embeds
  them and serves top-k similarity search. Supports ``.json`` and
  ``.md``/``.txt`` documents, with optional YAML front-matter for markdown.

Usage::

    from knowledge_base import create_knowledge_base, format_context_for_prompt

    kb = create_knowledge_base("kb")
    results = kb.search("refund policy", k=3)
    print(format_context_for_prompt(results))
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GOOGLE_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
MAX_RESULT_CHARS = 400


# --------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------- #
@dataclass
class Document:
    """A raw knowledge-base article loaded from disk."""

    doc_id: str
    title: str
    content: str
    source: str
    category: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A slice of a document after chunking."""

    chunk_id: str
    doc_id: str
    title: str
    content: str
    source: str
    category: Optional[str]
    tags: list[str]
    index: int
    embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """A retrieved chunk with its relevance score."""

    chunk: Chunk
    score: float


# --------------------------------------------------------------------- #
# Embedding providers
# --------------------------------------------------------------------- #
@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol every embedding provider must satisfy."""

    def fit(self, texts: Sequence[str]) -> None:
        """Statefully fit on a corpus. Stateless providers treat as a no-op."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an L2-normalised array of shape ``(len(texts), dim)``."""

    @property
    def dimensionality(self) -> int:
        """Output dimensionality of the embedding vectors."""


class GeminiEmbeddingProvider:
    """Dense embeddings sourced from the Gemini embedding model.

    Requires ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) to be set.
    """

    def __init__(
        self,
        client: Optional[genai.Client] = None,
        model: str = GOOGLE_EMBEDDING_MODEL,
    ) -> None:
        self._model = model
        self._client = client
        self._dim: Optional[int] = None

    def _client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client()
        return self._client

    def fit(self, texts: Sequence[str]) -> None:  # stateless -> no-op
        return None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        response = self._client().models.embed_content(
            model=self._model,
            contents=list(texts),
        )
        vectors = [np.asarray(e.values, dtype=np.float32) for e in response.embeddings]
        arr = np.vstack(vectors)
        self._dim = arr.shape[1]
        return _normalize(arr)

    @property
    def dimensionality(self) -> int:
        if self._dim is None:
            raise RuntimeError("Embedding dimensionality unknown until embed() is called.")
        return self._dim


class LocalEmbeddingProvider:
    """TF-IDF + hashing embedding provider that works fully offline.

    Quality is lower than dense model embeddings, but the retrieval path is
    deterministic and needs no network access, making it ideal for local
    development, CI and environments without a Gemini key.
    """

    def __init__(self, max_features: int = 4000) -> None:
        self._max_features = max_features
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._dim: int = 0

    def fit(self, texts: Sequence[str]) -> None:
        if self._vectorizer is not None:
            return
        self._vectorizer = TfidfVectorizer(
            max_features=self._max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )
        self._vectorizer.fit(list(texts))
        self._dim = len(self._vectorizer.vocabulary_)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._vectorizer is None:
            raise RuntimeError("LocalEmbeddingProvider must be fitted before embedding.")
        matrix = self._vectorizer.transform(list(texts))
        dense = np.asarray(matrix.toarray(), dtype=np.float32)
        return _normalize(dense)

    @property
    def dimensionality(self) -> int:
        if self._dim == 0:
            raise RuntimeError("Dimensionality unknown until fit() is called.")
        return self._dim


def _normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalise rows so dot-product == cosine similarity."""
    if arr.size == 0:
        return arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (arr / norms).astype(np.float32)


# --------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------- #
class VectorStore:
    """Minimal in-memory cosine-similarity vector store (numpy)."""

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._vectors: list[np.ndarray] = []
        self._metadatas: list[dict[str, Any]] = []

    def add(self, vector: np.ndarray, metadata: dict[str, Any]) -> None:
        vector = _normalize(vector.reshape(1, -1))[0]
        self._vectors.append(vector)
        self._metadatas.append(metadata)

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[dict[str, Any], float]]:
        if not self._vectors or k <= 0:
            return []
        q = _normalize(query.reshape(1, -1))[0]
        matrix = np.vstack(self._vectors)
        sims = cosine_similarity(q.reshape(1, -1), matrix)[0]
        top_idx = np.argsort(sims)[::-1][:k]
        return [
            (self._metadatas[i], float(sims[i]))
            for i in top_idx
            if sims[i] > 0.0
        ]


# --------------------------------------------------------------------- #
# Knowledge base
# --------------------------------------------------------------------- #
class KnowledgeBase:
    """Loads, chunks, embeds and indexes a directory of knowledge articles."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self.provider = provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: list[Chunk] = []
        self._store: Optional[VectorStore] = None

    # -- construction -------------------------------------------------- #
    @classmethod
    def from_directory(
        cls,
        path: str,
        provider: Optional[EmbeddingProvider] = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> "KnowledgeBase":
        kb = cls(provider or _select_provider(), chunk_size, chunk_overlap)
        kb.load_directory(path)
        return kb

    def load_directory(self, path: str) -> None:
        documents = self._load_documents(path)
        if not documents:
            self._chunks = []
            self._store = None
            return
        self._chunks = self._chunk_documents(documents)
        self._build_index()

    # -- document loading --------------------------------------------- #
    def _load_documents(self, path: str) -> list[Document]:
        documents: list[Document] = []
        for file_path in sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True)):
            if not os.path.isfile(file_path):
                continue
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".json":
                documents.append(self._load_json_doc(file_path))
            elif ext in (".md", ".txt"):
                documents.append(self._load_text_doc(file_path))
        return documents

    @staticmethod
    def _load_json_doc(file_path: str) -> Document:
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        stem = Path(file_path).stem
        return Document(
            doc_id=str(data.get("id", stem)),
            title=str(data.get("title", stem.replace("_", " ").title())),
            content=str(data.get("content", "")),
            source=file_path,
            category=data.get("category"),
            tags=list(data.get("tags", [])),
            metadata={
                k: v for k, v in data.items()
                if k not in ("id", "title", "content", "category", "tags")
            },
        )

    @staticmethod
    def _load_text_doc(file_path: str) -> Document:
        with open(file_path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        stem = Path(file_path).stem
        title = stem.replace("_", " ").title()
        category: Optional[str] = None
        content = raw
        frontmatter: dict[str, Any] = {}
        if raw.startswith("---"):
            match = re.split(r"^---\s*$(.*?)^---\s*$", raw, flags=re.DOTALL | re.MULTILINE)
            if match:
                header, content = match[1], match[2]
                for line in header.strip().splitlines():
                    if ":" in line:
                        key, value = line.split(":", 1)
                        frontmatter[key.strip()] = value.strip()
                title = frontmatter.get("title", title)
                category = frontmatter.get("category")
        return Document(
            doc_id=stem,
            title=title,
            content=content.strip(),
            source=file_path,
            category=category or frontmatter.get("category"),
            tags=[t.strip() for t in frontmatter.get("tags", "").split(",") if t.strip()],
            metadata=frontmatter,
        )

    # -- chunking ------------------------------------------------------ #
    def _chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            for idx, piece in enumerate(self._chunk_text(doc.content)):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}::c{idx}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        content=piece,
                        source=doc.source,
                        category=doc.category,
                        tags=doc.tags,
                        index=idx,
                    )
                )
        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        sentences = [s.strip() for s in re.split(r"(?<=[.!?;:])\s+", text) if s.strip()]
        if not sentences:
            sentences = [text]
        merged: list[str] = []
        current = ""
        for sent in sentences:
            if current and len(current) + len(sent) + 1 > self.chunk_size:
                merged.append(current)
                current = sent
            else:
                current = (current + " " + sent).strip()
        if current:
            merged.append(current)

        # add overlap between adjacent chunks
        if self.chunk_overlap > 0 and len(merged) > 1:
            overlapped: list[str] = []
            for i, chunk in enumerate(merged):
                if i == 0:
                    overlapped.append(chunk)
                else:
                    tail = merged[i - 1][-self.chunk_overlap:] if len(merged[i - 1]) > self.chunk_overlap else merged[i - 1]
                    overlapped.append((tail + " " + chunk).strip())
            return overlapped
        return merged

    # -- indexing ------------------------------------------------------ #
    def _build_index(self) -> None:
        texts = [c.content for c in self._chunks]
        self.provider.fit(texts)
        embeddings = self.provider.embed(texts)
        dim = embeddings.shape[1] if embeddings.size else 0
        self._store = VectorStore(dim) if dim else None
        if self._store is not None:
            for chunk, vec in zip(self._chunks, embeddings):
                chunk.embedding = vec
                self._store.add(vec, {"chunk_id": chunk.chunk_id})

    # -- retrieval ----------------------------------------------------- #
    def search(
        self,
        query: str,
        k: int = 5,
        category_filter: Optional[str] = None,
    ) -> list[RetrievalResult]:
        if self._store is None or not query.strip():
            return []
        query_vec = self.provider.embed([query])[0]
        matches = self._store.search(query_vec, k)
        chunk_map = {c.chunk_id: c for c in self._chunks}
        results = [
            RetrievalResult(chunk=chunk_map[meta["chunk_id"]], score=score)
            for meta, score in matches
        ]
        if category_filter:
            results = [r for r in results if (r.chunk.category or "").lower() == category_filter.lower()]
        return results

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _select_provider() -> EmbeddingProvider:
    """Pick Gemini when a key is available, otherwise fall back to local."""
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        if not os.getenv("AUDITFLOW_FORCE_LOCAL_KB"):
            return GeminiEmbeddingProvider()
    return LocalEmbeddingProvider()


def create_knowledge_base(
    path: str = "kb",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> KnowledgeBase:
    """Build a ``KnowledgeBase`` from ``path`` using the best available provider."""
    return KnowledgeBase.from_directory(path, _select_provider(), chunk_size, chunk_overlap)


def format_context_for_prompt(results: Sequence[RetrievalResult], max_chars: int = MAX_RESULT_CHARS) -> str:
    """Format retrieval results into a compact, citable context block."""
    if not results:
        return ""
    lines: list[str] = []
    for i, result in enumerate(results, 1):
        excerpt = result.chunk.content[:max_chars]
        category = result.chunk.category or "General"
        lines.append(
            f"[{i}] KB Title: \"{result.chunk.title}\" "
            f"(category: {category}) | relevance: {result.score:.2f} | source: {result.chunk.source}"
        )
        lines.append(f"    Excerpt: {excerpt}")
    return "\n".join(lines)
