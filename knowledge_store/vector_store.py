"""
ChromaDB vector store for prose chunks and table descriptions.

Used for semantic search: "what conditions apply to bulk carriers at Durban?"
NOT for exact numeric lookups — use tariff_store.py for that.

Uses a custom embedding function built on google.genai (new SDK) because
ChromaDB's built-in GoogleGenerativeAiEmbeddingFunction still depends on
the deprecated google.generativeai package which conflicts with our setup.
"""

from __future__ import annotations

import logging
from typing import List

import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from google import genai

import config

logger = logging.getLogger(__name__)

_client: chromadb.PersistentClient | None = None
_collection = None


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function using google.genai (new SDK)."""

    def __init__(self, api_key: str, model_name: str):
        self._genai_client = genai.Client(api_key=api_key)
        self._model = model_name

    def __call__(self, input: Documents) -> Embeddings:
        result = self._genai_client.models.embed_content(
            model=self._model,
            contents=list(input),
        )
        return [list(e.values) for e in result.embeddings]


def _get_collection():
    global _client, _collection

    if _collection is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

        # Try Gemini embeddings first; fall back to ChromaDB's built-in
        # local sentence-transformer model (no API key needed) if unavailable.
        embedding_fn = None
        if config.GEMINI_API_KEY:
            try:
                test_client = genai.Client(api_key=config.GEMINI_API_KEY)
                test_client.models.embed_content(
                    model=config.GEMINI_EMBEDDING_MODEL,
                    contents=["test"],
                )
                embedding_fn = GeminiEmbeddingFunction(
                    api_key=config.GEMINI_API_KEY,
                    model_name=config.GEMINI_EMBEDDING_MODEL,
                )
                logger.info("Vector store: using Gemini embeddings")
            except Exception as e:
                logger.warning(
                    f"Gemini embeddings unavailable ({e.__class__.__name__}), "
                    "falling back to local sentence-transformer embeddings"
                )

        # Fall back to ChromaDB's built-in local embedding function
        if embedding_fn is None:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            embedding_fn = DefaultEmbeddingFunction()
            logger.info("Vector store: using local DefaultEmbeddingFunction (all-MiniLM-L6-v2)")

        _collection = _client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    return _collection


def save_prose_chunks(port_name: str, chunks: list) -> int:
    if not chunks:
        return 0

    collection = _get_collection()
    saved = 0

    for i, chunk in enumerate(chunks):
        if not chunk.text.strip():
            continue

        doc_id = f"{port_name}_prose_{chunk.page_number}_{i}"

        collection.upsert(
            ids=[doc_id],
            documents=[chunk.text],
            metadatas=[{
                "port": port_name,
                "page_number": chunk.page_number,
                "section_heading": chunk.section_heading,
                "source": chunk.source,
                "type": "prose",
            }],
        )
        saved += 1

    logger.info(f"Vector store: saved {saved} prose chunks for '{port_name}'")
    return saved


def save_table_description(
    port_name: str,
    charge_type: str,
    description: str,
    page_number: int,
) -> None:
    if not description.strip():
        return

    collection = _get_collection()
    doc_id = f"{port_name}_table_{charge_type}"

    collection.upsert(
        ids=[doc_id],
        documents=[description],
        metadatas=[{
            "port": port_name,
            "charge_type": charge_type,
            "page_number": page_number,
            "type": "table_description",
        }],
    )


def query(
    question: str,
    port_name: str | None = None,
    n_results: int = 5,
) -> list[dict]:
    """
    Semantic search over prose rules and table descriptions.
    Returns list of { text, metadata, distance }.
    """
    collection = _get_collection()

    where: dict | None = None
    if port_name:
        where = {"port": port_name}

    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        where=where,
    )

    out = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        out.append({"text": doc, "metadata": meta, "distance": dist})

    return out
