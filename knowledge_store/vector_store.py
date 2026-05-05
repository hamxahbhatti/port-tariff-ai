"""
ChromaDB vector store for prose chunks and table descriptions.

Used for semantic search: "what conditions apply to bulk carriers at Durban?"
NOT for exact numeric lookups — use tariff_store.py for that.
"""

from __future__ import annotations

import logging

import chromadb
from chromadb.utils import embedding_functions

import config
from ingestion.docling_parser import ProseChunk

logger = logging.getLogger(__name__)

_client: chromadb.PersistentClient | None = None
_collection = None


def _get_collection():
    global _client, _collection

    if _collection is None:
        _client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))

        gemini_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
            api_key=config.GEMINI_API_KEY,
            model_name=config.GEMINI_EMBEDDING_MODEL,
        )

        _collection = _client.get_or_create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            embedding_function=gemini_ef,
            metadata={"hnsw:space": "cosine"},
        )

    return _collection


def save_prose_chunks(port_name: str, chunks: list[ProseChunk]) -> int:
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
