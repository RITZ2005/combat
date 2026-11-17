"""
Multi-Doc Router API — LangGraph graph definition.

Uses:
  - Ollama BGE-M3 for embeddings
  - Ollama Llama 3.2 for answer generation
  - ChromaDB for per-document vector stores

Nodes:
  1. router  — embeds the question, compares to doc summaries, picks the best match
  2. answer  — retrieves relevant chunks from the chosen doc's collection and
               generates a natural-language answer with Llama 3.2
"""

from __future__ import annotations

import os
from typing import Optional

import chromadb
import numpy as np
import ollama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EMBED_MODEL = "bge-m3"
CHAT_MODEL = "llama3.2"
AMBIGUITY_GAP_THRESHOLD = 0.05   # min gap between best and 2nd-best score
MIN_CONFIDENCE_SCORE = 0.25      # min absolute score for best match
CHUNK_SIZE = 200                 # words per chunk
CHUNK_OVERLAP = 30               # overlap in words
TOP_K_CHUNKS = 3                 # chunks to retrieve for the answer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Document registry
# ---------------------------------------------------------------------------

DOCUMENTS = [
    {
        "doc_id": "photosynthesis",
        "title": "Photosynthesis in Plants",
        "summary": (
            "How plants convert sunlight into energy through photosynthesis, "
            "including chloroplasts, light reactions, and the Calvin cycle."
        ),
        "file": os.path.join(BASE_DIR, "data", "photosynthesis.txt"),
    },
    {
        "doc_id": "black_holes",
        "title": "Black Holes and Spacetime",
        "summary": (
            "Astrophysics of black holes, event horizons, singularities, "
            "Hawking radiation, and their effect on spacetime."
        ),
        "file": os.path.join(BASE_DIR, "data", "black_holes.txt"),
    },
    {
        "doc_id": "python_programming",
        "title": "Python Programming Basics",
        "summary": (
            "Introduction to Python programming language, variables, data types, "
            "functions, loops, and object-oriented programming."
        ),
        "file": os.path.join(BASE_DIR, "data", "python_programming.txt"),
    },
]

# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _embed(texts: str | list[str]) -> list[list[float]]:
    """Embed one or more texts using BGE-M3 via Ollama."""
    if isinstance(texts, str):
        texts = [texts]
    resp = ollama.embed(model=EMBED_MODEL, input=texts)
    return resp.embeddings


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    return float(
        np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-10)
    )


# ---------------------------------------------------------------------------
# Chunking helper
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks or [text]


# ---------------------------------------------------------------------------
# ChromaDB setup — one collection per document
# ---------------------------------------------------------------------------

_chroma_client: chromadb.ClientAPI | None = None
_collections: dict[str, chromadb.Collection] = {}
_doc_summary_embeddings: dict[str, list[float]] = {}


def _get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        db_path = os.path.join(BASE_DIR, "chroma_db")
        _chroma_client = chromadb.PersistentClient(path=db_path)
    return _chroma_client


def _init_collections():
    """Create ChromaDB collections and index document chunks (idempotent)."""
    global _collections, _doc_summary_embeddings

    if _collections:
        return  # already initialised

    client = _get_chroma_client()

    for doc in DOCUMENTS:
        col_name = f"doc_{doc['doc_id']}"

        # Get or create the collection
        collection = client.get_or_create_collection(
            name=col_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Only index if the collection is empty
        if collection.count() == 0:
            with open(doc["file"], "r", encoding="utf-8") as f:
                content = f.read()

            chunks = _chunk_text(content)
            embeddings = _embed(chunks)

            collection.add(
                ids=[f"{doc['doc_id']}_chunk_{i}" for i in range(len(chunks))],
                embeddings=embeddings,
                documents=chunks,
                metadatas=[{"doc_id": doc["doc_id"], "chunk_idx": i} for i in range(len(chunks))],
            )

        _collections[doc["doc_id"]] = collection

        # Pre-compute summary embedding for routing
        summary_text = f"{doc['title']}. {doc['summary']}"
        _doc_summary_embeddings[doc["doc_id"]] = _embed(summary_text)[0]


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class RouterState(BaseModel):
    """State flowing through the LangGraph."""
    question: str = ""
    doc_scores: dict = Field(default_factory=dict)
    chosen_doc: Optional[str] = None
    chosen_doc_id: Optional[str] = None
    reason: str = ""
    is_ambiguous: bool = False
    answer: Optional[str] = None


# ---------------------------------------------------------------------------
# Node 1 — Router
# ---------------------------------------------------------------------------

def router_node(state: RouterState) -> dict:
    """Decide which document the question is about using embedding similarity."""
    _init_collections()

    q_emb = _embed(state.question)[0]

    scores: dict[str, float] = {}
    for doc in DOCUMENTS:
        sim = _cosine_similarity(q_emb, _doc_summary_embeddings[doc["doc_id"]])
        scores[doc["doc_id"]] = round(sim, 4)

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_id, best_score = sorted_scores[0]
    second_id, second_score = sorted_scores[1]
    gap = best_score - second_score

    # Ambiguity / low-confidence check
    if best_score < MIN_CONFIDENCE_SCORE or gap < AMBIGUITY_GAP_THRESHOLD:
        return {
            "doc_scores": scores,
            "chosen_doc": None,
            "chosen_doc_id": None,
            "reason": (
                f"ambiguous — matches multiple docs "
                f"(best: {best_id}={best_score:.2f}, "
                f"next: {second_id}={second_score:.2f}, gap={gap:.2f})"
            ),
            "is_ambiguous": True,
            "answer": None,
        }

    chosen = next(d for d in DOCUMENTS if d["doc_id"] == best_id)
    return {
        "doc_scores": scores,
        "chosen_doc": chosen["title"],
        "chosen_doc_id": best_id,
        "reason": (
            f"closest topic match: {best_score:.2f} similarity "
            f"vs {second_score:.2f} for next best ({second_id})"
        ),
        "is_ambiguous": False,
    }


# ---------------------------------------------------------------------------
# Node 2 — Answer (retrieval + generation)
# ---------------------------------------------------------------------------

def answer_node(state: RouterState) -> dict:
    """Retrieve relevant chunks from ChromaDB, then generate answer with Llama 3.2."""
    _init_collections()

    collection = _collections[state.chosen_doc_id]
    q_emb = _embed(state.question)[0]

    # Retrieve top-k chunks
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=min(TOP_K_CHUNKS, collection.count()),
    )

    context_chunks = results["documents"][0]  # list of chunk strings
    context = "\n\n".join(context_chunks)

    # Generate answer with Llama 3.2
    system_prompt = (
        "You are a helpful assistant. Answer the user's question based ONLY on the "
        "provided context. If the context does not contain enough information to answer, "
        "say so. Keep the answer concise and accurate."
    )
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {state.question}\n\n"
        f"Answer:"
    )

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return {"answer": response.message.content.strip()}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------

def _route_decision(state: RouterState) -> str:
    return "end" if state.is_ambiguous else "answer"


def build_graph():
    workflow = StateGraph(RouterState)
    workflow.add_node("router", router_node)
    workflow.add_node("answer", answer_node)
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        _route_decision,
        {"answer": "answer", "end": END},
    )
    workflow.add_edge("answer", END)
    return workflow.compile()


# Compile once
_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run_router_graph(question: str) -> dict:
    """Run the full router → answer pipeline and return the API response dict."""
    graph = get_graph()
    result = graph.invoke(RouterState(question=question))

    # LangGraph may return a dict or a Pydantic model depending on version
    if isinstance(result, dict):
        return {
            "answer": result.get("answer"),
            "chosen_doc": result.get("chosen_doc"),
            "reason": result.get("reason", ""),
        }
    return {
        "answer": result.answer,
        "chosen_doc": result.chosen_doc,
        "reason": result.reason,
    }
