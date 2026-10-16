"""Federated Client Node implementation for local vector search.

Each ClientNode manages:
- Local non-IID domain documents
- SentenceTransformer embeddings computation
- FAISS IndexFlatIP (Inner Product on L2-normalized vectors for exact Cosine Similarity)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class ClientNode:
    """Represents a federated client holding private, domain-specific documents."""

    def __init__(
        self,
        client_id: int,
        domain_name: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model: Optional[SentenceTransformer] = None,
    ) -> None:
        """Initialize the federated client node.

        Args:
            client_id: Unique integer identifier for this client.
            domain_name: Domain description (e.g., 'Healthcare / Medical').
            model_name: HuggingFace model identifier for sentence embeddings.
            model: Optional pre-instantiated SentenceTransformer instance for memory sharing.
        """
        self.client_id: int = client_id
        self.domain_name: str = domain_name
        self.model_name: str = model_name

        self.model: SentenceTransformer = (
            model if model is not None else SentenceTransformer(model_name)
        )
        self.documents: List[Dict[str, Any]] = []
        self.index: Optional[faiss.IndexFlatIP] = None
        self.embedding_dimension: Optional[int] = None

    def populate_index(self, documents: List[Dict[str, Any]]) -> None:
        """Encodes text documents into normalized embeddings and builds a FAISS index.

        Args:
            documents: List of document dicts, each with keys 'doc_id', 'text', 'domain'.
        """
        if not documents:
            self.documents = []
            self.index = None
            return

        self.documents = list(documents)
        texts: List[str] = [doc["text"] for doc in self.documents]

        # Generate dense embeddings with L2 normalization enabled
        raw_embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings = np.ascontiguousarray(raw_embeddings, dtype=np.float32)

        # Enforce exact unit norm for Cosine Similarity equivalence under Inner Product
        faiss.normalize_L2(embeddings)

        self.embedding_dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """Executes a local FAISS Inner Product search using the provided query embedding.

        Args:
            query_embedding: Numpy array containing query vector (1D or 2D).
            top_k: Maximum number of top candidates to retrieve locally.

        Returns:
            Dict containing:
                - client_id (int)
                - domain (str)
                - documents (List[Dict]): Retrieved metadata list
                - scores (List[float]): Cosine similarity scores bounded in [-1.0, 1.0]
        """
        if self.index is None or self.index.ntotal == 0 or len(self.documents) == 0:
            return {
                "client_id": self.client_id,
                "domain": self.domain_name,
                "documents": [],
                "scores": [],
            }

        # Ensure query vector is 2D float32 and contiguous
        query_vec = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = np.expand_dims(query_vec, axis=0)

        # Normalize query vector to unit length for exact cosine similarity
        faiss.normalize_L2(query_vec)

        # Clamp k to available indexed vectors
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        retrieved_docs: List[Dict[str, Any]] = []
        retrieved_scores: List[float] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            # Clip numerical precision drift to strict [-1.0, 1.0]
            bounded_score = max(-1.0, min(1.0, float(score)))
            retrieved_docs.append(doc)
            retrieved_scores.append(bounded_score)

        return {
            "client_id": self.client_id,
            "domain": self.domain_name,
            "documents": retrieved_docs,
            "scores": retrieved_scores,
        }
