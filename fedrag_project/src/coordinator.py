"""Federated Coordinator implementation for federated retrieval.

The FederatedCoordinator:
- Manages connections to registered ClientNode instances
- Encodes queries into normalized embeddings once
- Selectively routes queries via dynamic QueryAnalyzerAndRouter
- Broadcasts the query embedding to target clients
- Collects candidate sets and raw cosine similarity scores
- Supports pluggable defense layers (e.g. MIRABELDefense, ACDFedRAGDefense)
- Performs global ranking and returns the top globally relevant documents
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from .client import ClientNode


class FederatedCoordinator:
    """Coordinates federated search queries across distributed client nodes."""

    def __init__(
        self,
        clients: List[ClientNode],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model: Optional[SentenceTransformer] = None,
        defense: Optional[Any] = None,
        router: Optional[Any] = None,
    ) -> None:
        """Initialize the federated coordinator.

        Args:
            clients: List of ClientNode instances registered with the coordinator.
            model_name: HuggingFace model identifier for sentence embeddings.
            model: Optional pre-instantiated SentenceTransformer instance.
            defense: Optional pluggable defense instance (e.g., MIRABELDefense, ACDFedRAGDefense).
            router: Optional query-sensitive router (e.g., QueryAnalyzerAndRouter).
        """
        self.clients: List[ClientNode] = list(clients)
        self.model_name: str = model_name
        self.model: SentenceTransformer = (
            model if model is not None else SentenceTransformer(model_name)
        )
        self.defense: Optional[Any] = defense
        self.router: Optional[Any] = router

        self.last_defense_triggered: bool = False
        self.last_query_analysis: Optional[Dict[str, Any]] = None
        self.last_selected_clients: List[ClientNode] = []

    def search(
        self,
        query: str,
        top_k_per_client: int = 5,
        global_top_k: int = 5,
        defense: Optional[Any] = None,
        router: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Converts query to embedding once, selectively broadcasts to clients, and merges top results.

        If a router is active, selectively queries only relevant client partitions.
        If a defense layer is active, raw candidate sets pass through
        defense.inspect_and_defend() prior to global merging and ranking.

        Args:
            query: User search text query.
            top_k_per_client: Number of candidate documents to request from each client.
            global_top_k: Number of highest-scoring documents to return after global merging.
            defense: Optional per-call defense instance overriding coordinator default defense.
            router: Optional per-call router instance overriding coordinator default router.

        Returns:
            List[Dict]: Merged top documents formatted as:
                [
                    {
                        "doc_id": str,
                        "text": str,
                        "score": float,
                        "client_id": int,
                        "domain": str
                    },
                    ...
                ]
        """
        if not self.clients:
            return []

        # Convert query to normalized embedding once
        raw_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embedding = np.ascontiguousarray(raw_embedding, dtype=np.float32)

        # 1. Dynamic Client Routing
        active_router = router if router is not None else self.router
        if (
            active_router is not None
            and hasattr(active_router, "analyze_query")
            and hasattr(active_router, "select_clients")
        ):
            self.last_query_analysis = active_router.analyze_query(query)
            target_clients = active_router.select_clients(self.last_query_analysis, self.clients)
            self.last_selected_clients = target_clients
        else:
            self.last_query_analysis = None
            target_clients = self.clients
            self.last_selected_clients = list(self.clients)

        # 2. Broadcast query embedding ONLY to selected target clients
        raw_client_results: List[Dict[str, Any]] = []
        for client in target_clients:
            client_res = client.search(query_embedding, top_k=top_k_per_client)
            raw_client_results.append(client_res)

        # 3. Pluggable Defense Layer
        active_defense = defense if defense is not None else self.defense

        if active_defense is not None and hasattr(active_defense, "inspect_and_defend"):
            defense_output = active_defense.inspect_and_defend(raw_client_results)
            if isinstance(defense_output, tuple) and len(defense_output) == 2:
                client_results, triggered = defense_output
                self.last_defense_triggered = bool(triggered)
            else:
                client_results = defense_output
                self.last_defense_triggered = getattr(active_defense, "last_triggered", False)
        else:
            client_results = raw_client_results
            self.last_defense_triggered = False

        # 4. Global candidate collection
        all_candidates: List[Dict[str, Any]] = []

        if client_results and "score" in client_results[0] and "documents" not in client_results[0]:
            # Flattened candidate format
            all_candidates = list(client_results)
        else:
            # Per-client dictionary format
            for client_res in client_results:
                client_id = client_res.get("client_id", 0)
                domain = client_res.get("domain", "")
                docs = client_res.get("documents", [])
                scores = client_res.get("scores", [])

                for doc, score in zip(docs, scores):
                    candidate = {
                        "doc_id": doc["doc_id"],
                        "text": doc["text"],
                        "score": float(score),
                        "client_id": client_id,
                        "domain": domain,
                    }
                    all_candidates.append(candidate)

        # Sort globally in descending order of similarity score
        all_candidates.sort(key=lambda item: item["score"], reverse=True)

        # Return the merged top global_top_k documents
        return all_candidates[:global_top_k]
