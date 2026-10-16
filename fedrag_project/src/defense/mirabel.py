"""MIRABEL Defense implementation for Federated RAG.

MIRABEL (Membership Inference Reduction by Adaptive Boundary Enforcement in Language/Retrieval)
inspects returned similarity scores across client candidate sets.
When the top similarity score exceeds a tuned threshold (indicating an adversarial
verbatim or near-verbatim probing query), MIRABEL suppresses (removes/redacts)
the top-ranked document from the merged context, shifting remaining candidates up.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union


class MIRABELDefense:
    """Similarity-threshold inspection and top-candidate suppression defense."""

    def __init__(self, threshold: float = 0.85) -> None:
        """Initialize the MIRABEL defense engine.

        Args:
            threshold: Cosine similarity threshold for triggering suppression (default: 0.85).
        """
        self.threshold: float = float(threshold)
        self.last_triggered: bool = False
        self.total_inspections: int = 0
        self.total_suppressions: int = 0

    def inspect_and_defend(
        self,
        client_results: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Inspects candidate similarity scores across all clients and applies suppression.

        If the top candidate score s_top >= threshold:
            - defense_triggered is True
            - The top-ranked candidate document is removed/suppressed from context
            - Remaining candidates shift up in rank

        Args:
            client_results: Either a list of per-client result dictionaries:
                [{'client_id': int, 'domain': str, 'documents': List[Dict], 'scores': List[float]}, ...]
                or a list of flattened candidate dictionaries:
                [{'doc_id': str, 'text': str, 'score': float, 'client_id': int, 'domain': str}, ...]

        Returns:
            Tuple[List[Dict[str, Any]], bool]:
                - filtered_results: Updated candidate sets with top candidate suppressed if triggered
                - defense_triggered: Boolean flag indicating if suppression was applied
        """
        self.total_inspections += 1
        if not client_results:
            self.last_triggered = False
            return client_results, False

        # Check format: flattened candidate list vs per-client result dicts
        is_flattened = "score" in client_results[0] and "documents" not in client_results[0]

        if is_flattened:
            return self._defend_flattened(client_results)
        else:
            return self._defend_per_client(client_results)

    def _defend_per_client(
        self,
        client_results: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Handles per-client dictionary structure."""
        max_score = -float("inf")
        top_client_idx: Optional[int] = None
        top_doc_idx: Optional[int] = None

        # Inspect all scores across all client candidate sets
        for c_idx, c_res in enumerate(client_results):
            scores = c_res.get("scores", [])
            for d_idx, score in enumerate(scores):
                if float(score) > max_score:
                    max_score = float(score)
                    top_client_idx = c_idx
                    top_doc_idx = d_idx

        # Check if top score triggers the MIRABEL threshold
        if max_score >= self.threshold and top_client_idx is not None and top_doc_idx is not None:
            defense_triggered = True
            self.total_suppressions += 1

            # Deep copy to avoid mutating raw inputs in-place
            filtered_results: List[Dict[str, Any]] = []
            for c_idx, c_res in enumerate(client_results):
                new_c_res = {
                    "client_id": c_res.get("client_id"),
                    "domain": c_res.get("domain"),
                    "documents": list(c_res.get("documents", [])),
                    "scores": list(c_res.get("scores", [])),
                }

                if c_idx == top_client_idx:
                    # Remove / suppress the top-ranked candidate document and score
                    if top_doc_idx < len(new_c_res["documents"]):
                        new_c_res["documents"].pop(top_doc_idx)
                        new_c_res["scores"].pop(top_doc_idx)

                filtered_results.append(new_c_res)
        else:
            defense_triggered = False
            filtered_results = client_results

        self.last_triggered = defense_triggered
        return filtered_results, defense_triggered

    def _defend_flattened(
        self,
        candidates: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Handles flattened candidate list structure."""
        if not candidates:
            self.last_triggered = False
            return candidates, False

        # Find candidate with highest score
        max_idx, top_cand = max(enumerate(candidates), key=lambda x: float(x[1].get("score", 0.0)))
        top_score = float(top_cand.get("score", 0.0))

        if top_score >= self.threshold:
            defense_triggered = True
            self.total_suppressions += 1
            # Remove top candidate; remaining candidates shift up
            filtered_candidates = [cand for i, cand in enumerate(candidates) if i != max_idx]
        else:
            defense_triggered = False
            filtered_candidates = candidates

        self.last_triggered = defense_triggered
        return filtered_candidates, defense_triggered

    def reset_stats(self) -> None:
        """Reset inspection and suppression counters."""
        self.total_inspections = 0
        self.total_suppressions = 0
        self.last_triggered = False
