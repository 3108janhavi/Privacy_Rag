"""Membership Inference Attack (MIA) Suite for Federated RAG.

Evaluates privacy leakage against an undefended Federated RAG system.
The adversary probes the system using verbatim snippets and key phrases
derived from candidate target documents, extracting retrieval presence,
similarity scores, and ranking positions to compute an attack confidence score.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class MembershipBasedAttack:
    """Membership Inference Attack evaluator targeting federated document stores."""

    def __init__(self, target_document: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MembershipBasedAttack.

        Args:
            target_document: Optional document dict with at least 'doc_id' and 'text'.
        """
        self.target_document: Optional[Dict[str, Any]] = target_document

    def generate_probing_queries(self, doc_text: str, n_queries: int = 3) -> List[str]:
        """Extract key text snippets and verbatim phrases from doc_text to form probing queries.

        Args:
            doc_text: Content string of the target document.
            n_queries: Number of distinct probing queries to generate.

        Returns:
            List[str]: Formatted probing queries tailored to elicit the target document.
        """
        cleaned_text = " ".join(doc_text.strip().split())
        words = cleaned_text.split()
        queries: List[str] = []

        # Query 1: Full verbatim text (strongest probing signal)
        queries.append(cleaned_text)

        # Split into clauses or sentence fragments if punctuation exists
        clauses = [
            c.strip()
            for c in re.split(r"[,;:.!?]\s*", cleaned_text)
            if len(c.strip().split()) >= 3
        ]

        if clauses:
            # Add prominent leading clause
            queries.append(clauses[0])
            # Add secondary / trailing clause
            if len(clauses) > 1:
                queries.append(clauses[1])
            else:
                # Add second-half snippet if only one clause
                mid = len(words) // 2
                queries.append(" ".join(words[mid:]))
        else:
            # Word-slice fallback for single-clause sentences
            if len(words) > 6:
                mid = len(words) // 2
                queries.append(" ".join(words[:mid]))
                queries.append(" ".join(words[mid:]))
            else:
                queries.append(cleaned_text)

        # Deduplicate while preserving order
        unique_queries: List[str] = []
        for q in queries:
            if q and q not in unique_queries:
                unique_queries.append(q)

        # If more queries needed, generate sliding window n-grams
        if len(unique_queries) < n_queries and len(words) > 8:
            window_size = min(8, len(words) - 2)
            start_idx = len(words) // 4
            window_query = " ".join(words[start_idx : start_idx + window_size])
            if window_query not in unique_queries:
                unique_queries.append(window_query)

        # Pad with full text if fewer queries were generated
        while len(unique_queries) < n_queries:
            unique_queries.append(cleaned_text)

        return unique_queries[:n_queries]

    def evaluate_target(
        self,
        coordinator: Any,
        doc: Optional[Dict[str, Any]] = None,
        is_member: bool = False,
        top_k_per_client: int = 5,
        global_top_k: int = 5,
        n_queries: int = 3,
    ) -> Dict[str, Any]:
        """Runs probing queries against FederatedCoordinator and calculates attack signals.

        Args:
            coordinator: FederatedCoordinator instance.
            doc: Target document dictionary {'doc_id': str, 'text': str, ...}.
                 Defaults to self.target_document if not provided.
            is_member: Ground truth membership flag (True if member, False if non-member).
            top_k_per_client: Candidates requested per client node.
            global_top_k: Global candidate set size.
            n_queries: Number of probing queries to generate.

        Returns:
            Dict[str, Any]:
                - 'doc_id': str
                - 'is_member': bool
                - 'target_present': bool (direct retrieval presence)
                - 'best_rank': Optional[int] (1-indexed rank of target document if retrieved)
                - 'max_similarity_score': float (highest similarity observed across queries)
                - 'target_max_score': float (highest similarity specifically for target doc)
                - 'attack_score': float (aggregated attacker confidence in [0, 1])
                - 'retrieved_docs': List of documents retrieved from the primary query
        """
        target_doc = doc if doc is not None else self.target_document
        if target_doc is None:
            raise ValueError("No target document provided for MIA evaluation.")

        doc_id = target_doc.get("doc_id", "UNKNOWN")
        doc_text = target_doc.get("text", "")

        probing_queries = self.generate_probing_queries(doc_text, n_queries=n_queries)

        all_observed_scores: List[float] = []
        target_scores: List[float] = []
        target_ranks: List[int] = []
        primary_retrieved_docs: List[Dict[str, Any]] = []

        normalized_target_text = " ".join(doc_text.strip().lower().split())

        for idx, query in enumerate(probing_queries):
            results = coordinator.search(
                query=query,
                top_k_per_client=top_k_per_client,
                global_top_k=global_top_k,
            )

            # Store the top results of the primary (full verbatim) query for utility evaluation
            if idx == 0:
                primary_retrieved_docs = results

            for rank_0, item in enumerate(results):
                score = float(item.get("score", 0.0))
                all_observed_scores.append(score)

                # Check for target presence by doc_id match or normalized text equivalence
                item_doc_id = item.get("doc_id", "")
                item_text = " ".join(item.get("text", "").strip().lower().split())

                is_target = (item_doc_id == doc_id) or (item_text == normalized_target_text)

                if is_target:
                    target_scores.append(score)
                    target_ranks.append(rank_0 + 1)  # 1-indexed rank

        # Extract attack signals
        max_similarity_score = max(all_observed_scores) if all_observed_scores else 0.0
        target_present = len(target_scores) > 0
        best_rank = min(target_ranks) if target_present else None
        target_max_score = max(target_scores) if target_present else 0.0

        # Compute aggregate attacker confidence score S_attack in [0, 1]
        if target_present and best_rank is not None:
            # Target was directly retrieved: high confidence scaled by similarity & top rank
            rank_factor = 1.0 / float(best_rank)
            # Calibration: base presence (0.20) + score weight (0.50) + rank weight (0.30)
            s_attack = 0.20 + 0.50 * max(0.0, target_max_score) + 0.30 * rank_factor
        else:
            # Target was NOT directly retrieved: low confidence derived only from background score
            # Non-members yield generic background scores typically < 0.45
            s_attack = 0.20 * max(0.0, min(1.0, (max_similarity_score - 0.40) / 0.60))

        # Ensure strict bounding in [0.0, 1.0]
        s_attack = float(max(0.0, min(1.0, s_attack)))

        return {
            "doc_id": doc_id,
            "is_member": is_member,
            "target_present": target_present,
            "best_rank": best_rank,
            "max_similarity_score": float(max_similarity_score),
            "target_max_score": float(target_max_score),
            "attack_score": s_attack,
            "retrieved_docs": primary_retrieved_docs,
        }
