"""Adaptive Cross-Client Dispersion Defense (ACD-FedRAG) for Federated RAG.

ACD-FedRAG overcomes MIRABEL's catastrophic utility drop on high-relevance queries
by replacing blind candidate deletion with multi-signal risk detection and targeted demotion.

Detection Signals:
1. Local Score Gap (Delta_local): Checks if the score difference between rank-1 and
   rank-2 candidates within the top client meets or exceeds local_gap_threshold.
2. Cross-Client Dispersion (Delta_cross): Measures the standard deviation of top-1
   scores across all clients to identify unnatural cross-client score spikes.

Targeted Action:
When active signals trigger and top score >= abs_score_threshold, ACD-FedRAG
demotes the suspicious top candidate's similarity score to the client's local
baseline score rather than deleting it. This preserves context utility for legitimate
retrieval while attenuating outlier confidence spikes exploited by MIA adversaries.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class ACDFedRAGDefense:
    """Adaptive Cross-Client Dispersion Defense with targeted score demotion."""

    def __init__(
        self,
        local_gap_threshold: float = 0.15,
        cross_client_std_threshold: float = 0.10,
        abs_score_threshold: float = 0.85,
        use_local: bool = True,
        use_cross: bool = True,
        baseline_strategy: str = "rank2",
    ) -> None:
        """Initialize the ACD-FedRAG defense engine.

        Args:
            local_gap_threshold: Minimum gap between rank-1 and rank-2 scores in top client (default: 0.15).
            cross_client_std_threshold: Minimum standard deviation of top-1 scores across clients (default: 0.10).
            abs_score_threshold: Minimum absolute similarity score to trigger inspection (default: 0.85).
            use_local: Whether to incorporate local score gap signal (for ablation).
            use_cross: Whether to incorporate cross-client dispersion signal (for ablation).
            baseline_strategy: Strategy for local baseline demotion ('rank2', 'mean', or 'min').
        """
        self.local_gap_threshold: float = float(local_gap_threshold)
        self.cross_client_std_threshold: float = float(cross_client_std_threshold)
        self.abs_score_threshold: float = float(abs_score_threshold)
        self.use_local: bool = bool(use_local)
        self.use_cross: bool = bool(use_cross)
        self.baseline_strategy: str = baseline_strategy

        self.last_triggered: bool = False
        self.total_inspections: int = 0
        self.total_demotions: int = 0

    def inspect_and_defend(
        self,
        client_results: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """Inspects candidate sets using local gap and cross-client dispersion signals.

        If triggered, demotes the top candidate's score to the client's local baseline
        score without deleting the document.

        Args:
            client_results: List of per-client dictionaries or flattened candidate dictionaries.

        Returns:
            Tuple[List[Dict[str, Any]], bool]:
                - modified_results: Candidate sets with demoted top candidate score if triggered.
                - defense_triggered: Boolean flag indicating if demotion was applied.
        """
        self.total_inspections += 1
        if not client_results:
            self.last_triggered = False
            return client_results, False

        # Support flattened candidate list
        is_flattened = "score" in client_results[0] and "documents" not in client_results[0]
        if is_flattened:
            return self._defend_flattened(client_results)

        # 1. Identify global top candidate and collect client top-1 scores
        max_score = -float("inf")
        top_client_idx: Optional[int] = None
        top_doc_idx: Optional[int] = None
        top1_scores: List[float] = []

        for c_idx, c_res in enumerate(client_results):
            scores = c_res.get("scores", [])
            if scores:
                top1 = float(scores[0])
                top1_scores.append(top1)
                if top1 > max_score:
                    max_score = top1
                    top_client_idx = c_idx
                    top_doc_idx = 0

        # Absolute score gating
        if top_client_idx is None or max_score < self.abs_score_threshold:
            self.last_triggered = False
            return client_results, False

        # 2. Local Signal (Delta_local): Rank-1 vs Rank-2 score gap in top client
        local_triggered = False
        top_client_scores = client_results[top_client_idx].get("scores", [])
        if len(top_client_scores) >= 2:
            local_gap = float(top_client_scores[0]) - float(top_client_scores[1])
            local_triggered = local_gap >= self.local_gap_threshold
        elif len(top_client_scores) == 1:
            local_triggered = True  # Single document with high score is isolated

        # 3. Cross-Client Signal (Delta_cross): Standard deviation of top-1 scores across clients
        cross_triggered = False
        if len(top1_scores) >= 2:
            cross_std = float(np.std(top1_scores))
            cross_triggered = cross_std >= self.cross_client_std_threshold
        else:
            cross_std = 0.0
            cross_triggered = False

        # 4. Multi-Signal Decision Logic
        if self.use_local and self.use_cross:
            signals_active = local_triggered and cross_triggered
        elif self.use_local:
            signals_active = local_triggered
        elif self.use_cross:
            signals_active = cross_triggered
        else:
            signals_active = False

        if not signals_active:
            self.last_triggered = False
            return client_results, False

        # 5. Targeted Action: Demote top candidate's score to local baseline score
        defense_triggered = True
        self.total_demotions += 1

        filtered_results: List[Dict[str, Any]] = []
        for c_idx, c_res in enumerate(client_results):
            new_c_res = {
                "client_id": c_res.get("client_id"),
                "domain": c_res.get("domain"),
                "documents": list(c_res.get("documents", [])),
                "scores": list(c_res.get("scores", [])),
            }

            if c_idx == top_client_idx and new_c_res["scores"]:
                scores_list = new_c_res["scores"]
                if len(scores_list) >= 2:
                    if self.baseline_strategy == "mean":
                        baseline_score = float(np.mean(scores_list[1:]))
                    elif self.baseline_strategy == "min":
                        baseline_score = float(min(scores_list[1:]))
                    else:  # 'rank2' default
                        baseline_score = float(scores_list[1])
                else:
                    baseline_score = 0.0

                # Demote the outlier top score to the client's local baseline
                scores_list[0] = baseline_score

            filtered_results.append(new_c_res)

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

        # Sort copy to inspect
        sorted_cands = sorted(candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        top_cand = sorted_cands[0]
        top_score = float(top_cand.get("score", 0.0))

        if top_score < self.abs_score_threshold:
            self.last_triggered = False
            return candidates, False

        # Local gap
        local_triggered = False
        if len(sorted_cands) >= 2:
            gap = top_score - float(sorted_cands[1].get("score", 0.0))
            local_triggered = gap >= self.local_gap_threshold
        else:
            local_triggered = True

        cross_triggered = local_triggered  # In flattened view without client grouping, mirror signal

        if self.use_local and self.use_cross:
            signals_active = local_triggered and cross_triggered
        elif self.use_local:
            signals_active = local_triggered
        elif self.use_cross:
            signals_active = cross_triggered
        else:
            signals_active = False

        if not signals_active:
            self.last_triggered = False
            return candidates, False

        # Demote top candidate score
        defense_triggered = True
        self.total_demotions += 1

        baseline_score = float(sorted_cands[1].get("score", 0.0)) if len(sorted_cands) >= 2 else 0.0
        new_cands = []
        for cand in candidates:
            c_copy = dict(cand)
            if cand is top_cand or (cand.get("doc_id") == top_cand.get("doc_id") and float(cand.get("score", 0.0)) == top_score):
                c_copy["score"] = baseline_score
            new_cands.append(c_copy)

        self.last_triggered = defense_triggered
        return new_cands, defense_triggered

    def reset_stats(self) -> None:
        """Reset inspection and demotion statistics."""
        self.total_inspections = 0
        self.total_demotions = 0
        self.last_triggered = False
