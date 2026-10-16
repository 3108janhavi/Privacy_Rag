"""Privacy and retrieval utility metrics for Federated RAG evaluation.

Provides:
- Privacy leakage metrics: ROC-AUC and TPR @ 1% FPR for Membership Inference Attacks (MIA).
- Retrieval quality metrics: Recall@k and nDCG@k for evaluating task utility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def compute_privacy_metrics(
    y_true: List[int],
    y_scores: List[float],
) -> Dict[str, float]:
    """Calculate privacy vulnerability metrics under Membership Inference Attacks.

    Args:
        y_true: Ground truth binary membership labels (1 for member, 0 for non-member).
        y_scores: Continuous attack confidence scores in range [0, 1].

    Returns:
        Dict[str, float]:
            - 'roc_auc': Receiver Operating Characteristic Area Under the Curve.
            - 'tpr_at_1_percent_fpr': True Positive Rate at a fixed 1% False Positive Rate.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_scores_arr = np.asarray(y_scores, dtype=np.float64)

    if len(np.unique(y_true_arr)) < 2:
        return {
            "roc_auc": 0.5,
            "tpr_at_1_percent_fpr": 0.0,
        }

    # Compute ROC-AUC
    auc = float(roc_auc_score(y_true_arr, y_scores_arr))

    # Compute ROC curve to extract TPR at fixed 1% (0.01) FPR
    fpr, tpr, _ = roc_curve(y_true_arr, y_scores_arr)

    # Standard linear interpolation at FPR = 0.01
    tpr_at_1_pct = float(np.interp(0.01, fpr, tpr))

    return {
        "roc_auc": auc,
        "tpr_at_1_percent_fpr": tpr_at_1_pct,
    }


def compute_retrieval_utility(
    ground_truth_doc_ids: List[str],
    retrieved_docs: List[Dict[str, Any]],
    k: int = 5,
) -> Dict[str, float]:
    """Calculate retrieval utility metrics (Recall@k and nDCG@k).

    Args:
        ground_truth_doc_ids: List of ground truth relevant document IDs.
        retrieved_docs: List of retrieved document dictionaries with 'doc_id' key.
        k: Top-k ranking cutoff.

    Returns:
        Dict[str, float]:
            - 'recall@k': Recall at cutoff k.
            - 'ndcg@k': Normalized Discounted Cumulative Gain at cutoff k.
    """
    gt_set: Set[str] = set(ground_truth_doc_ids)
    if not gt_set:
        return {
            f"recall@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            "recall": 0.0,
            "ndcg": 0.0,
        }

    top_k_docs = retrieved_docs[:k]

    # Calculate Recall@k
    hits = sum(1 for d in top_k_docs if d.get("doc_id") in gt_set)
    recall = hits / len(gt_set)

    # Calculate Discounted Cumulative Gain (DCG@k) with binary relevance
    dcg = 0.0
    for idx, doc in enumerate(top_k_docs):
        if doc.get("doc_id") in gt_set:
            dcg += 1.0 / np.log2(idx + 2)

    # Calculate Ideal Discounted Cumulative Gain (IDCG@k)
    idcg = 0.0
    for idx in range(min(k, len(gt_set))):
        idcg += 1.0 / np.log2(idx + 2)

    ndcg = (dcg / idcg) if idcg > 0.0 else 0.0

    return {
        f"recall@{k}": float(recall),
        f"ndcg@{k}": float(ndcg),
        "recall": float(recall),
        "ndcg": float(ndcg),
    }
