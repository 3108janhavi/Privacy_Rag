"""UI Helper functions interfacing backend Federated RAG modules with Streamlit.

Provides cached initialization, live query execution, attack evaluation,
data formatting into pandas DataFrames, and Plotly visualization builders.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data.prepare_data import DEFAULT_SPLITS_PATH, load_client_splits
from run_phase2_attack import SYNTHETIC_NON_MEMBERS
from src.attacks.mia_base import MembershipBasedAttack
from src.client import ClientNode
from src.coordinator import FederatedCoordinator
from src.defense.acd_fedrag import ACDFedRAGDefense
from src.defense.mirabel import MIRABELDefense
from src.router.adaptive_router import QueryAnalyzerAndRouter
from src.utils.metrics import compute_privacy_metrics, compute_retrieval_utility


CLIENT_METADATA: Dict[int, Dict[str, str]] = {
    1: {"domain": "Healthcare / Medical", "badge": "🏥 Medical", "color": "#10B981"},
    2: {"domain": "Education / Academics", "badge": "🎓 Education", "color": "#3B82F6"},
    3: {"domain": "Industrial / Engineering", "badge": "⚙️ Industrial", "color": "#F59E0B"},
    4: {"domain": "General Knowledge / History", "badge": "📜 History", "color": "#8B5CF6"},
}

PRESET_QUERIES: Dict[str, Dict[str, Any]] = {
    "Healthcare: Diabetes Management": {
        "query": "What are recommended clinical treatments and management for type 2 diabetes?",
        "expected_client_id": 1,
        "domain": "Healthcare / Medical",
    },
    "Education: Cognitive Load": {
        "query": "How does cognitive load theory impact instructional design and formative assessment?",
        "expected_client_id": 2,
        "domain": "Education / Academics",
    },
    "Industrial: Predictive Maintenance": {
        "query": "Predictive maintenance vibration analysis for industrial CNC machinery and bearings",
        "expected_client_id": 3,
        "domain": "Industrial / Engineering",
    },
    "History: Fall of Rome & Renaissance": {
        "query": "What factors led to the fall of the Western Roman Empire and rise of the Renaissance?",
        "expected_client_id": 4,
        "domain": "General Knowledge / History",
    },
    "Multi-Domain: Industrial History & Education": {
        "query": "How did the history of the industrial revolution transform factory machinery and manufacturing engineering education?",
        "expected_client_id": None,
        "domain": "Multi-Domain",
    },
    "Verbatim Probing Attack (MED-002)": {
        "query": "Type 2 diabetes mellitus clinical guidelines recommend metformin as first-line pharmacotherapy alongside continuous glucose monitoring and HbA1c tracking.",
        "expected_client_id": 1,
        "domain": "Healthcare / Medical (Probe)",
    },
}


@st.cache_resource(show_spinner="Initializing Federated RAG Engine & Embedding Model...")
def get_cached_federated_system() -> Tuple[FederatedCoordinator, List[ClientNode], Dict[int, List[Dict[str, Any]]]]:
    """Initializes and caches ClientNodes and FederatedCoordinator to optimize UI performance."""
    splits = load_client_splits(DEFAULT_SPLITS_PATH)
    clients: List[ClientNode] = []
    shared_model = None

    for cid in sorted(splits.keys()):
        d_name = CLIENT_METADATA.get(cid, {}).get("domain", splits[cid][0]["domain"])
        client = ClientNode(client_id=cid, domain_name=d_name, model=shared_model)
        if shared_model is None:
            shared_model = client.model
        client.populate_index(splits[cid])
        clients.append(client)

    coordinator = FederatedCoordinator(clients=clients, model=shared_model)
    return coordinator, clients, splits


def get_dataset_overview_df() -> pd.DataFrame:
    """Returns a DataFrame summarizing the 4 client domain document sets."""
    splits = load_client_splits(DEFAULT_SPLITS_PATH)
    rows = []
    for cid, docs in splits.items():
        meta = CLIENT_METADATA.get(cid, {})
        for d in docs:
            rows.append({
                "Client ID": f"Client {cid}",
                "Domain": meta.get("domain", d["domain"]),
                "Doc ID": d["doc_id"],
                "Document Text": d["text"],
                "Text Length (words)": len(d["text"].split()),
            })
    return pd.DataFrame(rows)


def run_single_search(
    coordinator: FederatedCoordinator,
    query: str,
    top_k_per_client: int = 5,
    global_top_k: int = 5,
    defense: Optional[Any] = None,
    router: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], float]:
    """Executes a search query and returns a formatted DataFrame, metadata, and latency (ms)."""
    t0 = time.perf_counter()
    results = coordinator.search(
        query=query,
        top_k_per_client=top_k_per_client,
        global_top_k=global_top_k,
        defense=defense,
        router=router,
    )
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    meta = {
        "defense_triggered": coordinator.last_defense_triggered,
        "selected_clients": [c.client_id for c in coordinator.last_selected_clients],
        "query_analysis": coordinator.last_query_analysis,
    }

    rows = []
    for rank, doc in enumerate(results, start=1):
        cid = doc.get("client_id", 0)
        c_badge = CLIENT_METADATA.get(cid, {}).get("badge", f"Client {cid}")
        rows.append({
            "Rank": rank,
            "Doc ID": doc.get("doc_id", "N/A"),
            "Client": c_badge,
            "Domain": doc.get("domain", "N/A"),
            "Similarity Score": f"{doc.get('score', 0.0):.4f}",
            "Raw Score": float(doc.get("score", 0.0)),
            "Document Snippet": doc.get("text", ""),
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Rank", "Doc ID", "Client", "Domain", "Similarity Score", "Raw Score", "Document Snippet"])
    return df, meta, latency_ms


def run_live_mia_evaluation(
    coordinator: FederatedCoordinator,
    defense: Optional[Any] = None,
    progress_bar: Optional[Any] = None,
) -> Dict[str, Any]:
    """Runs the MembershipBasedAttack suite across 32 members and 32 non-members."""
    splits = load_client_splits(DEFAULT_SPLITS_PATH)
    all_members = [doc for docs in splits.values() for doc in docs]
    non_members = SYNTHETIC_NON_MEMBERS

    coordinator.defense = defense
    attacker = MembershipBasedAttack()

    y_true: List[int] = []
    y_scores: List[float] = []
    member_recalls: List[float] = []
    member_ndcgs: List[float] = []

    total_targets = len(all_members) + len(non_members)
    processed = 0

    # Members
    for doc in all_members:
        res = attacker.evaluate_target(coordinator, doc, is_member=True, top_k_per_client=5, global_top_k=5)
        y_true.append(1)
        y_scores.append(res["attack_score"])

        u = compute_retrieval_utility([doc["doc_id"]], res["retrieved_docs"], k=5)
        member_recalls.append(u["recall@5"])
        member_ndcgs.append(u["ndcg@5"])

        processed += 1
        if progress_bar:
            progress_bar.progress(processed / total_targets, text=f"Evaluating Targets: {processed}/{total_targets}")

    # Non-Members
    for doc in non_members:
        res = attacker.evaluate_target(coordinator, doc, is_member=False, top_k_per_client=5, global_top_k=5)
        y_true.append(0)
        y_scores.append(res["attack_score"])

        processed += 1
        if progress_bar:
            progress_bar.progress(processed / total_targets, text=f"Evaluating Targets: {processed}/{total_targets}")

    pm = compute_privacy_metrics(y_true, y_scores)

    scores_df = pd.DataFrame({
        "Target Status": ["Member (y=1)"] * len(all_members) + ["Non-Member (y=0)"] * len(non_members),
        "Attacker Confidence Score": y_scores,
    })

    return {
        "roc_auc": pm["roc_auc"],
        "tpr_at_1_percent_fpr": pm["tpr_at_1_percent_fpr"],
        "recall@5": float(np.mean(member_recalls)),
        "ndcg@5": float(np.mean(member_ndcgs)),
        "scores_df": scores_df,
        "y_true": y_true,
        "y_scores": y_scores,
    }


def build_score_distribution_chart(scores_df: pd.DataFrame) -> go.Figure:
    """Builds a Plotly histogram contrasting Member vs Non-Member confidence scores."""
    fig = px.histogram(
        scores_df,
        x="Attacker Confidence Score",
        color="Target Status",
        barmode="overlay",
        nbins=25,
        color_discrete_map={"Member (y=1)": "#EF4444", "Non-Member (y=0)": "#10B981"},
        title="Attacker Confidence Score Distribution ($S_{attack}$)",
        opacity=0.75,
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Aggregated Attacker Confidence Score (0.0 to 1.0)",
        yaxis_title="Count of Evaluated Documents",
        legend_title="Ground Truth",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_phase_comparison_chart() -> go.Figure:
    """Builds a Plotly bar chart comparing ROC-AUC vs Recall@5 across all benchmarked systems."""
    phases = [
        "Undefended Baseline",
        "MIRABEL (tau=0.85)",
        "ACD-FedRAG (Proposed)",
        "Adaptive + ACD-FedRAG",
    ]
    roc_auc = [1.0000, 0.9746, 1.0000, 1.0000]
    recall_5 = [1.0000, 0.0000, 1.0000, 0.9111]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=phases,
        y=roc_auc,
        name="MIA Vulnerability (ROC-AUC)",
        marker_color="#EF4444",
        text=[f"{v:.4f}" for v in roc_auc],
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        x=phases,
        y=recall_5,
        name="Retrieval Utility (Recall@5)",
        marker_color="#3B82F6",
        text=[f"{v:.4f}" for v in recall_5],
        textposition="auto",
    ))

    fig.update_layout(
        barmode="group",
        template="plotly_white",
        title="Privacy-Utility Tradeoff: Privacy Vulnerability vs Retained Retrieval Utility",
        yaxis=dict(title="Score (0.0 to 1.0)", range=[0, 1.15]),
        xaxis=dict(title="System / Defense Architecture"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig
