"""Phase 2: Membership Inference Attack (MIA) Evaluation Framework.

Evaluates privacy leakage on the undefended Phase 1 Federated RAG base retrieval engine:
1. Loads client splits from data/client_splits.json (Members, y=1).
2. Generates domain-matched Non-Member documents (y=0).
3. Executes MembershipBasedAttack across all targets.
4. Computes Privacy Metrics: MIA ROC-AUC, TPR @ 1% FPR.
5. Computes Retrieval Utility Metrics: Baseline Recall@5 and nDCG@5.
6. Prints a comprehensive vulnerability report.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.prepare_data import DEFAULT_SPLITS_PATH, load_client_splits
from src.attacks.mia_base import MembershipBasedAttack
from src.client import ClientNode
from src.coordinator import FederatedCoordinator
from src.utils.metrics import compute_privacy_metrics, compute_retrieval_utility


# Domain-matched synthetic non-member documents (y=0) never indexed in any client
SYNTHETIC_NON_MEMBERS: List[Dict[str, str]] = [
    # Domain: Healthcare / Medical (8 non-members)
    {
        "doc_id": "NONMEM-MED-001",
        "text": "Pediatric asthma management guidelines emphasize inhaled corticosteroids and step-up therapy with long-acting beta-agonists.",
        "domain": "Healthcare / Medical",
    },
    {
        "doc_id": "NONMEM-MED-002",
        "text": "Alzheimer's disease biomarkers include amyloid-beta 42/40 ratios and phosphorylated tau in cerebrospinal fluid and plasma.",
        "domain": "Healthcare / Medical",
    },
    {
        "doc_id": "NONMEM-MED-003",
        "text": "CRISPR-Cas9 gene editing therapies target the BCL11A erythroid enhancer to treat sickle cell disease and transfusion-dependent beta-thalassemia.",
        "domain": "Healthcare / Medical",
    },
    {
        "doc_id": "NONMEM-MED-004",
        "text": "Chronic kidney disease classification utilizes estimated glomerular filtration rate (eGFR) and urine albumin-to-creatinine ratio (uACR).",
        "domain": "Healthcare / Medical",
    },
    {
        "doc_id": "NONMEM-MED-005",
        "text": "Laparoscopic cholecystectomy is the surgical gold standard for symptomatic cholelithiasis and acute uncomplicated cholecystitis.",
        "domain": "Healthcare / Medical",
    },
    {
        "doc_id": "NONMEM-MED-006",
        "text": "Biologic disease-modifying antirheumatic drugs (bDMARDs) inhibiting TNF-alpha and IL-6 show efficacy in refractory rheumatoid arthritis.",
        "domain": "Healthcare / Medical",
    },
    {
        "doc_id": "NONMEM-MED-007",
        "text": "Percutaneous coronary intervention (PCI) with drug-eluting stents restores coronary blood flow in ST-elevation myocardial infarction.",
        "domain": "Healthcare / Medical",
    },
    {
        "doc_id": "NONMEM-MED-008",
        "text": "Palliative care symptom management applies multimodal analgesic titration and compassionate communication for advanced oncological disease.",
        "domain": "Healthcare / Medical",
    },
    # Domain: Education / Academics (8 non-members)
    {
        "doc_id": "NONMEM-EDU-001",
        "text": "Vygotsky's zone of proximal development provides a theoretical foundation for instructional scaffolding and collaborative peer learning.",
        "domain": "Education / Academics",
    },
    {
        "doc_id": "NONMEM-EDU-002",
        "text": "Gamification elements in educational technology, including mastery badges and challenge loops, boost student intrinsic motivation.",
        "domain": "Education / Academics",
    },
    {
        "doc_id": "NONMEM-EDU-003",
        "text": "Open Educational Resources (OER) reduce higher education textbook costs while promoting adaptable open-licensed pedagogical content.",
        "domain": "Education / Academics",
    },
    {
        "doc_id": "NONMEM-EDU-004",
        "text": "Bloom's revised taxonomy provides a hierarchal framework categorizing educational cognitive tasks from remembering to creative synthesis.",
        "domain": "Education / Academics",
    },
    {
        "doc_id": "NONMEM-EDU-005",
        "text": "Tenure-track faculty promotion reviews weigh peer-reviewed scholarship, external grant funding, teaching evaluations, and institutional service.",
        "domain": "Education / Academics",
    },
    {
        "doc_id": "NONMEM-EDU-006",
        "text": "Montessori educational philosophy centers on student-chosen activities, multi-age classrooms, and self-correcting sensory learning materials.",
        "domain": "Education / Academics",
    },
    {
        "doc_id": "NONMEM-EDU-007",
        "text": "Spaced repetition scheduling algorithms optimize long-term memory retention by progressively expanding review intervals for learned concepts.",
        "domain": "Education / Academics",
    },
    {
        "doc_id": "NONMEM-EDU-008",
        "text": "Doctoral dissertation committees require comprehensive oral candidacy exams and original dissertation defenses prior to degree conferral.",
        "domain": "Education / Academics",
    },
    # Domain: Industrial / Engineering (8 non-members)
    {
        "doc_id": "NONMEM-IND-001",
        "text": "Selective laser sintering (SLS) additive manufacturing produces complex functional thermoplastic components without requiring support structures.",
        "domain": "Industrial / Engineering",
    },
    {
        "doc_id": "NONMEM-IND-002",
        "text": "Lean manufacturing Kanban systems synchronize visual pull signals to eliminate work-in-progress inventory accumulation in assembly cells.",
        "domain": "Industrial / Engineering",
    },
    {
        "doc_id": "NONMEM-IND-003",
        "text": "Centrifugal pump cavitation occurs when local fluid pressure drops below vapor pressure, causing imploding bubbles and impeller erosion.",
        "domain": "Industrial / Engineering",
    },
    {
        "doc_id": "NONMEM-IND-004",
        "text": "ISO 14001 certified environmental management systems structure industrial compliance audits, resource efficiency, and emissions reductions.",
        "domain": "Industrial / Engineering",
    },
    {
        "doc_id": "NONMEM-IND-005",
        "text": "Programmable Automation Controllers (PACs) integrate multidisciplinary logic, motion control, and data acquisition onto open hardware.",
        "domain": "Industrial / Engineering",
    },
    {
        "doc_id": "NONMEM-IND-006",
        "text": "Cyber-physical system architectures in smart grids employ redundant telemetry and hardware authentication to guard against network intrusions.",
        "domain": "Industrial / Engineering",
    },
    {
        "doc_id": "NONMEM-IND-007",
        "text": "Titanium alloy aerospace machining mandates rigid tooling and high-pressure cryogenic cooling to manage extreme tool tip temperatures.",
        "domain": "Industrial / Engineering",
    },
    {
        "doc_id": "NONMEM-IND-008",
        "text": "Robotic laser triangulation seam trackers automatically compensate for joint gaps and thermal distortion in automated welding processes.",
        "domain": "Industrial / Engineering",
    },
    # Domain: General Knowledge / History (8 non-members)
    {
        "doc_id": "NONMEM-HIS-001",
        "text": "The Peloponnesian War between the Athenian empire and the Peloponnesian League reshaped classical Greek hegemony and weakened City-States.",
        "domain": "General Knowledge / History",
    },
    {
        "doc_id": "NONMEM-HIS-002",
        "text": "The Ming Dynasty constructed extensive brick and stone sections of the Great Wall of China to repel recurring nomadic cavalry incursions.",
        "domain": "General Knowledge / History",
    },
    {
        "doc_id": "NONMEM-HIS-003",
        "text": "The storming of the Bastille on July 14, 1789, ignited the French Revolution, leading to the Declaration of the Rights of Man.",
        "domain": "General Knowledge / History",
    },
    {
        "doc_id": "NONMEM-HIS-004",
        "text": "Emperor Justinian's Corpus Juris Civilis consolidated centuries of Roman legal jurisprudence, forming the foundation for continental civil law.",
        "domain": "General Knowledge / History",
    },
    {
        "doc_id": "NONMEM-HIS-005",
        "text": "The 1858 transatlantic telegraph cable linked North America and Great Britain, reducing intercontinental message transit from weeks to minutes.",
        "domain": "General Knowledge / History",
    },
    {
        "doc_id": "NONMEM-HIS-006",
        "text": "The Meiji Restoration of 1868 dismantled feudal Tokugawa shogunate rule, propelling Japan into rapid industrialization and modernization.",
        "domain": "General Knowledge / History",
    },
    {
        "doc_id": "NONMEM-HIS-007",
        "text": "The construction of the Panama Canal connected the Atlantic and Pacific oceans, overcoming malaria epidemics and mountainous excavation.",
        "domain": "General Knowledge / History",
    },
    {
        "doc_id": "NONMEM-HIS-008",
        "text": "Johannes Gutenberg's development of movable type printing around 1440 transformed European information dissemination and catalyzed the Reformation.",
        "domain": "General Knowledge / History",
    },
]


def run_phase2_evaluation() -> Dict[str, Any]:
    """Execute Membership Inference Attack benchmark on the undefended Federated RAG system.

    Returns:
        Dict[str, Any]: Summary dictionary containing privacy and utility metrics.
    """
    print("=" * 80)
    print(" PHASE 2: MEMBERSHIP INFERENCE ATTACK (MIA) BASELINE EVALUATION")
    print("=" * 80)

    # 1. Load dataset splits
    print("\n[1/5] Loading client splits from JSON...")
    splits = load_client_splits(DEFAULT_SPLITS_PATH)
    all_members: List[Dict[str, Any]] = []
    for client_id, docs in splits.items():
        all_members.extend(docs)

    non_members = SYNTHETIC_NON_MEMBERS

    print(f" -> Indexed Members (y=1): {len(all_members)} documents across {len(splits)} clients")
    print(f" -> Non-Members     (y=0): {len(non_members)} documents across {len(splits)} domains")
    print(f" -> Total MIA Evaluation Targets: {len(all_members) + len(non_members)}")

    # 2. Initialize Clients and Coordinator
    print("\n[2/5] Initializing Undefended Federated RAG System...")
    domain_names = {
        1: "Healthcare / Medical",
        2: "Education / Academics",
        3: "Industrial / Engineering",
        4: "General Knowledge / History",
    }

    clients: List[ClientNode] = []
    shared_model = None

    for cid in sorted(splits.keys()):
        d_name = domain_names.get(cid, splits[cid][0]["domain"])
        client = ClientNode(client_id=cid, domain_name=d_name, model=shared_model)
        if shared_model is None:
            shared_model = client.model
        client.populate_index(splits[cid])
        clients.append(client)
        print(f" -> Client {cid} [{d_name}]: {client.index.ntotal} documents indexed.")

    coordinator = FederatedCoordinator(clients=clients, model=shared_model)
    print(" -> FederatedCoordinator initialized with 4 client nodes.")

    # 3. Instantiate MIA Attacker Suite
    print("\n[3/5] Executing Membership Inference Attacks against FederatedCoordinator...")
    attacker = MembershipBasedAttack()

    y_true: List[int] = []
    y_scores: List[float] = []

    member_recalls: List[float] = []
    member_ndcgs: List[float] = []
    member_direct_presence_count = 0
    non_member_false_presence_count = 0

    # Evaluate Members (y=1)
    print(" -> Evaluating Members (y=1)...")
    for doc in all_members:
        res = attacker.evaluate_target(
            coordinator=coordinator,
            doc=doc,
            is_member=True,
            top_k_per_client=5,
            global_top_k=5,
            n_queries=3,
        )
        y_true.append(1)
        y_scores.append(res["attack_score"])

        if res["target_present"]:
            member_direct_presence_count += 1

        # Compute utility for member query
        utility = compute_retrieval_utility(
            ground_truth_doc_ids=[doc["doc_id"]],
            retrieved_docs=res["retrieved_docs"],
            k=5,
        )
        member_recalls.append(utility["recall@5"])
        member_ndcgs.append(utility["ndcg@5"])

    # Evaluate Non-Members (y=0)
    print(" -> Evaluating Non-Members (y=0)...")
    for doc in non_members:
        res = attacker.evaluate_target(
            coordinator=coordinator,
            doc=doc,
            is_member=False,
            top_k_per_client=5,
            global_top_k=5,
            n_queries=3,
        )
        y_true.append(0)
        y_scores.append(res["attack_score"])

        if res["target_present"]:
            non_member_false_presence_count += 1

    # 4. Compute Metrics
    print("\n[4/5] Computing Quantitative Metrics...")
    privacy_metrics = compute_privacy_metrics(y_true=y_true, y_scores=y_scores)
    mean_recall = float(np.mean(member_recalls))
    mean_ndcg = float(np.mean(member_ndcgs))

    member_scores = [s for s, y in zip(y_scores, y_true) if y == 1]
    non_member_scores = [s for s, y in zip(y_scores, y_true) if y == 0]

    # 5. Print Detailed Diagnostic & Vulnerability Report
    print("\n[5/5] Diagnostic Summary & Vulnerability Analysis")
    print("=" * 80)
    print(" BASELINE PRIVACY VULNERABILITY (UNDEFENDED FEDERATED RAG)")
    print("=" * 80)

    print("\n[+] ATTACKER PERFORMANCE METRICS (PRIVACY LEAKAGE):")
    print(f"    - MIA ROC-AUC:              {privacy_metrics['roc_auc']:.4f} (Ideal Defense = 0.5000)")
    print(f"    - TPR @ 1% FPR:             {privacy_metrics['tpr_at_1_percent_fpr'] * 100:.2f}% (Attacker True Positive Rate at strict 1% false alarms)")
    print(f"    - Member Direct Extraction: {member_direct_presence_count}/{len(all_members)} ({member_direct_presence_count / len(all_members) * 100:.1f}%)")
    print(f"    - Non-Member False Alarm:   {non_member_false_presence_count}/{len(non_members)} ({non_member_false_presence_count / len(non_members) * 100:.1f}%)")
    print(f"    - Mean Attacker Score (y=1): {np.mean(member_scores):.4f} (std: {np.std(member_scores):.4f})")
    print(f"    - Mean Attacker Score (y=0): {np.mean(non_member_scores):.4f} (std: {np.std(non_member_scores):.4f})")

    print("\n[+] BASELINE RETRIEVAL UTILITY (ON MEMBERS):")
    print(f"    - Baseline Recall@5:        {mean_recall:.4f} ({mean_recall * 100:.1f}%)")
    print(f"    - Baseline nDCG@5:          {mean_ndcg:.4f}")

    print("\n" + "-" * 80)
    print(" SECURITY RISK SUMMARY:")
    if privacy_metrics["roc_auc"] > 0.85:
        print(" [CRITICAL PRIVACY VULNERABILITY CONFIRMED]")
        print("  The undefended Federated RAG system leaks acute membership information.")
        print("  - Verbatim and partial probing queries allow an adversary to isolate internal private documents.")
        print("  - Raw cosine similarity scores and verbatim document snippets are returned without noise,")
        print("    thresholding, or differential privacy mechanisms.")
        print("  - High ROC-AUC and near-perfect TPR @ 1% FPR demonstrate that an attacker can reliably")
        print("    determine whether private records exist within client data partitions.")
    else:
        print(" [STATUS: Low to Moderate MIA Risk Observed]")

    print("=" * 80)

    return {
        "roc_auc": privacy_metrics["roc_auc"],
        "tpr_at_1_percent_fpr": privacy_metrics["tpr_at_1_percent_fpr"],
        "recall@5": mean_recall,
        "ndcg@5": mean_ndcg,
    }


if __name__ == "__main__":
    metrics = run_phase2_evaluation()
