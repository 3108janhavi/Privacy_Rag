# Privacy-Preserving Federated Retrieval-Augmented Generation (FedRAG)

An end-to-end research framework implementing a modular Federated RAG engine, Membership Inference Attack (MIA) evaluation suite, baseline threshold defenses (MIRABEL), our proposed Adaptive Cross-Client Dispersion Defense (ACD-FedRAG), and Query-Sensitive Adaptive Routing.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |          User Search Query            |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |    QueryAnalyzerAndRouter (Phase 5)   |
                      |  - Intent Keyword Classification      |
                      |  - Syntactic Probing Detection        |
                      |  - Selective vs Broadcast Routing     |
                      +---------------------------------------+
                                          |
                        +-----------------+-----------------+
                        | (Single-Domain)                   | (Multi-Domain / Probe)
                        v                                   v
             [Selective Target Client]             [Broadcast to All Clients]
                        |                                   |
                        +-----------------+-----------------+
                                          |
                                          v
                      +---------------------------------------+
                      |        Federated Clients (1 - 4)      |
                      |  - Local Non-IID Domain FAISS Index   |
                      |  - Cosine Similarity (IndexFlatIP)    |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |        ACD-FedRAG Defense (Phase 4)   |
                      |  - Local Score Gap Check (Delta_local)|
                      |  - Cross-Client Dispersion (Std Dev)  |
                      |  - Targeted Score Demotion (No Purge) |
                      +---------------------------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |          Global Rank & Merge          |
                      |  Top-k Document Context Generation    |
                      +---------------------------------------+
```

---

## Directory Structure

```plaintext
Privacy_Rag/
├── fedrag_project/
│   ├── data/
│   │   ├── client_splits.json        # 32 Non-IID domain-partitioned documents across 4 clients
│   │   └── prepare_data.py           # Synthetic dataset generator & serialization utilities
│   ├── src/
│   │   ├── __init__.py
│   │   ├── client.py                 # ClientNode with local FAISS IndexFlatIP (Cosine Similarity)
│   │   ├── coordinator.py            # FederatedCoordinator supporting pluggable defenses and routing
│   │   ├── attacks/
│   │   │   ├── __init__.py
│   │   │   └── mia_base.py           # MembershipBasedAttack suite & probing query generator
│   │   ├── defense/
│   │   │   ├── __init__.py
│   │   │   ├── mirabel.py            # Baseline MIRABEL detection & top-candidate suppression
│   │   │   └── acd_fedrag.py         # Proposed ACD-FedRAG defense (Local Gap + Cross Dispersion)
│   │   ├── router/
│   │   │   ├── __init__.py
│   │   │   └── adaptive_router.py    # QueryAnalyzerAndRouter for dynamic client selection
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── metrics.py            # Privacy (ROC-AUC, TPR@1% FPR) and Utility (Recall@k, nDCG@k)
│   ├── test_pipeline.py              # Phase 1: Base Federated RAG automated test suite
│   ├── run_phase2_attack.py          # Phase 2: MIA privacy vulnerability baseline benchmark
│   ├── run_phase3_mirabel.py         # Phase 3: MIRABEL defense evaluation across threshold sweeps
│   ├── run_phase4_acd.py             # Phase 4: ACD-FedRAG defense and multi-signal ablation study
│   ├── run_phase5_adaptive.py        # Phase 5: Query-Sensitive Adaptive Routing efficiency benchmark
│   └── README.md
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Research Phases & Execution

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Phase 1: Base Federated RAG Pipeline
Builds non-IID document partitions across 4 client nodes (Healthcare, Education, Industrial, History) and validates FAISS indexing and coordinator ranking:
```bash
python fedrag_project/data/prepare_data.py
python fedrag_project/test_pipeline.py
```

### 3. Phase 2: Membership Inference Attack (MIA) Benchmark
Evaluates privacy leakage of the undefended retrieval engine using verbatim probing queries across 64 targets (32 members vs 32 non-members):
```bash
python fedrag_project/run_phase2_attack.py
```
- **Undefended MIA ROC-AUC**: `1.0000`
- **Undefended TPR @ 1% FPR**: `100.00%`

### 4. Phase 3: MIRABEL Baseline Defense Reproduction
Evaluates similarity-threshold suppression ($\tau \in [0.80, 0.85, 0.90]$):
```bash
python fedrag_project/run_phase3_mirabel.py
```
- **Finding**: Purging top candidates results in catastrophic utility failure (`Recall@5 = 0.0000`), while sub-threshold partial queries evade detection.

### 5. Phase 4: ACD-FedRAG Defense & Ablation Framework
Evaluates multi-signal risk detection ($\Delta_{\text{local}}$ and $\Delta_{\text{cross}}$) combined with targeted score demotion:
```bash
python fedrag_project/run_phase4_acd.py
```
- **Result**: Restores full generation utility (`Recall@5 = 1.0000`, `nDCG@5 = 1.0000`) without purging candidate documents.

### 6. Phase 5: Query-Sensitive Adaptive Routing Benchmark
Evaluates selective client broadcast based on query intent and complexity:
```bash
python fedrag_project/run_phase5_adaptive.py
```
- **Communication Reduction**: 50.00% reduction in client query volume (2.00 vs 4.00 clients on average).
- **Retrieval Fidelity**: Retained Recall@5 of 91.1% and nDCG@5 of 93.9% with safety fallback on verbatim probing attempts.
