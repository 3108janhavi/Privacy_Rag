# Federated RAG (FedRAG) Research Framework

A modular, extensible research pipeline for Federated Retrieval-Augmented Generation (FedRAG), Membership Inference Attack (MIA) evaluation, multi-signal privacy defenses, and query-sensitive adaptive routing.

## Project Structure
```plaintext
fedrag_project/
├── data/
│   ├── client_splits.json        # Generated non-IID domain document partitions
│   └── prepare_data.py           # Synthetic dataset generator & JSON serialization helpers
├── src/
│   ├── __init__.py               # Package exports
│   ├── client.py                 # ClientNode with local FAISS IndexFlatIP (Cosine Similarity)
│   ├── coordinator.py            # FederatedCoordinator with selective broadcast, defenses & routing
│   ├── attacks/
│   │   ├── __init__.py           # Attack exports
│   │   └── mia_base.py           # MembershipBasedAttack suite & probing query generator
│   ├── defense/
│   │   ├── __init__.py           # Defense exports
│   │   ├── mirabel.py            # Baseline MIRABEL detection & top-candidate suppression
│   │   └── acd_fedrag.py         # Proposed ACD-FedRAG defense (Local Gap + Cross Dispersion + Demotion)
│   ├── router/
│   │   ├── __init__.py           # Router exports
│   │   └── adaptive_router.py    # QueryAnalyzerAndRouter for dynamic client pruning
│   └── utils/
│       ├── __init__.py           # Utility exports
│       └── metrics.py            # Privacy (ROC-AUC, TPR@1% FPR) and Utility (Recall@k, nDCG@k)
├── requirements.txt              # Project dependencies
├── test_pipeline.py              # Phase 1: Automated end-to-end pipeline verification test
├── run_phase2_attack.py          # Phase 2: Membership Inference Attack (MIA) baseline evaluation
├── run_phase3_mirabel.py         # Phase 3: MIRABEL defense benchmark & comparative evaluation
├── run_phase4_acd.py             # Phase 4: ACD-FedRAG defense & ablation study benchmark
└── run_phase5_adaptive.py        # Phase 5: Query-Sensitive Adaptive Routing efficiency benchmark
```

## Setup & Dependencies

Install dependencies:
```bash
pip install -r requirements.txt
```

## Execution Commands

### Phase 1: Base Federated Retrieval Pipeline
```bash
python data/prepare_data.py
python test_pipeline.py
```

### Phase 2: Membership Inference Attack (MIA) Baseline Evaluation
```bash
python run_phase2_attack.py
```

### Phase 3: MIRABEL Baseline Defense Reproduction
```bash
python run_phase3_mirabel.py
```

### Phase 4: ACD-FedRAG Defense & Ablation Framework
```bash
python run_phase4_acd.py
```

### Phase 5: Query-Sensitive Adaptive Routing Benchmark
```bash
python run_phase5_adaptive.py
```
