# 🛡️ Federated RAG Interactive Presentation Dashboard

A Streamlit web dashboard allowing evaluators to explore, test, and benchmark all 5 research phases of the Privacy-Preserving Federated RAG project.

---

## 🚀 Quick Launch Instructions

### 1. Install Dependencies
Ensure all dependencies including `streamlit` and `plotly` are installed:
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard
From the `fedrag_project/` directory:
```bash
streamlit run app.py
```
Or from the project root:
```bash
python -m streamlit run fedrag_project/app.py
```

The web dashboard will automatically open in your default browser at:
`http://localhost:8501`

---

## 🧭 Dashboard Page Guide

| Sidebar Navigation Phase | Description & Interactive Features |
| :--- | :--- |
| **🏛️ Overview & System Architecture** | Mermaid architecture diagram of the 4 client domains (Healthcare, Education, Industrial, History), research objectives, and partitioned document inspector. |
| **⚡ Phase 1: Base Pipeline** | Interactive search box with domain query presets, cosine similarity scores, source client badges, and routing integrity checks. |
| **🕵️ Phase 2: MIA Baseline** | One-click live attack execution, ROC-AUC score, TPR@1% FPR, and interactive Plotly score distribution histograms ($y=1$ vs $y=0$). |
| **🛡️ Phase 3: MIRABEL Reproduction** | Dynamic threshold slider ($\tau \in [0.80, 0.90]$), live document purging inspection, and explanation of candidate deletion failure. |
| **🔬 Phase 4: Proposed ACD-FedRAG** | Interactive defense toggles (Undefended, MIRABEL, ACD-FedRAG, Local-Only, Cross-Only) demonstrating targeted score demotion restoring Recall@5 to 1.0000. |
| **🧭 Phase 5: Adaptive Routing** | Query complexity and probing detection simulator showing the 50.0% network payload reduction and selective broadcast logic. |
| **📊 Consolidated Benchmark** | Unified comparison table and Plotly grouped bar chart plotting Privacy Vulnerability (ROC-AUC) vs Retrieval Utility (Recall@5). |
