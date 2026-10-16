"""End-to-End Automated Test Pipeline for Base Federated RAG (Phase 1).

Validates:
1. Dataset preparation and JSON split persistence/loading.
2. Initialization of 4 domain-specific ClientNodes and FAISS index population.
3. FederatedCoordinator query broadcast and global top-k ranking.
4. Numerical integrity: similarity scores strictly bounded in [-1.0, 1.0].
5. Monotonic score descending order.
6. Domain routing integrity: queries retrieve the corresponding domain client at top ranks.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure fedrag_project directory is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.prepare_data import (
    DEFAULT_SPLITS_PATH,
    get_synthetic_client_data,
    load_client_splits,
    save_client_splits,
)
from src.client import ClientNode
from src.coordinator import FederatedCoordinator


CLIENT_DOMAIN_METADATA = {
    1: "Healthcare / Medical",
    2: "Education / Academics",
    3: "Industrial / Engineering",
    4: "General Knowledge / History",
}

TEST_QUERIES = [
    {
        "query": "What are recommended clinical treatments and management for type 2 diabetes?",
        "expected_client_id": 1,
        "expected_domain": "Healthcare / Medical",
    },
    {
        "query": "How does cognitive load theory impact instructional design and formative assessment?",
        "expected_client_id": 2,
        "expected_domain": "Education / Academics",
    },
    {
        "query": "Predictive maintenance vibration analysis for industrial CNC machinery and bearings",
        "expected_client_id": 3,
        "expected_domain": "Industrial / Engineering",
    },
    {
        "query": "What factors led to the fall of the Western Roman Empire and rise of the Renaissance?",
        "expected_client_id": 4,
        "expected_domain": "General Knowledge / History",
    },
]


def run_pipeline_tests() -> bool:
    """Execute all pipeline validations and print diagnostic report.

    Returns:
        bool: True if all test suites pass, False otherwise.
    """
    all_passed = True
    print("=" * 80)
    print(" FEDERATED RAG (PHASE 1) - AUTOMATED PIPELINE VERIFICATION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Step 1: Dataset Generation and Serialization Check
    # -------------------------------------------------------------------------
    print("\n[STEP 1] Generating and verifying synthetic client splits...")
    try:
        raw_splits = get_synthetic_client_data()
        assert len(raw_splits) == 4, f"Expected 4 client splits, got {len(raw_splits)}"

        saved_path = save_client_splits(raw_splits, DEFAULT_SPLITS_PATH)
        assert saved_path.exists(), f"Splits file was not created at: {saved_path}"

        loaded_splits = load_client_splits(saved_path)
        assert len(loaded_splits) == 4, f"Loaded client count mismatch: {len(loaded_splits)}"

        total_docs = sum(len(docs) for docs in loaded_splits.values())
        print(f" -> Successfully saved and loaded {total_docs} documents across 4 clients.")
        print(f" -> File location: {saved_path}")
        print(" [PASS] Step 1: Dataset Preparation and JSON Persistence")
    except Exception as e:
        print(f" [FAIL] Step 1: Dataset Preparation failed with error: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Step 2: ClientNode Initialization & Index Population
    # -------------------------------------------------------------------------
    print("\n[STEP 2] Initializing 4 ClientNodes and building FAISS IndexFlatIP...")
    clients: List[ClientNode] = []
    shared_model = None

    try:
        for client_id, domain_name in CLIENT_DOMAIN_METADATA.items():
            client = ClientNode(
                client_id=client_id,
                domain_name=domain_name,
                model=shared_model,
            )
            # Share embedding model instance across clients to optimize memory & startup time
            if shared_model is None:
                shared_model = client.model

            client_docs = loaded_splits[client_id]
            client.populate_index(client_docs)
            clients.append(client)

            assert client.index is not None, f"FAISS index is None for Client {client_id}"
            assert client.index.ntotal == len(client_docs), (
                f"Index vector count ({client.index.ntotal}) != document count ({len(client_docs)})"
            )
            print(f" -> Client {client_id} [{domain_name}]: {client.index.ntotal} vectors indexed.")

        print(" [PASS] Step 2: ClientNodes Initialized and Indexed")
    except Exception as e:
        print(f" [FAIL] Step 2: ClientNode initialization failed: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Step 3: FederatedCoordinator Initialization
    # -------------------------------------------------------------------------
    print("\n[STEP 3] Initializing FederatedCoordinator...")
    try:
        coordinator = FederatedCoordinator(clients=clients, model=shared_model)
        assert len(coordinator.clients) == 4, "Coordinator did not register all 4 clients"
        print(" [PASS] Step 3: FederatedCoordinator Initialized")
    except Exception as e:
        print(f" [FAIL] Step 3: Coordinator setup failed: {e}")
        all_passed = False

    # -------------------------------------------------------------------------
    # Step 4: Domain-Specific Query Execution & Integrity Tests
    # -------------------------------------------------------------------------
    print("\n[STEP 4] Executing federated domain routing and validation tests...")
    top_k_per_client = 3
    global_top_k = 5

    for idx, test_case in enumerate(TEST_QUERIES, start=1):
        query = test_case["query"]
        expected_cid = test_case["expected_client_id"]
        expected_dom = test_case["expected_domain"]

        print(f"\n--- Test Query {idx}/{len(TEST_QUERIES)} ---")
        print(f" Query: \"{query}\"")
        print(f" Expected Domain: Client {expected_cid} ({expected_dom})")

        try:
            results = coordinator.search(
                query=query,
                top_k_per_client=top_k_per_client,
                global_top_k=global_top_k,
            )

            # Check 1: Non-empty results
            assert len(results) > 0, f"Query '{query}' returned empty results!"
            assert len(results) <= global_top_k, (
                f"Result count ({len(results)}) exceeded global_top_k ({global_top_k})"
            )

            # Check 2: Schema validation, score bounds [-1.0, 1.0], and monotonic descending order
            prev_score = float("inf")
            for doc_rank, doc in enumerate(results, start=1):
                assert "doc_id" in doc, f"Missing doc_id at rank {doc_rank}"
                assert "text" in doc, f"Missing text at rank {doc_rank}"
                assert "score" in doc, f"Missing score at rank {doc_rank}"
                assert "client_id" in doc, f"Missing client_id at rank {doc_rank}"
                assert "domain" in doc, f"Missing domain at rank {doc_rank}"

                score = doc["score"]
                assert isinstance(score, float), f"Score is not float: {type(score)}"
                assert -1.0 <= score <= 1.0, (
                    f"Score {score} out of strict Cosine Similarity bounds [-1.0, 1.0]"
                )
                assert score <= prev_score + 1e-6, (
                    f"Global descending ranking violated: {score} > {prev_score}"
                )
                prev_score = score

            # Check 3: Domain routing integrity (Top-1 document belongs to targeted client)
            top_result = results[0]
            routing_match = top_result["client_id"] == expected_cid

            print(" Retrieved Top Candidates:")
            for rank, r in enumerate(results, start=1):
                marker = "==> [TOP-1]" if rank == 1 else "   "
                text_snippet = (r["text"][:75] + "...") if len(r["text"]) > 75 else r["text"]
                print(
                    f" {marker} Rank {rank} | Score: {r['score']:.4f} | "
                    f"Client {r['client_id']} ({r['domain']}) | "
                    f"Doc: {r['doc_id']} | Text: \"{text_snippet}\""
                )

            if routing_match:
                print(f" [PASS] Domain Routing Integrity: Matched Client {expected_cid}")
            else:
                print(
                    f" [FAIL] Domain Routing Integrity: Expected Client {expected_cid}, "
                    f"got Client {top_result['client_id']}"
                )
                all_passed = False

            print(f" [PASS] Numerical Bounding & Sorting: All scores in [-1.0, 1.0], sorted descending.")

        except Exception as e:
            print(f" [FAIL] Test query execution failed: {e}")
            all_passed = False

    # -------------------------------------------------------------------------
    # Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    if all_passed:
        print(" OVERALL STATUS: ALL PHASE 1 PIPELINE TESTS PASSED [100%]")
        print("=" * 80)
    else:
        print(" OVERALL STATUS: SOME TESTS FAILED")
        print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = run_pipeline_tests()
    sys.exit(0 if success else 1)
