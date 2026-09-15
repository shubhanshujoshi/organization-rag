from __future__ import annotations

import time

from organization_rag.chroma_store import ChromaStore
from organization_rag.config import TOP_K
from organization_rag.postgres_store import PostgresVectorStore
from organization_rag.retrieve import retrieve


# ============================================================
# Evaluation questions
# ============================================================

EVALUATION_CASES = [
    {
        "question": "How many placement attempts are allowed?",
        "department": "placement",
        "relevant_pages": {11, 12},
    },
    {
        "question": (
            "What happens if a student wins a top-3 position "
            "in a case competition?"
        ),
        "department": "placement",
        "relevant_pages": {11},
    },
    {
        "question": (
            "What happens when a student reaches the national finals?"
        ),
        "department": "placement",
        "relevant_pages": {11},
    },
    {
        "question": "What is the placement process?",
        "department": "placement",
        "relevant_pages": {5, 10, 12},
    },
    {
        "question": "What are the rules for placement registration?",
        "department": "placement",
        "relevant_pages": {5, 11, 12},
    },
]


# ============================================================
# Utility functions
# ============================================================

def normalize_page(value):
    """Convert page metadata to an integer when possible."""

    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def calculate_hit(retrieved_pages, relevant_pages):
    """Return 1 if at least one relevant page was retrieved."""

    return int(bool(set(retrieved_pages) & set(relevant_pages)))


def calculate_recall(retrieved_pages, relevant_pages):
    """
    Recall@K = relevant retrieved pages / total relevant pages.
    """

    relevant_pages = set(relevant_pages)

    if not relevant_pages:
        return 0.0

    retrieved_pages = set(retrieved_pages)

    return len(
        retrieved_pages & relevant_pages
    ) / len(relevant_pages)


def print_result(
    question,
    retrieved_pages,
    expected_pages,
    latency_ms,
):
    hit = calculate_hit(
        retrieved_pages,
        expected_pages,
    )

    recall = calculate_recall(
        retrieved_pages,
        expected_pages,
    )

    print(
        f"Question: {question}"
    )

    print(
        f"  Retrieved pages : {retrieved_pages}"
    )

    print(
        f"  Expected pages  : {sorted(expected_pages)}"
    )

    print(
        f"  Hit@{TOP_K}       : {hit}"
    )

    print(
        f"  Recall@{TOP_K}    : {recall:.2f}"
    )

    print(
        f"  Latency          : {latency_ms:.2f} ms"
    )

    print()


# ============================================================
# QDRANT
# ============================================================

def run_qdrant():
    print("=" * 70)
    print("QDRANT")
    print("=" * 70)

    all_hits = []
    all_recalls = []
    all_latencies = []

    for case in EVALUATION_CASES:

        question = case["question"]
        department = case["department"]
        expected_pages = case["relevant_pages"]

        start = time.perf_counter()

        results = retrieve(
            query=question,
            department=department,
            top_k=TOP_K,
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        retrieved_pages = []

        for result in results:

            payload = result.payload or {}

            page = normalize_page(
                payload.get("page")
            )

            if page is not None:
                retrieved_pages.append(page)

        hit = calculate_hit(
            retrieved_pages,
            expected_pages,
        )

        recall = calculate_recall(
            retrieved_pages,
            expected_pages,
        )

        all_hits.append(hit)
        all_recalls.append(recall)
        all_latencies.append(latency_ms)

        print_result(
            question=question,
            retrieved_pages=retrieved_pages,
            expected_pages=expected_pages,
            latency_ms=latency_ms,
        )

    print(
        f"Average Hit@{TOP_K}: "
        f"{sum(all_hits) / len(all_hits):.2f}"
    )

    print(
        f"Average Recall@{TOP_K}: "
        f"{sum(all_recalls) / len(all_recalls):.2f}"
    )

    print(
        f"Average latency: "
        f"{sum(all_latencies) / len(all_latencies):.2f} ms"
    )

    print()


# ============================================================
# CHROMA
# ============================================================

def run_chroma():
    print("=" * 70)
    print("CHROMA")
    print("=" * 70)

    store = ChromaStore()

    all_hits = []
    all_recalls = []
    all_latencies = []

    for case in EVALUATION_CASES:

        question = case["question"]
        department = case["department"]
        expected_pages = case["relevant_pages"]

        start = time.perf_counter()

        results = store.search(
            query=question,
            department=department,
            top_k=TOP_K,
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        retrieved_pages = []

        metadatas = results.get(
            "metadatas",
            [[]],
        )

        if metadatas:

            for metadata in metadatas[0]:

                metadata = metadata or {}

                page = normalize_page(
                    metadata.get("page")
                )

                if page is not None:
                    retrieved_pages.append(page)

        hit = calculate_hit(
            retrieved_pages,
            expected_pages,
        )

        recall = calculate_recall(
            retrieved_pages,
            expected_pages,
        )

        all_hits.append(hit)
        all_recalls.append(recall)
        all_latencies.append(latency_ms)

        print_result(
            question=question,
            retrieved_pages=retrieved_pages,
            expected_pages=expected_pages,
            latency_ms=latency_ms,
        )

    print(
        f"Average Hit@{TOP_K}: "
        f"{sum(all_hits) / len(all_hits):.2f}"
    )

    print(
        f"Average Recall@{TOP_K}: "
        f"{sum(all_recalls) / len(all_recalls):.2f}"
    )

    print(
        f"Average latency: "
        f"{sum(all_latencies) / len(all_latencies):.2f} ms"
    )

    print()


# ============================================================
# POSTGRESQL / PGVECTOR
# ============================================================

def run_postgres():
    print("=" * 70)
    print("POSTGRESQL / PGVECTOR")
    print("=" * 70)

    store = PostgresVectorStore()

    all_hits = []
    all_recalls = []
    all_latencies = []

    try:
        for case in EVALUATION_CASES:
            question = case["question"]
            department = case["department"]
            expected_pages = case["relevant_pages"]

            start = time.perf_counter()

            results = store.search(
                query=question,
                department=department,
                top_k=TOP_K,
            )

            latency_ms = (
                time.perf_counter() - start
            ) * 1000

            retrieved_pages = []

            for row in results:
                # PostgresVectorStore.search() currently returns
                # dictionaries, but support tuple rows as well.
                if isinstance(row, dict):
                    metadata = row.get("metadata", {})
                else:
                    # Expected tuple structure from the SQL query:
                    # (id, content, metadata, similarity)
                    metadata = row[2] if len(row) > 2 else {}

                if metadata is None:
                    metadata = {}

                page = normalize_page(
                    metadata.get("page")
                )

                if page is not None:
                    retrieved_pages.append(page)

            hit = calculate_hit(
                retrieved_pages,
                expected_pages,
            )

            recall = calculate_recall(
                retrieved_pages,
                expected_pages,
            )

            all_hits.append(hit)
            all_recalls.append(recall)
            all_latencies.append(latency_ms)

            print_result(
                question=question,
                retrieved_pages=retrieved_pages,
                expected_pages=expected_pages,
                latency_ms=latency_ms,
            )

        print(
            f"Average Hit@{TOP_K}: "
            f"{sum(all_hits) / len(all_hits):.2f}"
        )

        print(
            f"Average Recall@{TOP_K}: "
            f"{sum(all_recalls) / len(all_recalls):.2f}"
        )

        print(
            f"Average latency: "
            f"{sum(all_latencies) / len(all_latencies):.2f} ms"
        )

        print()

    finally:
        store.close()

# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("THREE-WAY VECTOR STORE COMPARISON")
    print(f"TOP_K = {TOP_K}")
    print()

    run_qdrant()
    run_chroma()
    run_postgres()

    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()