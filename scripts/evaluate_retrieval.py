import time

from organization_rag.chroma_store import ChromaStore
from organization_rag.config import TOP_K
from organization_rag.retrieve import retrieve


# Ground-truth pages based on the Placement Handbook.
# Add more questions/pages as the document set grows.
EVALUATION_CASES = [
    {
        "question": "How many placement attempts are allowed?",
        "relevant_pages": {11, 12},
    },
    {
        "question": "What happens if a student wins a top-3 position in a case competition?",
        "relevant_pages": {11},
    },
    {
        "question": "What happens when a student reaches the national finals?",
        "relevant_pages": {11},
    },
    {
        "question": "What is the placement process?",
        "relevant_pages": {5, 10, 12},
    },
    {
        "question": "What are the rules for placement registration?",
        "relevant_pages": {5, 11, 12},
    },
]


def reciprocal_rank(retrieved_pages, relevant_pages):
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in relevant_pages:
            return 1 / rank
    return 0.0


def evaluate(retrieved_pages, relevant_pages):
    retrieved_set = set(retrieved_pages)

    hits = retrieved_set.intersection(relevant_pages)

    hit_rate = 1.0 if hits else 0.0
    recall = len(hits) / len(relevant_pages)
    mrr = reciprocal_rank(retrieved_pages, relevant_pages)

    return hit_rate, recall, mrr


def qdrant_results():
    output = []

    for case in EVALUATION_CASES:
        start = time.perf_counter()

        points = retrieve(
            query=case["question"],
            department="placement",
            top_k=TOP_K,
        )

        latency = (time.perf_counter() - start) * 1000

        pages = [
            point.payload.get("page")
            for point in points
            if point.payload
        ]

        pages = [int(page) for page in pages]

        hit, recall, mrr = evaluate(
            pages,
            case["relevant_pages"],
        )

        output.append(
            {
                "question": case["question"],
                "pages": pages,
                "latency": latency,
                "hit": hit,
                "recall": recall,
                "mrr": mrr,
            }
        )

    return output


def chroma_results():
    store = ChromaStore()
    output = []

    for case in EVALUATION_CASES:
        start = time.perf_counter()

        result = store.search(
            query=case["question"],
            department="placement",
            top_k=TOP_K,
        )

        latency = (time.perf_counter() - start) * 1000

        metadata = result.get("metadatas", [[]])[0]

        pages = [
            int(item["page"])
            for item in metadata
            if item.get("page") is not None
        ]

        hit, recall, mrr = evaluate(
            pages,
            case["relevant_pages"],
        )

        output.append(
            {
                "question": case["question"],
                "pages": pages,
                "latency": latency,
                "hit": hit,
                "recall": recall,
                "mrr": mrr,
            }
        )

    return output


def print_summary(name, results):
    avg_latency = sum(r["latency"] for r in results) / len(results)
    avg_hit = sum(r["hit"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_mrr = sum(r["mrr"] for r in results) / len(results)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    for r in results:
        print(f"\n{r['question']}")
        print(f"Retrieved pages : {r['pages']}")
        print(f"Latency         : {r['latency']:.2f} ms")
        print(f"Hit@{TOP_K}          : {r['hit']:.2f}")
        print(f"Recall@{TOP_K}       : {r['recall']:.2f}")
        print(f"MRR@{TOP_K}          : {r['mrr']:.2f}")

    print("\nAVERAGES")
    print(f"Latency  : {avg_latency:.2f} ms")
    print(f"Hit@{TOP_K}   : {avg_hit:.2%}")
    print(f"Recall@{TOP_K}: {avg_recall:.2%}")
    print(f"MRR@{TOP_K}   : {avg_mrr:.2%}")


def main():
    print("Retrieval Quality Evaluation")

    qdrant = qdrant_results()
    chroma = chroma_results()

    print_summary("QDRANT", qdrant)
    print_summary("CHROMA", chroma)


if __name__ == "__main__":
    main()
