import time

from organization_rag.chroma_store import ChromaStore
from organization_rag.config import TOP_K
from organization_rag.retrieve import retrieve


QUERIES = [
    "How many placement attempts are allowed?",
    "What happens if a student wins a top-3 position in a case competition?",
    "What happens when a student reaches the national finals?",
    "What is the placement process?",
    "What are the rules for placement registration?",
]


def benchmark_qdrant():
    results = []

    for query in QUERIES:
        start = time.perf_counter()

        points = retrieve(
            query=query,
            department="placement",
            top_k=TOP_K,
        )

        elapsed = (time.perf_counter() - start) * 1000

        pages = [
            point.payload.get("page")
            for point in points
            if point.payload
        ]

        results.append({
            "query": query,
            "latency_ms": elapsed,
            "results": len(points),
            "pages": pages,
        })

    return results


def benchmark_chroma():
    store = ChromaStore()
    results = []

    for query in QUERIES:
        start = time.perf_counter()

        result = store.search(
            query=query,
            department="placement",
            top_k=TOP_K,
        )

        elapsed = (time.perf_counter() - start) * 1000

        metadatas = result.get("metadatas", [[]])[0]

        pages = [
            metadata.get("page")
            for metadata in metadatas
        ]

        results.append({
            "query": query,
            "latency_ms": elapsed,
            "results": len(metadatas),
            "pages": pages,
        })

    return results


def print_results(name, results):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    total_latency = 0

    for item in results:
        total_latency += item["latency_ms"]

        print(f"\nQuery: {item['query']}")
        print(f"Latency: {item['latency_ms']:.2f} ms")
        print(f"Results: {item['results']}")
        print(f"Pages: {item['pages']}")

    average = total_latency / len(results)

    print(f"\nAverage latency: {average:.2f} ms")


def main():
    print("Vector Store Comparison")
    print(f"TOP_K = {TOP_K}")

    qdrant_results = benchmark_qdrant()
    chroma_results = benchmark_chroma()

    print_results("QDRANT", qdrant_results)
    print_results("CHROMA", chroma_results)

    qdrant_avg = sum(
        item["latency_ms"] for item in qdrant_results
    ) / len(qdrant_results)

    chroma_avg = sum(
        item["latency_ms"] for item in chroma_results
    ) / len(chroma_results)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Qdrant average latency : {qdrant_avg:.2f} ms")
    print(f"Chroma average latency : {chroma_avg:.2f} ms")


if __name__ == "__main__":
    main()
