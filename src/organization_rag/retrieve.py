from qdrant_client import QdrantClient

from organization_rag.config import (
    COLLECTION_NAME,
    QDRANT_URL,
    TOP_K,
)

from organization_rag.embeddings import get_embedding


def retrieve(query: str, top_k: int = TOP_K):
    query_embedding = get_embedding(query)

    client = QdrantClient(url=QDRANT_URL)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    return results.points


def main() -> None:
    question = input("Enter your question: ")

    results = retrieve(question)

    print("\nTop retrieved chunks:")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Score: {result.score:.4f}")
        print(f"Chunk ID: {result.payload['chunk_id']}")
        print(f"Source: {result.payload['source']}")
        print()
        print(result.payload["text"][:1000])


if __name__ == "__main__":
    main()