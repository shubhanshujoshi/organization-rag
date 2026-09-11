from qdrant_client import QdrantClient, models

from organization_rag.config import (
    COLLECTION_NAME,
    QDRANT_URL,
    TOP_K,
)

from organization_rag.embeddings import get_embedding


def retrieve(
    query: str,
    department: str | None = None,
    top_k: int = TOP_K,
):
    """
    Retrieve the most relevant document chunks from Qdrant.

    If a department is provided, only documents belonging
    to that department are retrieved.
    """

    query_embedding = get_embedding(query)

    client = QdrantClient(url=QDRANT_URL)

    query_filter = None

    if department:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="department",
                    match=models.MatchValue(value=department),
                )
            ]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    return results.points


def main() -> None:
    question = input("Enter your question: ")

    department = input(
        "Enter department (e.g. placement) or press Enter for all: "
    ).strip()

    if not department:
        department = None

    results = retrieve(
        query=question,
        department=department,
    )

    print("\nTop retrieved chunks:")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Score: {result.score:.4f}")
        print(f"Chunk ID: {result.payload['chunk_id']}")
        print(f"Department: {result.payload['department']}")
        print(f"Source: {result.payload['source']}")
        print()
        print(result.payload["text"][:1000])


if __name__ == "__main__":
    main()