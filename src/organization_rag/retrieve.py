import requests
from qdrant_client import QdrantClient


OLLAMA_URL = "http://localhost:11434/api/embed"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "organization_documents"


def get_embedding(text: str) -> list[float]:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "nomic-embed-text",
            "input": text,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]


def main() -> None:
    query = input("Enter your question: ")

    # Convert the user's question into a vector.
    query_embedding = get_embedding(query)

    # Connect to Qdrant.
    client = QdrantClient(url=QDRANT_URL)

    # Search for the most similar chunks.
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=3,
        with_payload=True,
    )

    print("\nTop retrieved chunks:")
    print("=" * 80)

    for rank, result in enumerate(results.points, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Score: {result.score:.4f}")
        print(f"Chunk ID: {result.payload['chunk_id']}")
        print(f"Source: {result.payload['source']}")
        print()
        print(result.payload["text"][:1000])


if __name__ == "__main__":
    main()