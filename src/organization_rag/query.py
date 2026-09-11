from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "organization_documents"


def main() -> None:
    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
    )

    client = QdrantClient(url=QDRANT_URL)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
    )

    index = VectorStoreIndex.from_vector_store(vector_store)

    retriever = index.as_retriever(similarity_top_k=3)

    question = input("\nAsk a question: ")

    results = retriever.retrieve(question)

    print("\nRetrieved results:")
    print("=" * 80)

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("-" * 80)
        print(result.text[:1500])
        print(f"\nSimilarity score: {result.score}")


if __name__ == "__main__":
    main()