from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from qdrant_client import QdrantClient

from organization_rag.config import COLLECTION_NAME, QDRANT_URL
from organization_rag.llamaindex_store import build_index


def main():
    # Create a tiny test document
    document = Document(
        text=(
            "The placement office coordinates campus recruitment. "
            "Students can participate in placement activities according "
            "to the organization's placement policies."
        ),
        metadata={
            "source": "test_document",
            "department": "placement",
        },
    )

    # Convert document into LlamaIndex nodes
    splitter = SentenceSplitter(
        chunk_size=100,
        chunk_overlap=20,
    )

    nodes = splitter.get_nodes_from_documents([document])

    print(f"Nodes created: {len(nodes)}")

    # Build LlamaIndex index backed by Qdrant
    index = build_index(nodes)

    print("LlamaIndex index created.")

    # Check Qdrant directly
    client = QdrantClient(url=QDRANT_URL)

    info = client.get_collection(COLLECTION_NAME)

    print(f"Qdrant points: {info.points_count}")

    # Test LlamaIndex retrieval
    retriever = index.as_retriever(similarity_top_k=1)

    results = retriever.retrieve(
        "What does the placement office do?"
    )

    print(f"Retrieved results: {len(results)}")

    for result in results:
        print()
        print("Retrieved text:")
        print(result.text)


if __name__ == "__main__":
    main()