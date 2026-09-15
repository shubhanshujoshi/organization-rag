from pathlib import Path

from organization_rag.chunk import chunk_documents
from organization_rag.document_loader import discover_documents
from organization_rag.parser import parse_document_pages
from organization_rag.chroma_store import ChromaStore


RAW_DIR = Path("data/raw")


def main():
    source_documents = discover_documents(RAW_DIR)

    print(f"Found {len(source_documents)} source document(s).")

    all_chunks = []

    for source in source_documents:
        print(f"\nParsing: {source.name}")

        pages = parse_document_pages(source)

        department = source.parent.name

        documents = []

        for page in pages:
            metadata = {
                "source": source.name,
                "source_path": str(source),
                "document_type": source.suffix.lower().lstrip("."),
                "department": department,
                "page": page["page"],
            }

            if page.get("section"):
                metadata["section"] = page["section"]

            from llama_index.core import Document

            documents.append(
                Document(
                    text=page["text"],
                    metadata=metadata,
                )
            )

        chunks = chunk_documents(documents)

        all_chunks.extend(chunks)

    print(f"\nTotal chunks: {len(all_chunks)}")

    store = ChromaStore()
    store.reset()

    count = store.add_chunks(all_chunks)

    print("\n" + "=" * 50)
    print("CHROMA INGESTION COMPLETED")
    print("=" * 50)
    print(f"Chunks stored: {count}")
    print(f"Database: {store.collection.name}")
    print(f"Path: {store.collection}")
    

if __name__ == "__main__":
    main()
