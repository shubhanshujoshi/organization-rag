from pathlib import Path

from llama_index.core import Document

from organization_rag.chunk import chunk_documents
from organization_rag.document_loader import discover_documents
from organization_rag.parser import parse_document_pages
from organization_rag.postgres_store import PostgresVectorStore


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

            documents.append(
                Document(
                    text=page["text"],
                    metadata=metadata,
                )
            )

        all_chunks.extend(chunk_documents(documents))

    print(f"\nTotal chunks: {len(all_chunks)}")

    store = PostgresVectorStore()
    store.reset()

    count = store.add_chunks(all_chunks)

    print("\n" + "=" * 50)
    print("POSTGRESQL + PGVECTOR INGESTION COMPLETED")
    print("=" * 50)
    print(f"Chunks stored: {count}")
    print("Table: document_chunks")
    print("Database: organization_rag")

    store.close()


if __name__ == "__main__":
    main()
