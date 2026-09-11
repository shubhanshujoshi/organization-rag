from pathlib import Path

from organization_rag.document_loader import discover_documents
from organization_rag.parser import parse_document


RAW_DIR = Path("data/raw")
MARKDOWN_DIR = Path("data/markdown")


def main() -> None:
    documents = discover_documents(RAW_DIR)

    if not documents:
        print("No supported documents found.")
        return

    print(f"Found {len(documents)} document(s).")
    print()

    for document in documents:
        try:
            parse_document(
                input_file=document,
                output_dir=MARKDOWN_DIR,
            )
        except Exception as exc:
            print(
                f"Failed to parse {document.name}: {exc}"
            )

        print()

    print("Document parsing completed.")


if __name__ == "__main__":
    main()