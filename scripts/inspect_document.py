from pathlib import Path

from docling.document_converter import DocumentConverter


INPUT_FILE = Path(
    "data/raw/Placement Handbook for Session 2026-2027 (1).pdf"
)


def main() -> None:
    converter = DocumentConverter()
    result = converter.convert(str(INPUT_FILE))
    document = result.document

    print(f"Pages: {document.num_pages}")
    print(f"Tables: {len(document.tables)}")
    print(f"Pictures: {len(document.pictures)}")
    print(f"Text items: {len(document.texts)}")

    print("\nFirst 25 document items:")
    print("-" * 80)

    for index, item in enumerate(document.texts[:25]):
        page_no = None

        if item.prov:
            page_no = item.prov[0].page_no

        print(
            f"{index:02d} | "
            f"{type(item).__name__:20} | "
            f"page={page_no!s:>2} | "
            f"{item.text[:120]!r}"
        )


if __name__ == "__main__":
    main()