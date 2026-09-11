from pathlib import Path

from docling.document_converter import DocumentConverter


INPUT_FILE = Path(
    "data/raw/Placement Handbook for Session 2026-2027 (1).pdf"
)

OUTPUT_DIR = Path("data/markdown")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Parsing: {INPUT_FILE}")

    converter = DocumentConverter()
    result = converter.convert(str(INPUT_FILE))

    markdown = result.document.export_to_markdown()

    output_file = OUTPUT_DIR / f"{INPUT_FILE.stem}.md"
    output_file.write_text(markdown, encoding="utf-8")

    print(f"Saved: {output_file}")
    print(f"Characters extracted: {len(markdown):,}")


if __name__ == "__main__":
    main()