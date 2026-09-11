from pathlib import Path

from docling.document_converter import DocumentConverter


def parse_document(
    input_file: Path,
    output_dir: Path,
) -> Path:
    """
    Parse a supported document using Docling and save
    the extracted content as Markdown.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing: {input_file.name}")

    converter = DocumentConverter()

    result = converter.convert(str(input_file))

    markdown = result.document.export_to_markdown()

    output_file = output_dir / f"{input_file.stem}.md"

    output_file.write_text(
        markdown,
        encoding="utf-8",
    )

    print(
        f"Saved: {output_file.name} "
        f"({len(markdown):,} characters)"
    )

    return output_file