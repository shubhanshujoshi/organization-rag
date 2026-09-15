from pathlib import Path

from organization_rag.parser import parse_document_pages


def test_supported_extensions():
    from organization_rag.parser import SUPPORTED_EXTENSIONS

    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS
    assert ".xlsx" in SUPPORTED_EXTENSIONS
    assert ".xls" in SUPPORTED_EXTENSIONS


def test_csv_parser(tmp_path: Path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text(
        "Name,Department\nAlice,HR\nBob,Finance\n",
        encoding="utf-8",
    )

    pages = parse_document_pages(csv_file)

    assert len(pages) == 1
    assert "Alice" in pages[0]["text"]
    assert "Finance" in pages[0]["text"]
