from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".xlsx",
    ".xls",
    ".csv",
    ".txt",
    ".md",
}


def discover_documents(data_dir: Path) -> list[Path]:
    """
    Discover supported documents inside a directory.
    """

    documents = []

    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents.append(path)

    return sorted(documents)


def main() -> None:
    data_dir = Path("data/raw")

    documents = discover_documents(data_dir)

    print(f"Documents found: {len(documents)}")
    print()

    for document in documents:
        print(f"- {document.name}")


if __name__ == "__main__":
    main()