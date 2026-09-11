from organization_rag.rag import OrganizationRAG


def main() -> None:
    question = input("Enter your question: ")

    rag = OrganizationRAG()

    answer = rag.answer(question)

    print("\nAnswer:")
    print("=" * 80)
    print(answer)


if __name__ == "__main__":
    main()