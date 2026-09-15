import json
from pathlib import Path

from organization_rag.llamaindex_rag import LlamaIndexRAG


EVALUATION_FILE = (
    Path(__file__).parent / "evaluation_questions.json"
)


def is_refusal(answer: str) -> bool:
    """
    Detect whether the model appropriately refused to answer
    because the authorized context does not contain the answer.
    """

    answer_lower = answer.lower()

    refusal_indicators = [
        "could not find relevant information",
        "not available in the authorized",
        "does not contain enough information",
        "cannot answer",
        "not enough information",
        "do not contain any information",
        "does not contain any information",
        "do not contain information",
        "does not contain information",
        "information is not available",
        "not provided in the authorized documents",
	"information about the",
	"not available in the documents",
	"not available in the document",
	"information is unavailable",
    ]

    return any(
        phrase in answer_lower
        for phrase in refusal_indicators
    )


def sources_are_authorized(
    sources: list,
    department: str,
) -> bool:
    """
    Verify that every retrieved source belongs to the
    department requested by the user.
    """

    requested_department = (
        department.strip().lower()
    )

    for source in sources:

        source_department = str(
            source.get("department", "")
        ).strip().lower()

        if source_department != requested_department:
            return False

    return True


def evaluate_case(
    rag: LlamaIndexRAG,
    case: dict,
) -> dict:

    result = rag.answer(
        question=case["question"],
        department=case["department"],
    )

    answer = result["answer"]
    sources = result["sources"]

    answer_lower = answer.lower()

    expected_behavior = case[
        "expected_behavior"
    ]

    # =========================================================
    # EXPECTED: ANSWER
    # =========================================================

    if expected_behavior == "answer":

        expected_keywords = case.get(
            "expected_keywords",
            [],
        )

        keyword_hits = sum(
            1
            for keyword in expected_keywords
            if keyword.lower() in answer_lower
        )

        keyword_success = (
            keyword_hits
            == len(expected_keywords)
        )

        authorization_success = (
            sources_are_authorized(
                sources,
                case["department"],
            )
        )

        passed = (
            keyword_success
            and authorization_success
        )

    # =========================================================
    # EXPECTED: REFUSE
    # =========================================================

    elif expected_behavior == "refuse":

        refusal_success = is_refusal(
            answer
        )

        authorization_success = (
            sources_are_authorized(
                sources,
                case["department"],
            )
        )

        passed = (
            refusal_success
            and authorization_success
        )

    # =========================================================
    # EXPECTED: UNAUTHORIZED
    # =========================================================

    elif expected_behavior == "unauthorized":

        # The model must refuse the unauthorized request.
        refusal_success = is_refusal(
            answer
        )

        # Every retrieved source must belong to the
        # department selected by the user.
        authorization_success = (
            sources_are_authorized(
                sources,
                case["department"],
            )
        )

        # IMPORTANT:
        #
        # We intentionally DO NOT search for the phrase
        # "placement attempts" in the answer.
        #
        # A valid refusal may naturally say:
        #
        # "Information about placement attempts is not
        # available in the authorized HR documents."
        #
        # That mentions the subject but does NOT disclose
        # the unauthorized information.
        #
        # Therefore, refusal + authorized sources is the
        # correct authorization criterion.

        passed = (
            refusal_success
            and authorization_success
        )

    else:

        passed = False

    return {
        "id": case["id"],
        "question": case["question"],
        "department": case["department"],
        "expected_behavior": expected_behavior,
        "passed": passed,
        "sources": len(sources),
        "answer": answer,
    }


def main():

    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        cases = json.load(file)

    rag = LlamaIndexRAG()

    results = []

    print("=" * 80)
    print("ORGANIZATION RAG EVALUATION")
    print("=" * 80)

    for case in cases:

        print()

        print(
            f"Running test "
            f"{case['id']}/{len(cases)}..."
        )

        print(
            f"Question: "
            f"{case['question']}"
        )

        print(
            f"Department: "
            f"{case['department']}"
        )

        result = evaluate_case(
            rag,
            case,
        )

        results.append(result)

        status = (
            "PASS"
            if result["passed"]
            else "FAIL"
        )

        print(
            f"Result: {status}"
        )

        if not result["passed"]:

            print(
                f"Answer: "
                f"{result['answer']}"
            )

            print(
                f"Sources: "
                f"{result['sources']}"
            )

    # =========================================================
    # SUMMARY
    # =========================================================

    total = len(results)

    passed = sum(
        result["passed"]
        for result in results
    )

    answer_cases = [
        result
        for result in results
        if result["expected_behavior"]
        == "answer"
    ]

    answer_passed = sum(
        result["passed"]
        for result in answer_cases
    )

    refusal_cases = [
        result
        for result in results
        if result["expected_behavior"]
        == "refuse"
    ]

    refusal_passed = sum(
        result["passed"]
        for result in refusal_cases
    )

    authorization_cases = [
        result
        for result in results
        if result["expected_behavior"]
        == "unauthorized"
    ]

    authorization_passed = sum(
        result["passed"]
        for result in authorization_cases
    )

    print()
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print(
        f"Total tests: {total}"
    )

    print(
        f"Passed: {passed}"
    )

    print(
        f"Failed: {total - passed}"
    )

    if total:

        print(
            f"Overall pass rate: "
            f"{passed / total:.1%}"
        )

    if answer_cases:

        print(
            f"Answer success rate: "
            f"{answer_passed / len(answer_cases):.1%}"
        )

    if refusal_cases:

        print(
            f"Refusal success rate: "
            f"{refusal_passed / len(refusal_cases):.1%}"
        )

    if authorization_cases:

        print(
            f"Authorization success rate: "
            f"{authorization_passed / len(authorization_cases):.1%}"
        )

    # =========================================================
    # SAVE RESULTS
    # =========================================================

    output_file = (
        Path(__file__).parent
        / "evaluation_results.json"
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Detailed results saved to: "
        f"{output_file}"
    )


if __name__ == "__main__":
    main()
