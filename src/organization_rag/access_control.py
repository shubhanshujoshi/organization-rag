ALLOWED_DEPARTMENTS = {
    "placement",
    "hr",
    "finance",
    "academic",
    "admissions",
    "administration",
    "library",
    "hostel",
    "it",
    "general",
}


def normalize_department(department: str) -> str:
    return department.strip().lower()


def is_valid_department(department: str) -> bool:
    return normalize_department(department) in ALLOWED_DEPARTMENTS


def can_access(requested_department: str, document_department: str) -> bool:
    return (
        normalize_department(requested_department)
        == normalize_department(document_department)
    )
