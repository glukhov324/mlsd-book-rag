from .categories import QUESTION_CATEGORY_VALUES

GENERATE_SCHEMA = {
    "type": "object",
    "properties": {
        "able_to_create_answer": {"type": "boolean"},
        "category": {"type": "string", "enum": QUESTION_CATEGORY_VALUES},
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["able_to_create_answer", "category", "question", "answer", "evidence"],
}

VALIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "accept": {"type": "boolean"},
        "category": {"type": "string", "enum": QUESTION_CATEGORY_VALUES},
        "reason": {"type": "string"},
    },
    "required": ["accept", "category", "reason"],
}
