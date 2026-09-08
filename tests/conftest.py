"""Non-production environment used by the isolated test suite."""

import json
import os

_TEST_PASSWORD_HASH = (
    "cf74315433aa7ca9fc5eea7f5a68db891e4ac7704b0c4ce8104f1e490d8aba48"
)
_TEST_USERS = [
    {
        "username": "admin",
        "employee_id": "ADMIN001",
        "password_salt": "docuverse-prototype-v1",
        "password_hash": _TEST_PASSWORD_HASH,
        "role": "admin",
    },
    {
        "username": "user01",
        "employee_id": "EMP001",
        "password_salt": "docuverse-prototype-v1",
        "password_hash": _TEST_PASSWORD_HASH,
        "role": "user",
    },
    {
        "username": "user02",
        "employee_id": "EMP002",
        "password_salt": "docuverse-prototype-v1",
        "password_hash": _TEST_PASSWORD_HASH,
        "role": "user",
    },
]

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("BACKEND_URL", "http://localhost:8000")
os.environ.setdefault("AUTH_USERS_JSON", json.dumps(_TEST_USERS))
