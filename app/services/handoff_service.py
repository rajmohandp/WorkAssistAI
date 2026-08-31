"""Idempotent internal queue for policy questions requiring human review."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from threading import Lock
from typing import Literal


@dataclass(frozen=True)
class HandoffRecord:
    handoff_id: str
    status: Literal["queued"]
    reason: str
    username: str
    employee_id: str
    question: str
    conversation_history: tuple[dict[str, str], ...]
    created_at: str


class HandoffService:
    """Process-local queue with deterministic duplicate prevention."""

    def __init__(self) -> None:
        self._records: dict[str, HandoffRecord] = {}
        self._lock = Lock()

    @staticmethod
    def idempotency_key(employee_id: str, question: str) -> str:
        normalized = " ".join(question.casefold().split())
        return sha256(f"{employee_id.upper()}\0{normalized}".encode()).hexdigest()

    async def create_handoff(
        self,
        *,
        username: str,
        employee_id: str,
        question: str,
        conversation_history: list[dict[str, str]],
        reason: str,
    ) -> HandoffRecord:
        key = self.idempotency_key(employee_id, question)
        handoff_id = f"WA-{key[:12].upper()}"
        with self._lock:
            existing = self._records.get(key)
            if existing is not None:
                return existing
            record = HandoffRecord(
                handoff_id=handoff_id,
                status="queued",
                reason=reason,
                username=username,
                employee_id=employee_id,
                question=question,
                conversation_history=tuple(conversation_history[-6:]),
                created_at=datetime.now(UTC).isoformat(),
            )
            self._records[key] = record
            return record

    def list_handoffs(self) -> list[dict[str, object]]:
        with self._lock:
            records = sorted(
                self._records.values(), key=lambda item: item.created_at, reverse=True
            )
        return [asdict(record) for record in records]

    def clear(self) -> None:
        """Clear process-local records for isolated tests."""

        with self._lock:
            self._records.clear()


handoff_service = HandoffService()
