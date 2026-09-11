"""Strictly read-only access to employee PTO balances."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    select,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.database import discard_failed_session, get_session_factory

logger = logging.getLogger(__name__)
PTOType = Literal[
    "VACATION",
    "SICK",
    "PERSONAL",
    "FLOATING_HOLIDAY",
    "OTHER",
]

_PTO_TYPE_ALIASES: dict[str, PTOType] = {
    "vacation": "VACATION",
    "sick": "SICK",
    "personal": "PERSONAL",
    "floating holiday": "FLOATING_HOLIDAY",
    "other": "OTHER",
}

_metadata = MetaData()
employee_pto_balances = Table(
    "employee_pto_balances",
    _metadata,
    Column("pto_balance_id", BigInteger, primary_key=True),
    Column("employee_id", String(50), nullable=False),
    Column("user_name", String(100), nullable=False),
    Column("employee_name", String(150), nullable=False),
    Column("email", String(255), nullable=True),
    Column("pto_type", String(32), nullable=False),
    Column("balance_year", Integer, nullable=False),
    Column("opening_balance", Numeric(8, 2), nullable=False),
    Column("accrued_hours", Numeric(8, 2), nullable=False),
    Column("used_hours", Numeric(8, 2), nullable=False),
    Column("pending_hours", Numeric(8, 2), nullable=False),
    Column("adjusted_hours", Numeric(8, 2), nullable=False),
    Column("available_hours", Numeric(8, 2), nullable=False),
    Column("last_updated", DateTime, nullable=False),
)


class PTORepositoryError(RuntimeError):
    """Raised when PTO data cannot be read safely."""


@dataclass(frozen=True)
class PTOBalance:
    """One employee PTO balance returned by the application repository."""

    pto_balance_id: int
    employee_id: str
    employee_name: str
    pto_type: PTOType
    balance_year: int
    opening_balance: Decimal
    accrued_hours: Decimal
    used_hours: Decimal
    pending_hours: Decimal
    adjusted_hours: Decimal
    available_hours: Decimal
    last_updated: datetime


@dataclass(frozen=True)
class PTOBalanceResult:
    """Typed repository result that makes a missing record explicit."""

    found: bool
    employee_id: str
    balance_year: int
    requested_pto_type: PTOType | None
    balances: tuple[PTOBalance, ...]


def normalize_pto_type(pto_type: str) -> PTOType:
    """Normalize supported human-readable PTO labels to database values."""

    normalized = " ".join(
        pto_type.strip().casefold().replace("_", " ").replace("-", " ").split()
    )
    try:
        return _PTO_TYPE_ALIASES[normalized]
    except KeyError as exc:
        allowed = ", ".join(_PTO_TYPE_ALIASES)
        raise ValueError(f"Invalid PTO type. Expected one of: {allowed}.") from exc


def _current_year() -> int:
    return datetime.now(UTC).year


def _decimal(value: Any) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def _to_balance(row: Any) -> PTOBalance:
    return PTOBalance(
        pto_balance_id=int(row["pto_balance_id"]),
        employee_id=str(row["employee_id"]),
        employee_name=str(row["employee_name"]),
        pto_type=normalize_pto_type(str(row["pto_type"])),
        balance_year=int(row["balance_year"]),
        opening_balance=_decimal(row["opening_balance"]),
        accrued_hours=_decimal(row["accrued_hours"]),
        used_hours=_decimal(row["used_hours"]),
        pending_hours=_decimal(row["pending_hours"]),
        adjusted_hours=_decimal(row["adjusted_hours"]),
        available_hours=_decimal(row["available_hours"]),
        last_updated=row["last_updated"],
    )


def get_current_pto_balance(
    employee_id: str | None = None,
    pto_type: str | None = None,
    balance_year: int | None = None,
    *,
    user_name: str | None = None,
    session: Session | None = None,
) -> PTOBalanceResult:
    """Read an employee's PTO balances using bound SQLAlchemy predicates."""

    normalized_employee_id = employee_id.strip() if employee_id else None
    normalized_user_name = user_name.strip().casefold() if user_name else None
    if bool(normalized_employee_id) == bool(normalized_user_name):
        raise ValueError("Provide exactly one employee ID or username.")
    if normalized_employee_id and len(normalized_employee_id) > 50:
        raise ValueError("Employee ID is too long.")
    if normalized_user_name and len(normalized_user_name) > 100:
        raise ValueError("Username is too long.")

    resolved_year = balance_year if balance_year is not None else _current_year()
    if not 1901 <= resolved_year <= 2155:
        raise ValueError("Balance year must be a valid MySQL YEAR value.")
    normalized_pto_type = normalize_pto_type(pto_type) if pto_type else None

    identity_predicate = (
        employee_pto_balances.c.employee_id == normalized_employee_id
        if normalized_employee_id
        else employee_pto_balances.c.user_name == normalized_user_name.upper()
    )
    statement = select(employee_pto_balances).where(
        identity_predicate,
        employee_pto_balances.c.balance_year == resolved_year,
    )
    if normalized_pto_type is not None:
        statement = statement.where(
            employee_pto_balances.c.pto_type == normalized_pto_type
        )
    statement = statement.order_by(employee_pto_balances.c.pto_type)

    owns_session = session is None
    resolved_session = session or get_session_factory()()
    try:
        rows = resolved_session.execute(statement).mappings().all()
    except SQLAlchemyError as exc:
        discard_failed_session(resolved_session, exc)
        logger.error(
            "PTO balance lookup failed",
            extra={
                "operation": "database",
                "event": "pto_balance_lookup_failed",
                "error_type": type(exc).__name__,
            },
        )
        raise PTORepositoryError("PTO balances could not be retrieved.") from exc
    finally:
        if owns_session:
            resolved_session.close()

    balances = tuple(_to_balance(row) for row in rows)
    resolved_employee_id = (
        balances[0].employee_id if balances else (normalized_employee_id or "")
    )
    return PTOBalanceResult(
        found=bool(balances),
        employee_id=resolved_employee_id,
        balance_year=resolved_year,
        requested_pto_type=normalized_pto_type,
        balances=balances,
    )


def get_employee_email(
    employee_id: str,
    *,
    session: Session | None = None,
) -> str | None:
    """Look up an employee's email address from their PTO balance records."""

    normalized_employee_id = employee_id.strip() if employee_id else ""
    if not normalized_employee_id:
        return None

    statement = (
        select(employee_pto_balances.c.email)
        .where(employee_pto_balances.c.employee_id == normalized_employee_id)
        .limit(1)
    )
    owns_session = session is None
    resolved_session = session or get_session_factory()()
    try:
        email = resolved_session.execute(statement).scalar_one_or_none()
    except SQLAlchemyError as exc:
        discard_failed_session(resolved_session, exc)
        logger.error(
            "Employee email lookup failed",
            extra={
                "operation": "database",
                "event": "employee_email_lookup_failed",
                "error_type": type(exc).__name__,
            },
        )
        raise PTORepositoryError("Employee email could not be retrieved.") from exc
    finally:
        if owns_session:
            resolved_session.close()

    return str(email).strip() if email else None
