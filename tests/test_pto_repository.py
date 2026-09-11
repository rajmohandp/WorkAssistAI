"""Unit tests for the read-only PTO repository."""

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import OperationalError

from app.repositories.pto_repository import (
    PTORepositoryError,
    get_current_pto_balance,
    get_employee_email,
)


def row(pto_type: str, available_hours: str) -> dict[str, object]:
    return {
        "pto_balance_id": 1 if pto_type == "VACATION" else 2,
        "employee_id": "EMP001",
        "employee_name": "Example Employee",
        "pto_type": pto_type,
        "balance_year": 2026,
        "opening_balance": Decimal("40.00"),
        "accrued_hours": Decimal("48.00"),
        "used_hours": Decimal("24.00"),
        "pending_hours": Decimal("8.00"),
        "adjusted_hours": Decimal("0.00"),
        "available_hours": Decimal(available_hours),
        "last_updated": datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    }


def session_with_rows(*rows: dict[str, object]) -> MagicMock:
    session = MagicMock()
    session.execute.return_value.mappings.return_value.all.return_value = list(rows)
    return session


def compiled_parameters(session: MagicMock) -> dict[str, object]:
    statement = session.execute.call_args.args[0]
    return statement.compile().params


def test_returns_vacation_and_sick_balances():
    session = session_with_rows(row("VACATION", "56.00"), row("SICK", "40.00"))

    result = get_current_pto_balance("EMP001", balance_year=2026, session=session)

    assert result.found is True
    assert {balance.pto_type for balance in result.balances} == {"VACATION", "SICK"}
    assert result.balances[0].available_hours == Decimal("56.00")


def test_filters_by_normalized_specific_pto_type():
    session = session_with_rows(row("FLOATING_HOLIDAY", "8.00"))

    result = get_current_pto_balance(
        "EMP001",
        " floating-holiday ",
        2026,
        session=session,
    )

    assert result.requested_pto_type == "FLOATING_HOLIDAY"
    assert "FLOATING_HOLIDAY" in compiled_parameters(session).values()


def test_filters_by_normalized_username():
    session = session_with_rows(row("VACATION", "56.00"))

    result = get_current_pto_balance(
        user_name=" User01 ", balance_year=2026, session=session
    )

    assert result.found is True
    assert result.employee_id == "EMP001"
    statement = session.execute.call_args.args[0]
    assert "employee_pto_balances.user_name" in str(statement)
    assert "USER01" in statement.compile().params.values()


def test_returns_explicit_not_found_result_for_missing_employee():
    result = get_current_pto_balance(
        "MISSING",
        balance_year=2026,
        session=session_with_rows(),
    )

    assert result.found is False
    assert result.balances == ()
    assert result.employee_id == "MISSING"


def test_rejects_invalid_pto_type_before_database_access():
    session = session_with_rows()

    with pytest.raises(ValueError, match="Invalid PTO type"):
        get_current_pto_balance("EMP001", "unlimited", session=session)
    session.execute.assert_not_called()


def test_defaults_to_current_year(monkeypatch):
    session = session_with_rows()
    monkeypatch.setattr("app.repositories.pto_repository._current_year", lambda: 2030)

    result = get_current_pto_balance("EMP001", session=session)

    assert result.balance_year == 2030
    assert 2030 in compiled_parameters(session).values()


def test_wraps_database_exception_without_provider_details():
    session = MagicMock()
    session.execute.side_effect = OperationalError(
        "SELECT sensitive detail",
        {},
        Exception("provider detail"),
    )

    with pytest.raises(PTORepositoryError) as caught:
        get_current_pto_balance("EMP001", balance_year=2026, session=session)
    assert str(caught.value) == "PTO balances could not be retrieved."


def test_get_employee_email_returns_stored_address():
    session = MagicMock()
    session.execute.return_value.scalar_one_or_none.return_value = (
        "employee@example.com"
    )

    result = get_employee_email("EMP001", session=session)

    assert result == "employee@example.com"


def test_get_employee_email_returns_none_when_missing():
    session = MagicMock()
    session.execute.return_value.scalar_one_or_none.return_value = None

    assert get_employee_email("MISSING", session=session) is None


def test_get_employee_email_returns_none_for_blank_employee_id():
    session = MagicMock()

    assert get_employee_email("  ", session=session) is None
    session.execute.assert_not_called()


def test_get_employee_email_wraps_database_exception():
    session = MagicMock()
    session.execute.side_effect = OperationalError(
        "SELECT sensitive detail",
        {},
        Exception("provider detail"),
    )

    with pytest.raises(PTORepositoryError) as caught:
        get_employee_email("EMP001", session=session)
    assert str(caught.value) == "Employee email could not be retrieved."
