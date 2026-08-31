"""Focused tests for the structured PTO LangChain tool."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from app.repositories.pto_repository import (
    PTOBalance,
    PTOBalanceResult,
    PTORepositoryError,
    PTOType,
)
from app.tools.pto_tool import PTOBalanceToolInput, get_current_pto_balance


def balance(pto_type: PTOType = "VACATION") -> PTOBalance:
    return PTOBalance(
        pto_balance_id=1,
        employee_id="EMP001",
        employee_name="Example Employee",
        pto_type=pto_type,
        balance_year=2026,
        opening_balance=Decimal("40.00"),
        accrued_hours=Decimal("48.00"),
        used_hours=Decimal("24.00"),
        pending_hours=Decimal("8.00"),
        adjusted_hours=Decimal("0.00"),
        available_hours=Decimal("56.00"),
        last_updated=datetime(2026, 2, 3, 4, 5, 6, tzinfo=UTC),
    )


def test_tool_has_typed_schema_and_required_description():
    assert isinstance(get_current_pto_balance, BaseTool)
    assert get_current_pto_balance.name == "get_current_pto_balance"
    assert get_current_pto_balance.args_schema is PTOBalanceToolInput
    assert "Use this tool only for questions about an employee's current PTO" in (
        get_current_pto_balance.description
    )


def test_tool_returns_structured_pto_data(monkeypatch):
    repository_result = PTOBalanceResult(
        found=True,
        employee_id="EMP001",
        balance_year=2026,
        requested_pto_type=None,
        balances=(balance("VACATION"), balance("SICK")),
    )
    captured = {}

    def lookup(**kwargs):
        captured.update(kwargs)
        return repository_result

    monkeypatch.setattr(
        "app.tools.pto_tool.pto_repository.get_current_pto_balance",
        lookup,
    )

    result = get_current_pto_balance.invoke(
        {"employee_id": "EMP001", "balance_year": 2026}
    )

    assert captured == {
        "employee_id": "EMP001",
        "user_name": None,
        "pto_type": None,
        "balance_year": 2026,
    }
    assert result["employee_id"] == "EMP001"
    assert result["employee_name"] == "Example Employee"
    assert result["balance_year"] == 2026
    assert result["source"] == "aws_mysql"
    assert result["last_updated"] == "2026-02-03T04:05:06Z"
    assert {item["pto_type"] for item in result["balances"]} == {
        "VACATION",
        "SICK",
    }
    assert result["balances"][0]["available_hours"] == "56.00"


def test_tool_returns_structured_not_found_result(monkeypatch):
    monkeypatch.setattr(
        "app.tools.pto_tool.pto_repository.get_current_pto_balance",
        lambda **_kwargs: PTOBalanceResult(
            found=False,
            employee_id="MISSING",
            balance_year=2026,
            requested_pto_type="VACATION",
            balances=(),
        ),
    )

    result = get_current_pto_balance.invoke(
        {
            "employee_id": "MISSING",
            "pto_type": "vacation",
            "balance_year": 2026,
        }
    )

    assert result["found"] is False
    assert result["balances"] == []
    assert result["employee_name"] is None
    assert result["last_updated"] is None
    assert result["error"] is None


def test_tool_sanitizes_repository_exceptions(monkeypatch):
    def fail(**_kwargs):
        raise PTORepositoryError("internal database detail")

    monkeypatch.setattr(
        "app.tools.pto_tool.pto_repository.get_current_pto_balance",
        fail,
    )

    result = get_current_pto_balance.invoke(
        {"employee_id": "EMP001", "balance_year": 2026}
    )

    assert result["error"] == "PTO balance data is currently unavailable."
    assert "internal database detail" not in str(result)
    assert result["source"] == "aws_mysql"


def test_tool_schema_rejects_invalid_input_before_repository(monkeypatch):
    called = False

    def lookup(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(
        "app.tools.pto_tool.pto_repository.get_current_pto_balance",
        lookup,
    )

    with pytest.raises(ValidationError):
        get_current_pto_balance.invoke({"employee_id": "", "balance_year": 1800})

    assert called is False
