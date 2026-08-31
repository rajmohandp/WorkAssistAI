"""Structured LangChain tool for employee PTO balance lookups."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

from app.repositories import pto_repository
from app.repositories.pto_repository import PTOBalanceResult, PTORepositoryError

PTO_TOOL_DESCRIPTION = (
    "Use this tool only for questions about an employee's current PTO, vacation, "
    "sick, personal leave, floating-holiday, accrued, used, pending, or available "
    "leave balance."
)


class PTOBalanceToolInput(BaseModel):
    """Validated arguments accepted by the PTO balance tool."""

    employee_id: str | None = Field(
        default=None,
        max_length=50,
        description="The authenticated employee's immutable employee identifier.",
    )
    user_name: str | None = Field(
        default=None,
        max_length=100,
        description="The authenticated user's normalized login name.",
    )
    pto_type: str | None = Field(
        default=None,
        description=(
            "Optional PTO category: vacation, sick, personal, floating holiday, "
            "or other. Omit to return every category."
        ),
    )
    balance_year: int | None = Field(
        default=None,
        ge=1901,
        le=2155,
        description="Optional balance year. Omit to use the current year.",
    )

    @model_validator(mode="after")
    def validate_single_identity(self) -> PTOBalanceToolInput:
        employee_id = self.employee_id.strip() if self.employee_id else ""
        user_name = self.user_name.strip() if self.user_name else ""
        if bool(employee_id) == bool(user_name):
            raise ValueError("Provide exactly one employee ID or username.")
        return self


class PTOCategoryOutput(BaseModel):
    pto_type: str
    opening_balance: Decimal
    accrued_hours: Decimal
    used_hours: Decimal
    pending_hours: Decimal
    adjusted_hours: Decimal
    available_hours: Decimal


class PTOBalanceToolOutput(BaseModel):
    employee_id: str
    employee_name: str | None
    balance_year: int
    balances: list[PTOCategoryOutput] = Field(default_factory=list)
    last_updated: datetime | None
    source: Literal["aws_mysql"] = "aws_mysql"
    found: bool
    error: str | None = None


def _current_year() -> int:
    return datetime.now(UTC).year


def _serialize_result(result: PTOBalanceResult) -> dict[str, object]:
    balances = [
        PTOCategoryOutput(
            pto_type=balance.pto_type,
            opening_balance=balance.opening_balance,
            accrued_hours=balance.accrued_hours,
            used_hours=balance.used_hours,
            pending_hours=balance.pending_hours,
            adjusted_hours=balance.adjusted_hours,
            available_hours=balance.available_hours,
        )
        for balance in result.balances
    ]
    output = PTOBalanceToolOutput(
        employee_id=result.employee_id,
        employee_name=result.balances[0].employee_name if result.balances else None,
        balance_year=result.balance_year,
        balances=balances,
        last_updated=(
            max(balance.last_updated for balance in result.balances)
            if result.balances
            else None
        ),
        found=result.found,
    )
    return output.model_dump(mode="json")


@tool(
    "get_current_pto_balance",
    args_schema=PTOBalanceToolInput,
    description=PTO_TOOL_DESCRIPTION,
)
def get_current_pto_balance(
    employee_id: str | None = None,
    user_name: str | None = None,
    pto_type: str | None = None,
    balance_year: int | None = None,
) -> dict[str, object]:
    """Return structured, read-only PTO balance data from AWS MySQL."""

    try:
        result = pto_repository.get_current_pto_balance(
            employee_id=employee_id,
            user_name=user_name,
            pto_type=pto_type,
            balance_year=balance_year,
        )
    except PTORepositoryError:
        output = PTOBalanceToolOutput(
            employee_id=employee_id.strip() if employee_id else "",
            employee_name=None,
            balance_year=balance_year or _current_year(),
            balances=[],
            last_updated=None,
            found=False,
            error="PTO balance data is currently unavailable.",
        )
        return output.model_dump(mode="json")
    return _serialize_result(result)
