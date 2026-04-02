from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from typing_extensions import TypedDict


class IntentStatus(str, Enum):
    VALID = "valid"
    INVALID = "invalid"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    NOT_FOUND = "not_found"
    API_ERROR = "api_error"
    PARTIAL_DATA = "partial_data"


class AgentState(TypedDict):
    # Input
    user_query: str

    # Node 1 — intent identification
    intent_status: Optional[IntentStatus]
    country_name: Optional[str]
    requested_fields: list[str]
    intent_error_msg: Optional[str]
    was_corrected: bool          # True when country name was inferred from a misspelling

    # Node 2 — tool invocation
    tool_status: Optional[ToolStatus]
    raw_api_data: Optional[dict[str, Any]]
    tool_error_msg: Optional[str]

    # Node 3 — answer synthesis
    final_answer: Optional[str]
