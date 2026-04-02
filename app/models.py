from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User question about a country")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Human-readable answer to the query")
    country: Optional[str] = Field(None, description="Extracted country name")
    fields_requested: list[str] = Field(default_factory=list, description="Fields identified in the query")
    tool_status: Optional[str] = Field(None, description="Status of the API call: success | not_found | api_error | partial_data")


class HealthResponse(BaseModel):
    status: str
    model: str
