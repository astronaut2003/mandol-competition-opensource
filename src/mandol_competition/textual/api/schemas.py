"""Pydantic models for the AML Add/Search contract.

Contract reference (https://agentmemories.ai/docs):

* Add request : request_id, messages[{role, content, timestamp?}], user_id,
                session_id
* Add response: HTTP 200 with {"success": true, "request_id", "user_id",
                "session_id"} (all ids echoed byte-for-byte)
* Search req  : query, options?, user_id, top_k
* Search resp : {"data": [{id, content, score?, created_at?}]} ordered by
                relevance, at most top_k items
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator


class AddMessage(BaseModel):
    """One message inside an Add request."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)
    timestamp: StrictInt | None = Field(default=None,ge=-(2**63),le=2**63-1,
        description="Unix milliseconds representable by the canonical SQLite integer column")

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must not be blank")
        return value

    @field_validator("role")
    @classmethod
    def _role_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("role must not be blank")
        return value


class AddRequest(BaseModel):
    request_id: str = Field(..., min_length=1)
    messages: list[AddMessage] = Field(..., min_length=1, max_length=1024)
    user_id: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)


class AddResponse(BaseModel):
    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    options: list[str] | None = None
    user_id: str = Field(..., min_length=1)
    top_k: StrictInt = Field(..., ge=1)

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class SearchHit(BaseModel):
    model_config = ConfigDict(strict=True,allow_inf_nan=False)
    id: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    score: float | None = None
    created_at: str | None = None


class SearchResponse(BaseModel):
    data: list[SearchHit] = Field(default_factory=list)
