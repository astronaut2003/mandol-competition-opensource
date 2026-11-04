"""Structural AML Add and Search contracts; media validation happens separately."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ImageURL(_ContractModel):
    url: str = Field(min_length=1)


class TextPart(_ContractModel):
    type: Literal["text"]
    text: str = Field(min_length=1)


class ImagePart(_ContractModel):
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]
MessageContent = Annotated[str | list[ContentPart], Field(min_length=1)]


class Message(_ContractModel):
    role: Literal["user", "assistant"]
    timestamp: StrictInt | None = Field(default=None,ge=-(2**63),le=2**63-1)
    content: MessageContent

    def parts(self) -> list[ContentPart]:
        if isinstance(self.content, str):
            return [TextPart(type="text", text=self.content)]
        return list(self.content)


class AddRequest(_ContractModel):
    request_id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class AddResponse(_ContractModel):
    success: Literal[True]
    request_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class SearchRequest(_ContractModel):
    query: MessageContent
    options: list[str] | None = None
    user_id: str = Field(min_length=1)
    top_k: StrictInt = Field(gt=0)

    def parts(self) -> list[ContentPart]:
        if isinstance(self.query, str):
            return [TextPart(type="text", text=self.query)]
        return list(self.query)


class SearchItem(_ContractModel):
    model_config = ConfigDict(strict=True,extra='ignore',allow_inf_nan=False)
    id: str = Field(min_length=1)
    content: MessageContent
    score: float | None = None
    created_at: str | None = None


class SearchResponse(_ContractModel):
    data: list[SearchItem]


def join_text_parts(parts: list[ContentPart]) -> str:
    return "\n".join(part.text for part in parts if isinstance(part, TextPart))
