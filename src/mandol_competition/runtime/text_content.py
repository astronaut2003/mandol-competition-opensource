"""Ordered plain-text wire content for the memory-api-v1.1 boundary."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class TextContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"]
    text: str = Field(min_length=1)


TextContent = Annotated[str | list[TextContentPart], Field(min_length=1)]


def text_content(value: str | list[TextContentPart]) -> str:
    """Keep strings unchanged and join ordered text parts at their boundary."""
    if isinstance(value, str):
        return value
    return "\n".join(part.text for part in value)
