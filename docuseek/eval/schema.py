"""
docuseek/eval/schema.py
----------------
Pydantic model of gold eval set.
"""

from typing import Literal

from pydantic import BaseModel, Field


class GoldQuestion(BaseModel):
    question: str = Field(..., min_length=10)
    answer: str = Field(..., min_length=10)
    source_urls: list[str] = Field(..., min_length=1)
    library: Literal["transformers", "diffusers", "datasets", "peft", "tokenizers", "accelerate"]
    difficulty: Literal["easy", "medium", "hard"]
