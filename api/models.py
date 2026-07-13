"""
Phase 6 — API Pydantic Schemas
Defines the request and response shapes for all FastAPI endpoints.
"""
from pydantic import BaseModel, Field
from typing import Literal


class TranslationRequest(BaseModel):
    """Request body for POST /translate"""
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="The source text to translate",
        examples=["ውሻው ሰውን ነከሰ።"]
    )
    source_lang: Literal["am"] = Field(
        default="am",
        description="Source language code — only Amharic supported"
    )
    target_lang: Literal["en"] = Field(
        default="en",
        description="Target language code — only English supported"
    )


class TranslationResponse(BaseModel):
    """Response body for POST /translate"""
    translation: str = Field(
        ...,
        description="The translated text"
    )
    source_lang: str = Field(
        ...,
        description="Source language code"
    )
    target_lang: str = Field(
        ...,
        description="Target language code"
    )
    compute_time_ms: float = Field(
        ...,
        description="Time taken to translate in milliseconds"
    )


class HealthResponse(BaseModel):
    """Response body for GET /health"""
    status: str = Field(default="ok")
    model_loaded: bool = Field(
        ...,
        description="Whether the translation model is loaded and ready"
    )
    device: str = Field(
        ...,
        description="Device the model is running on (cuda or cpu)"
    )


class LanguagesResponse(BaseModel):
    """Response body for GET /languages"""
    supported_pairs: list = Field(
        default=[{"source": "am", "target": "en"}],
        description="List of supported language translation pairs"
    )
    source_languages: list = Field(
        default=["am"],
        description="Available source languages"
    )
    target_languages: list = Field(
        default=["en"],
        description="Available target languages"
    )
