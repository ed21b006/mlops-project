from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    image_base64: Optional[str] = None
    pixel_array: Optional[List[float]] = Field(default=None, min_length=784, max_length=784)

    @field_validator("pixel_array")
    @classmethod
    def check_range(cls, v):
        if v is not None and any(val < 0.0 or val > 1.0 for val in v):
            raise ValueError("Pixel values must be in [0, 1]")
        return v


class PredictionResponse(BaseModel):
    predicted_digit: int
    confidence: float
    probabilities: List[float]
    inference_time_ms: float


class FeedbackRequest(BaseModel):
    correct_label: int = Field(ge=0, le=9)
    predicted_label: int = Field(ge=0, le=9)
    pixel_array: Optional[List[float]] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str
