from typing import List, Optional
from pydantic import BaseModel

class Symptoms(BaseModel):
    description: str
    duration: Optional[str]
    severity: Optional[str]

class PatientTextData(BaseModel):
    age: Optional[int] = None
    symptoms_description: str
    medical_history: Optional[str] = None
    current_medications: Optional[List[str]] = None
    lifestyle_factors: Optional[str] = None

class ImageAnalysisResult(BaseModel):
    detected_conditions: List[str]
    confidence_scores: List[float]
    visual_markers: List[str]