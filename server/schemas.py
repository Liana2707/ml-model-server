from typing import List, Dict, Any, Optional

from pydantic import BaseModel

class FitRequest(BaseModel):
    model_name: str
    model_type: str
    X: List[List[float]]  
    y: List[float]
    config: Dict[str, Any]

class PredictRequest(BaseModel):
    X: List[List[float]]
    model_name: str

class LoadRequest(BaseModel):
    model_name: str

class UnloadRequest(BaseModel):
    model_name: str

class RemoveRequest(BaseModel):
        model_name: str


class Response(BaseModel):
    message: str
    data: Optional[Any] = None


class ErrorResponse(BaseModel):
    message: str

