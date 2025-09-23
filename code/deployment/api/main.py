import pickle
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Prediction Service", version="1.0.0")


class FeatureItem(BaseModel):
    """Explicit, typed features that the model expects (matching training columns)."""

    Time_spent_Alone: float = Field(..., description="Time spent alone (numeric)")
    Social_event_attendance: float = Field(..., description="Social event attendance (numeric)")
    Going_outside: float = Field(..., description="Going outside (numeric)")
    Friends_circle_size: float = Field(..., description="Friends circle size (numeric)")
    Post_frequency: float = Field(..., description="Post frequency (numeric)")
    Stage_fear: str = Field(..., description="Stage fear (categorical, e.g., Yes/No)")
    Drained_after_socializing: str = Field(
        ..., description="Feeling drained after socializing (categorical, e.g., Yes/No)"
    )


class PredictTypedRequest(BaseModel):
    """Preferred request schema: single sample or batch of structured feature items."""

    samples: Union[FeatureItem, List[FeatureItem]] = Field(
        ..., description="Single FeatureItem or list of FeatureItem objects"
    )


class PredictLegacyRequest(BaseModel):
    """Legacy request schema: list[float] or list[list[float]].

    The expected order is:
    [Time_spent_Alone, Social_event_attendance, Going_outside, Friends_circle_size,
     Post_frequency, Stage_fear, Drained_after_socializing]
    """

    features: Union[List[float], List[List[float]]] = Field(
        ..., description="Single sample as list[float] or batch as list[list[float]]"
    )

def _load_model() -> Any:
    # Load strictly from repo root models/model.pkl
    # File location: repo_root/code/deployment/api/main.py
    # Repo root is three levels up from this file
    model_path = Path(__file__).resolve().parents[3] / "models" / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at: {model_path}. Ensure 'models/model.pkl' exists."
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


# Load model at startup
try:
    MODEL = _load_model()
except Exception as exc:  # noqa: BLE001
    # Delay hard failure until first request so the process can start for diagnostics
    MODEL = exc  # type: ignore[assignment]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(
    payload: Union[PredictTypedRequest, PredictLegacyRequest] = Body(...),
) -> Dict[str, Any]:
    # Validate model is loaded
    if isinstance(MODEL, Exception):
        raise HTTPException(status_code=500, detail=f"Model load error: {MODEL}")

    # Build DataFrame depending on payload schema
    columns = [
        "Time_spent_Alone",
        "Social_event_attendance",
        "Going_outside",
        "Friends_circle_size",
        "Post_frequency",
        "Stage_fear",
        "Drained_after_socializing",
    ]

    if isinstance(payload, PredictTypedRequest):
        if isinstance(payload.samples, list):
            records = [item.dict() for item in payload.samples]
        else:
            records = [payload.samples.dict()]
        X = pd.DataFrame.from_records(records, columns=columns)
    else:
        features = payload.features
        if features and isinstance(features[0], (int, float)):
            batch: List[List[Any]] = [features]  # type: ignore[list-item]
        else:
            batch = features  # type: ignore[assignment]
        X = pd.DataFrame(batch, columns=columns)

    # Run prediction; support common APIs (.predict or callable)
    try:
        if hasattr(MODEL, "predict") and callable(getattr(MODEL, "predict")):
            preds = MODEL.predict(X)
        elif callable(MODEL):
            preds = MODEL(X)
        else:
            raise TypeError("Loaded model does not support prediction via .predict or __call__")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    # Convert to plain list
    if hasattr(preds, "tolist"):
        preds = preds.tolist()

    return {"predictions": preds}


if __name__ == "__main__":
    # Run with: python code/deployment/api/main.py
    # Or: uvicorn code.deployment.api.main:app --host 0.0.0.0 --port 8000
    import uvicorn

    uvicorn.run("code.deployment.api.main:app", host="0.0.0.0", port=8000, reload=False)


