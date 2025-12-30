import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from census.config import CAT_FEATURES, ENCODER_PATH, LB_PATH, MODEL_PATH
from census.data.clean_data import clean
from census.data.process_data import process


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.encoder = None
    app.state.lb = None

    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(LB_PATH):
            app.state.model = joblib.load(MODEL_PATH)
            app.state.encoder = joblib.load(ENCODER_PATH)
            app.state.lb = joblib.load(LB_PATH)
        else:
            raise ValueError("Artifacts not found. API will reject inference requests.")

    except Exception as exc:
        # Fail fast on load errors
        raise RuntimeError(f"Failed to load artifacts: {exc}") from exc
    yield

    # Shutdown: optional cleanup
    app.state.model = None
    app.state.encoder = None
    app.state.lb = None


app = FastAPI(
    title="Census Bureau Classification", description="", version="0.1.0", lifespan=lifespan
)


def get_artifacts(request: Request) -> Tuple[Any, Any, Any]:
    model = getattr(request.app.state, "model", None)
    encoder = getattr(request.app.state, "encoder", None)
    lb = getattr(request.app.state, "lb", None)
    if model is None or encoder is None or lb is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    return model, encoder, lb


class InferenceRequest(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "age": 18,
                "workclass": "Private",
                "fnlgt": 123456,
                "education": "None",
                "education-num": 12,
                "marital-status": "Separated",
                "occupation": "Prof-specialty",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 50,
                "capital-loss": 5000,
                "hours-per-week": 0,
                "native-country": "United-States",
            }
        },
    )


@app.get("/")
def root() -> Dict:
    return {"message": "Welcome to the FastAPI model inference service."}


@app.post("/infer")
def infer(
    sample: InferenceRequest, artifacts: Tuple[Any, Any, Any] = Depends(get_artifacts)
) -> Dict:
    """
    Infer trained model via FastAPI Call.
    """
    model, encoder, lb = artifacts
    payload = sample.model_dump(by_alias=True)  # keep hyphenated keys
    df = pd.DataFrame([payload])
    df = clean(df)  # one row
    sample, _, _, _ = process(
        df, categorical_features=CAT_FEATURES, training=False, encoder=encoder, lb=lb
    )

    prediction = model.predict(sample)

    if prediction[0] == 0:
        prediction = ">50K"
    else:
        prediction = "<=50K"

    return {"prediction": prediction}
