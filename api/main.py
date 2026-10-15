"""FastAPI service:  uvicorn api.main:app --reload   ->  http://127.0.0.1:8000/docs"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from creditscore import config as C
from creditscore import predict as P
from creditscore.train import main as train_model

from .schemas import BatchRequest, BatchResponse, CreditRecord, Prediction

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("RETRAIN") == "1" or not C.MODEL_PATH.exists():
        train_model()                      # first start without a saved model trains one
    state["bundle"] = P.load_bundle()
    yield
    state.clear()


app = FastAPI(
    title="Smart Credit - Credit Score Classification API",
    description="Predicts whether a customer's credit score is Poor, Standard or Good.",
    version="1.0.0",
    lifespan=lifespan,
)


def _to_predictions(records: list[CreditRecord]) -> list[Prediction]:
    df = pd.DataFrame([r.model_dump() for r in records])
    out = P.predict_frame(df, state["bundle"])
    labels = state["bundle"]["labels"]
    return [
        Prediction(
            credit_score=row["Credit_Score"],
            class_id=int(row["class_id"]),
            probabilities={n: round(float(row[f"prob_{n}"]), 4) for n in labels},
        )
        for _, row in out.iterrows()
    ]


@app.get("/", tags=["info"])
def root():
    return {"service": "credit-score-api", "docs": "/docs", "health": "/health"}


@app.get("/health", tags=["info"])
def health():
    return {"status": "ok", "model_loaded": "bundle" in state}


@app.get("/model-info", tags=["info"])
def model_info():
    b = state["bundle"]
    return {"classes": b["labels"], "n_features": len(b["features"]), "params": b["params"],
            "holdout_metrics": b["metrics"], "trained_at": b["trained_at"],
            "sklearn_version": b["sklearn_version"]}


@app.post("/predict", response_model=Prediction, tags=["prediction"])
def predict(record: CreditRecord):
    try:
        return _to_predictions([record])[0]
    except Exception as exc:               # pragma: no cover
        raise HTTPException(status_code=422, detail=f"Could not score record: {exc}")


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(req: BatchRequest):
    preds = _to_predictions(req.records)
    return BatchResponse(count=len(preds), predictions=preds)
