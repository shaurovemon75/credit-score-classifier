"""Load the saved model and score raw records (used by the API and the batch script)."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .cleaning import prepare


@lru_cache(maxsize=1)
def load_bundle(path=C.MODEL_PATH) -> dict:
    return joblib.load(path)


def predict_frame(raw: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """raw records -> DataFrame with label, class id and one probability column per class."""
    bundle = bundle or load_bundle()
    X = prepare(raw).reindex(columns=bundle["features"])
    proba = bundle["pipeline"].predict_proba(X)
    ids = proba.argmax(axis=1)
    out = pd.DataFrame(proba, columns=[f"prob_{n}" for n in bundle["labels"]], index=raw.index)
    out.insert(0, "Credit_Score", [bundle["labels"][i] for i in ids])
    out.insert(1, "class_id", ids)
    return out
