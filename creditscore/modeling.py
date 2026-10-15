"""Model definitions, evaluation helpers and the final training routine."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config as C
from .cleaning import feature_columns, prepare


# ---------------------------------------------------------------- data helpers
def load_xy(path=C.TRAIN_PATH):
    raw = pd.read_csv(path, low_memory=False)
    X = prepare(raw)
    y = raw[C.TARGET].map(C.LABEL_TO_ID).to_numpy()
    groups = raw[C.GROUP_COL].to_numpy()
    return X, y, groups


def grouped_split(X, y, groups, test_size=0.2):
    """Hold-out split by customer: the same person never lands in both parts."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=C.RANDOM_STATE)
    tr, va = next(gss.split(X, y, groups))
    return tr, va


# ---------------------------------------------------------------- pipelines
def _numeric_cols():
    return [c for c in feature_columns() if c not in C.CATEGORICAL]


def preprocessor(kind: str = "ordinal") -> ColumnTransformer:
    num = _numeric_cols()
    if kind == "onehot":                           # for linear models
        num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("oh", OneHotEncoder(handle_unknown="ignore"))])
    else:                                          # for tree models
        num_pipe = SimpleImputer(strategy="median")
        cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                             ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                    unknown_value=-1))])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, C.CATEGORICAL)])


LGB_DEFAULT = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=30,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                   class_weight="balanced", random_state=C.RANDOM_STATE, n_jobs=-1, verbose=-1)


def candidate_models() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("prep", preprocessor("onehot")),
            ("clf", LogisticRegression(C=0.5, max_iter=500, class_weight="balanced"))]),
        "Random Forest": Pipeline([
            ("prep", preprocessor()),
            ("clf", RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=-1,
                                           class_weight="balanced", random_state=C.RANDOM_STATE))]),
        "HistGradientBoosting": Pipeline([
            ("prep", preprocessor()),
            ("clf", HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08,
                                                   class_weight="balanced",
                                                   random_state=C.RANDOM_STATE))]),
        "LightGBM": Pipeline([("prep", preprocessor()), ("clf", lgb.LGBMClassifier(**LGB_DEFAULT))]),
    }


def lightgbm_pipeline(**params) -> Pipeline:
    return Pipeline([("prep", preprocessor()),
                     ("clf", lgb.LGBMClassifier(**{**LGB_DEFAULT, **params}))])


# ---------------------------------------------------------------- evaluation
def score(model, X, y) -> dict:
    pred = model.predict(X)
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "weighted_f1": float(f1_score(y, pred, average="weighted"))}


def compare_models(X, y, groups) -> pd.DataFrame:
    tr, va = grouped_split(X, y, groups)
    rows = []
    for name, model in candidate_models().items():
        t = time.time()
        model.fit(X.iloc[tr], y[tr])
        s = score(model, X.iloc[va], y[va])
        s.update(model=name, seconds=round(time.time() - t, 1))
        rows.append(s)
        print(f"{name:22s} acc={s['accuracy']:.4f}  macroF1={s['macro_f1']:.4f}  ({s['seconds']}s)")
    return pd.DataFrame(rows).set_index("model").sort_values("macro_f1", ascending=False)


TUNING_GRID = [
    dict(num_leaves=31, learning_rate=0.05, n_estimators=400, min_child_samples=50),
    dict(num_leaves=63, learning_rate=0.05, n_estimators=400, min_child_samples=30),
    dict(num_leaves=127, learning_rate=0.03, n_estimators=600, min_child_samples=30),
    dict(num_leaves=63, learning_rate=0.08, n_estimators=300, min_child_samples=100, reg_lambda=5.0),
]


def tune_lightgbm(X, y, groups) -> tuple[dict, pd.DataFrame]:
    tr, va = grouped_split(X, y, groups)
    rows = []
    for params in TUNING_GRID:
        m = lightgbm_pipeline(**params).fit(X.iloc[tr], y[tr])
        s = score(m, X.iloc[va], y[va])
        rows.append({**params, **s})
        print(params, f"-> macroF1={s['macro_f1']:.4f}")
    res = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    best = {k: res.loc[0, k] for k in TUNING_GRID[0] | {"reg_lambda": 1.0} if k in res.columns}
    best = {k: (None if pd.isna(v) else v) for k, v in best.items()}
    best = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, np.floating) else v)
            for k, v in best.items() if v is not None}
    return best, res


# ---------------------------------------------------------------- final fit
def holdout_report(model, X, y, groups) -> dict:
    tr, va = grouped_split(X, y, groups)
    model.fit(X.iloc[tr], y[tr])
    pred = model.predict(X.iloc[va])
    return {
        **score(model, X.iloc[va], y[va]),
        "confusion_matrix": confusion_matrix(y[va], pred).tolist(),
        "classification_report": classification_report(y[va], pred, target_names=C.LABELS,
                                                       output_dict=True),
    }


def save_bundle(model: Pipeline, metrics: dict, params: dict) -> None:
    C.MODEL_DIR.mkdir(exist_ok=True)
    bundle = {
        "pipeline": model,
        "features": feature_columns(),
        "labels": C.LABELS,
        "params": params,
        "metrics": {k: metrics[k] for k in ("accuracy", "macro_f1", "weighted_f1")},
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, C.MODEL_PATH, compress=3)


def save_metrics(payload: dict) -> None:
    C.REPORT_DIR.mkdir(exist_ok=True)
    C.METRICS_PATH.write_text(json.dumps(payload, indent=2))
