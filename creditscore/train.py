"""End-to-end training:  python -m creditscore.train

1. clean + featurise train.csv
2. compare four classifiers on a customer-grouped hold-out set
3. tune the best family (LightGBM) over a small grid
4. report hold-out metrics, save figures, refit on ALL training rows and save the model
5. score test.csv (no labels) and write reports/test_predictions.csv
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config as C
from . import modeling as M
from .cleaning import prepare
from .plots import plot_confusion, plot_importance, plot_model_comparison
from .predict import predict_frame


def main() -> None:
    t0 = time.time()
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    X, y, groups = M.load_xy()
    print(f"train features: {X.shape}, customers: {len(set(groups))}")

    print("\n== 1. model comparison (grouped 80/20 hold-out) ==")
    comparison = M.compare_models(X, y, groups)
    plot_model_comparison(comparison)

    print("\n== 2. LightGBM tuning ==")
    best_params, tuning = M.tune_lightgbm(X, y, groups)
    print("best:", best_params)

    print("\n== 3. hold-out evaluation of tuned model ==")
    report = M.holdout_report(M.lightgbm_pipeline(**best_params), X, y, groups)
    print({k: round(v, 4) for k, v in report.items() if isinstance(v, float)})
    plot_confusion(np.array(report["confusion_matrix"]))

    print("\n== 4. final fit on full training data ==")
    final = M.lightgbm_pipeline(**best_params).fit(X, y)
    plot_importance(final, X.columns.tolist())
    M.save_bundle(final, report, best_params)

    M.save_metrics({
        "model_comparison": comparison.round(4).reset_index().to_dict("records"),
        "tuning": tuning.round(4).to_dict("records"),
        "best_params": best_params,
        "holdout": report,
    })

    print("\n== 5. scoring test.csv ==")
    test_raw = pd.read_csv(C.TEST_PATH, low_memory=False)
    pred = predict_frame(test_raw)
    out = pd.concat([test_raw[["ID", "Customer_ID", "Month"]], pred], axis=1)
    out.round(4).to_csv(C.SUBMISSION_PATH, index=False)
    print(out["Credit_Score"].value_counts(normalize=True).round(3).to_dict())
    print(f"done in {time.time() - t0:.0f}s -> {C.MODEL_PATH.name}, {C.SUBMISSION_PATH.name}")


if __name__ == "__main__":
    main()
