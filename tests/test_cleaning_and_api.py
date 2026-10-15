import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import EXAMPLE
from creditscore import cleaning as K
from creditscore import config as C


def test_dirty_values_become_nan_or_numbers():
    raw = pd.DataFrame({"Age": ["28_", "-500", "34"], "Annual_Income": ["34847.84_", "_", "100"],
                        "Credit_Mix": ["_", "Good", "Bad"], "Payment_Behaviour": ["!@9#%8", "x", "y"],
                        "Credit_History_Age": ["22 Years and 9 Months", np.nan, "5"]})
    out = K.clean(raw)
    assert out["Age"].iloc[0] == 28 and np.isnan(out["Age"].iloc[1])
    assert out["Annual_Income"].iloc[0] == pytest.approx(34847.84) and np.isnan(out["Annual_Income"].iloc[1])
    assert pd.isna(out["Credit_Mix"].iloc[0]) and pd.isna(out["Payment_Behaviour"].iloc[0])
    assert out["Credit_History_Age"].iloc[0] == 273


def test_customer_level_filling():
    raw = pd.DataFrame({"Customer_ID": ["A", "A", "A"], "Month": ["January", "February", "March"],
                        "Age": [30, -500, 30], "Credit_History_Age": ["10 Years and 0 Months", np.nan,
                                                                       "10 Years and 2 Months"]})
    out = K.clean(raw)
    assert out["Age"].tolist() == [30, 30, 30]
    assert out["Credit_History_Age"].iloc[1] == 121


def test_feature_frame_has_expected_columns():
    X = K.prepare(pd.DataFrame([EXAMPLE]))
    assert list(X.columns) == K.feature_columns()


@pytest.fixture(scope="module")
def client():
    if not C.MODEL_PATH.exists():
        pytest.skip("run `python -m creditscore.train` first")
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_predict_single(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["credit_score"] in C.LABELS
    assert sum(body["probabilities"].values()) == pytest.approx(1, abs=1e-3)


def test_predict_empty_record_and_batch(client):
    assert client.post("/predict", json={}).status_code == 200
    r = client.post("/predict/batch", json={"records": [EXAMPLE, {"Age": 40}]})
    assert r.status_code == 200 and r.json()["count"] == 2


def test_batch_validation(client):
    assert client.post("/predict/batch", json={"records": []}).status_code == 422
