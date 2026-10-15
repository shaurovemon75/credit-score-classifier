"""Demo client: shows how to talk to the running API.

    python scripts/call_api.py                       # uses http://127.0.0.1:8000
    python scripts/call_api.py --url https://your-host
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.schemas import EXAMPLE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000")
args = ap.parse_args()
base = args.url.rstrip("/")


def show(title, resp):
    print(f"\n### {title}  [HTTP {resp.status_code}]")
    print(json.dumps(resp.json(), indent=2)[:900])


show("GET /health", requests.get(f"{base}/health", timeout=10))
show("GET /model-info", requests.get(f"{base}/model-info", timeout=10))
show("POST /predict (single record)", requests.post(f"{base}/predict", json=EXAMPLE, timeout=30))

# batch: first 3 rows of the real test file, sent as JSON
test = pd.read_csv(Path(__file__).resolve().parents[1] / "data/raw/test.csv.gz", nrows=3, low_memory=False)
fields = list(EXAMPLE)
records = [{k: (None if pd.isna(v) else v) for k, v in row.items()}
           for row in test[fields].to_dict("records")]
show("POST /predict/batch (3 rows from test.csv)",
     requests.post(f"{base}/predict/batch", json={"records": records}, timeout=60))
