# Running this project in VS Code

1. **Open**: `File > Open Folder...` and pick the project folder (or open `credit-score-classifier.code-workspace`).
   Click **Install** when VS Code offers the recommended extensions (Python, Jupyter, REST Client, Docker, GitHub).
2. **Environment**: open a terminal (`Ctrl+``) and run
   ```
   python -m venv venv
   venv\Scripts\activate          # Windows;  macOS/Linux: source venv/bin/activate
   pip install -r requirements-dev.txt
   ```
   Then `Ctrl+Shift+P > Python: Select Interpreter` and choose the `venv` one.
3. **Notebook**: open `notebooks/01_credit_score_analysis.ipynb`, pick the `venv` kernel, `Run All`.
4. **Run the API**: `Run and Debug` panel (`Ctrl+Shift+D`) > **1. Run API (FastAPI)** > green play button.
   Open http://127.0.0.1:8000/docs.
5. **Test the API connection**
   - open `api_requests.http` and click **Send Request** above any request (REST Client), or
   - run **2. Call API demo client** from Run and Debug.
6. **Train again** (optional): **3. Train model**. **Tests**: **4. Run tests** or the Testing sidebar.
7. **Git / GitHub**: Source Control panel (`Ctrl+Shift+G`) > *Initialize Repository* > commit > *Publish to GitHub*.
   `Terminal > Run Task...` also lists Install dependencies, Train model, Start API, Run tests.
