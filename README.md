# Advertising Sales Prediction using Linear Regression

Individual assignment - Machine Learning Application (CO5).

## Files

| File | Purpose |
|---|---|
| `Advertising_Budget_and_Sales.csv` | The dataset (200 markets: TV, Radio, Newspaper budgets and Sales) |
| `train_model.py` | Preprocessing, model comparison, 5-fold CV, evaluation, saves plots and the model |
| `desktop_app.py` | Desktop application (Tkinter) |
| `predictor.py` | Prediction engine used by the app (standard library only) |
| `build_report.py` | Rebuilds `Assignment2_Report.docx` from the saved results |
| `plots/` | The 7 graphs produced by `train_model.py` |
| `model.json`, `sales_model.joblib`, `results.json`, `model_comparison.csv` | Outputs of `train_model.py` |
| `Assignment2_Report.docx` | The report (A4, 5 pages) |

## Run (Python 3.9+)

```
pip install -r requirements.txt
python train_model.py        # trains, evaluates, writes plots/ and model.json
streamlit run app.py         # opens the Streamlit web app
python build_report.py       # optional: regenerates the report
```

The project also includes a compatibility wrapper at `desktop_app.py`, which launches the same Streamlit app if you run it directly.

Run all commands from this folder; the app looks for `model.json`, the CSV and `plots/` next to itself.

## Optional: single-file Windows .exe (not tested)

```
pip install pyinstaller
pyinstaller --onefile --windowed app.py --add-data "model.json;." --add-data "results.json;." --add-data "model_comparison.csv;." --add-data "Advertising_Budget_and_Sales.csv;." --add-data "plots;plots"
```

(On macOS/Linux use `:` instead of `;` in `--add-data`.)
