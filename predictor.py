"""
Prediction engine for the desktop app (standard library only - no scikit-learn needed at run time).

train_model.py exports each fitted model to model.json as plain numbers:
    features -> [polynomial terms] -> (x - mean) / scale -> intercept + sum(coef * z)
which is exactly what scikit-learn's Pipeline(PolynomialFeatures -> StandardScaler -> LinearRegression)
computes inside .predict().
"""
import json
import math
import sys
from pathlib import Path


def resource_dir() -> Path:
    """Folder that holds model.json, the CSV and plots (works for a normal run and for PyInstaller)."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


class InputError(ValueError):
    """Raised with a user-friendly message when an input box holds an invalid value."""


class SalesPredictor:
    def __init__(self, model_path):
        with open(model_path, encoding="utf-8") as f:
            data = json.load(f)
        self.features = data["features"]
        self.labels = data["labels"]            # e.g. {"TV": "TV Ad Budget ($)", ..., "Sales": "Sales ($)"}
        self.final = data["final"]
        self.baseline = data["baseline"]
        self.models = data["models"]
        self.ranges = data["ranges"]
        self.means = data["means"]

    # ------------------------------------------------------------------ inputs
    def parse_inputs(self, raw: dict) -> dict:
        """Convert the text typed in the boxes to numbers; raise InputError with a readable message."""
        values = {}
        for feat in self.features:
            label = self.labels[feat]
            text = str(raw.get(feat, "")).strip()
            if text == "":
                raise InputError(f"Please enter a value for {label}.")
            try:
                v = float(text)
            except ValueError:
                raise InputError(f"'{text}' is not a number (field: {label}).") from None
            if not math.isfinite(v):
                raise InputError(f"{label} must be a finite number.")
            if v < 0:
                raise InputError(f"{label} cannot be negative.")
            values[feat] = v
        return values

    def range_warnings(self, values: dict) -> list:
        """Warn when an input lies outside the budgets seen in training (extrapolation is unreliable)."""
        out = []
        for feat in self.features:
            lo, hi = self.ranges[feat]
            if values[feat] < lo or values[feat] > hi:
                out.append(f"{self.labels[feat]} = {values[feat]:g} is outside the training range "
                           f"({lo:g} to {hi:g}); the prediction may be unreliable.")
        return out

    # ------------------------------------------------------------------ prediction
    def predict_raw(self, model_name: str, values: dict) -> float:
        m = self.models[model_name]
        x = [values[f] for f in self.features]
        total = m["intercept"]
        for powers, mean, scale, coef in zip(m["powers"], m["mean"], m["scale"], m["coef"]):
            term = 1.0
            for xi, p in zip(x, powers):
                term *= xi ** p
            total += coef * (term - mean) / scale
        return total

    def predict(self, model_name: str, values: dict):
        """Return (sales, was_clipped). Sales cannot be negative, so negative outputs are shown as 0."""
        y = self.predict_raw(model_name, values)
        return (0.0, True) if y < 0 else (y, False)

    def test_rmse(self, model_name: str) -> float:
        return self.models[model_name]["test"]["RMSE"]

    def test_r2(self, model_name: str) -> float:
        return self.models[model_name]["test"]["R2"]

    @property
    def model_names(self):
        return list(self.models)
