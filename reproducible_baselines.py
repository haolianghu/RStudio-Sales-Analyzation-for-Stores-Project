"""Reproduce sales-forecast and customer-classification baselines."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _sales_design(frame, origin):
    quarter = frame["quarter"].astype("string").str.strip().str.upper()
    if not quarter.isin({"Q1", "Q2", "Q3", "Q4"}).all():
        raise ValueError("Quarter values must be Q1, Q2, Q3, or Q4.")
    time = pd.to_numeric(frame["year"], errors="raise").to_numpy(float)
    if not np.isfinite(time).all():
        raise ValueError("Years must be finite.")
    centered = time - origin
    result = pd.DataFrame({
        "time": centered,
        "time_squared": centered**2,
    }, index=frame.index)
    for label in ("Q2", "Q3", "Q4"):
        result[f"quarter_{label}"] = (quarter == label).astype(int).to_numpy()
    return result


def sales_forecast(frame, holdout_quarters=8):
    required = {"sales", "quarter", "year"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Sales data must contain: {', '.join(sorted(required))}.")
    if isinstance(holdout_quarters, bool) or not isinstance(holdout_quarters, int) or holdout_quarters < 4:
        raise ValueError("Holdout must be an integer of at least four quarters.")
    data = frame.copy().reset_index(drop=True)
    data["sales"] = pd.to_numeric(data["sales"], errors="raise")
    if data["sales"].isna().any() or not np.isfinite(data["sales"].to_numpy(float)).all() or (data["sales"] <= 0).any():
        raise ValueError("Sales must be finite and positive for the log-linear model.")
    if len(data) <= holdout_quarters + 8:
        raise ValueError("Insufficient history for the requested holdout.")
    quarter = data["quarter"].astype("string").str.strip().str.upper()
    expected_year = np.floor(pd.to_numeric(data["year"], errors="raise")) + (
        quarter.str[-1].astype(int).to_numpy() - 1
    ) / 4
    if not np.allclose(data["year"].to_numpy(float), expected_year, atol=1e-8, rtol=0):
        raise ValueError("Year must encode quarters as year + (quarter - 1) / 4.")
    if not np.allclose(np.diff(data["year"].to_numpy(float)), 0.25, atol=1e-8, rtol=0):
        raise ValueError("Sales rows must be consecutive quarterly observations.")
    train, test = data.iloc[:-holdout_quarters], data.iloc[-holdout_quarters:]
    origin = float(train["year"].min())
    model = LinearRegression().fit(_sales_design(train, origin), np.log(train["sales"]))
    predicted = np.exp(model.predict(_sales_design(test, origin)))

    # Seasonal-naive baseline: same quarter from four quarters earlier.
    actual = test["sales"].to_numpy(float)
    seasonal_naive = data["sales"].iloc[len(train) - 4:len(data) - 4].to_numpy(float)

    def metrics(observed, forecast):
        return {
            "mae": float(mean_absolute_error(observed, forecast)),
            "rmse": float(np.sqrt(mean_squared_error(observed, forecast))),
            "mape_percent": float(np.mean(np.abs((observed - forecast) / observed)) * 100),
        }

    future = pd.DataFrame({"year": [float(data["year"].iloc[-1]) + step / 4 for step in (1, 2, 3, 4)]})
    future["quarter"] = [f"Q{(int(round((year - np.floor(year)) * 4)) % 4) + 1}" for year in future["year"]]
    all_origin = float(data["year"].min())
    all_model = LinearRegression().fit(_sales_design(data, all_origin), np.log(data["sales"]))
    forecast = np.exp(all_model.predict(_sales_design(future, all_origin)))
    return {
        "rows": len(data),
        "holdout_quarters": holdout_quarters,
        "model": metrics(actual, predicted),
        "seasonal_naive": metrics(actual, seasonal_naive),
        "next_four_quarters": [
            {"year": float(year), "quarter": quarter, "forecast_sales": float(value)}
            for year, quarter, value in zip(future["year"], future["quarter"], forecast)
        ],
    }


def customer_classification(frame, seed=100):
    required = {"freq", "female", "spend", "type"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Purchase data must contain: {', '.join(sorted(required))}.")
    data = frame[["freq", "female", "spend", "type"]].copy()
    data[list(("freq", "female", "spend"))] = data[list(("freq", "female", "spend"))].apply(pd.to_numeric, errors="raise")
    if data.isna().any().any() or not np.isfinite(data[["freq", "female", "spend"]].to_numpy(float)).all():
        raise ValueError("Purchase predictors and labels must be complete and finite.")
    if data["type"].nunique() != 2:
        raise ValueError("Customer type must contain two classes.")
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns="type"), data["type"], test_size=0.2,
        random_state=seed, stratify=data["type"],
    )
    results = {}
    for name, classifier in (
        ("knn", KNeighborsClassifier(n_neighbors=11)),
        ("logistic_regression", LogisticRegression(max_iter=1000, random_state=seed)),
    ):
        model = make_pipeline(StandardScaler(), classifier).fit(X_train, y_train)
        prediction = model.predict(X_test)
        labels = sorted(data["type"].unique())
        results[name] = {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "confusion_matrix": confusion_matrix(y_test, prediction, labels=labels).tolist(),
            "classes": labels,
        }
    return {"rows": len(data), "train_rows": len(X_train), "test_rows": len(X_test), "seed": seed, "models": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sales", type=Path, default=Path("whole_foods.csv"))
    parser.add_argument("--purchases", type=Path, default=Path("purchases_type.csv"))
    args = parser.parse_args()
    try:
        result = {
            "sales": sales_forecast(pd.read_csv(args.sales)),
            "customer_classification": customer_classification(pd.read_csv(args.purchases)),
        }
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
