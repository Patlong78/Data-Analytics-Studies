"""Standalone version of notebook #13: CO2 analysis and model comparison.

Requires owid-co2-data.csv in the same directory.
Install dependencies with:
    python -m pip install pandas numpy scikit-learn xgboost catboost
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBClassifier


DATA_PATH = Path(__file__).with_name("owid-co2-data.csv")
FEATURE_COLS = [
    "coal_co2",
    "gas_co2",
    "oil_co2",
    "co2_per_capita",
    "co2_per_gdp",
]


def classification_metrics(model_name, y_true, y_pred, elapsed_time):
    """Return the shared classification metrics used by every model."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Sensitivity": sensitivity,
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "Time (sec)": elapsed_time,
    }


def run_model(model_name, model, x_train, x_test, y_train, y_test):
    """Fit a classifier and return its predictions and shared metrics."""
    start_time = time.perf_counter()
    model.fit(x_train, y_train)
    predictions = np.asarray(model.predict(x_test)).ravel()
    elapsed_time = time.perf_counter() - start_time
    metrics = classification_metrics(model_name, y_test, predictions, elapsed_time)
    return predictions, metrics


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["co2"] = pd.to_numeric(df["co2"], errors="coerce")

    if df["co2"].notna().sum() == 0:
        raise ValueError("The dataset contains no usable values in the co2 column.")

    latest_year = int(df.loc[df["co2"].notna(), "year"].max())
    start_year = latest_year - 49
    recent_df = df[
        (df["year"] >= start_year)
        & (df["year"] <= latest_year)
        & df["co2"].notna()
    ].copy()

    print(f"Dataset shape: {df.shape}")
    print(f"Model years: {start_year}-{latest_year}")
    print(f"Model rows before feature cleaning: {len(recent_df)}")

    country_totals = (
        recent_df.groupby("country")["co2"].sum().sort_values(ascending=False)
    )
    top_50 = country_totals.head(50)
    results_df = top_50.rename_axis("Country").reset_index(name="Total_CO2_Mt")
    results_df["Rank"] = range(1, len(results_df) + 1)
    results_df["Pct_of_Global"] = (
        results_df["Total_CO2_Mt"] / country_totals.sum() * 100
    ).round(2)
    results_df["Avg_Annual_CO2_Mt"] = (results_df["Total_CO2_Mt"] / 50).round(2)
    results_df = results_df[
        ["Rank", "Country", "Total_CO2_Mt", "Pct_of_Global", "Avg_Annual_CO2_Mt"]
    ]

    print("\nTop 10 countries by aggregate CO2:")
    print(results_df.head(10).to_string(index=False))

    coal_summary = recent_df.groupby("country").agg(
        total_co2=("co2", "sum"),
        coal_co2=("coal_co2", "sum"),
        gas_co2=("gas_co2", "sum"),
        oil_co2=("oil_co2", "sum"),
    )
    coal_summary["coal_share_of_total_pct"] = (
        coal_summary["coal_co2"] / coal_summary["total_co2"] * 100
    ).round(2)
    print("\nCorrelation of total CO2 with source emissions:")
    print(coal_summary[["total_co2", "coal_co2", "gas_co2", "oil_co2"]].corr().round(3))

    model_df = recent_df[FEATURE_COLS + ["co2"]].copy()
    model_df[FEATURE_COLS] = model_df[FEATURE_COLS].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)
    model_df = model_df.dropna(subset=["co2"]).copy()

    threshold = model_df["co2"].quantile(0.5)
    model_df["co2_high"] = (model_df["co2"] > threshold).astype(int)
    if model_df["co2_high"].nunique() < 2:
        raise ValueError("The CO2 target does not contain two usable classes.")

    x = model_df[FEATURE_COLS]
    y = model_df["co2_high"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_poly)
    x_test_scaled = scaler.transform(x_test_poly)

    models = []
    predictions = {}

    predictions["Quantum AI Random Forest"], metrics = run_model(
        "Quantum AI Random Forest",
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        x_train_scaled,
        x_test_scaled,
        y_train,
        y_test,
    )
    models.append(metrics)

    predictions["Classical Random Forest"], metrics = run_model(
        "Classical Random Forest",
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        x_train,
        x_test,
        y_train,
        y_test,
    )
    models.append(metrics)

    predictions["Gradient Boosting Classifier"], metrics = run_model(
        "Gradient Boosting Classifier",
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
        x_train,
        x_test,
        y_train,
        y_test,
    )
    models.append(metrics)

    predictions["Neural Network (MLP)"], metrics = run_model(
        "Neural Network (MLP)",
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            random_state=42,
        ),
        x_train_scaled,
        x_test_scaled,
        y_train,
        y_test,
    )
    models.append(metrics)

    predictions["Logistic Regression"], metrics = run_model(
        "Logistic Regression",
        LogisticRegression(max_iter=1000, random_state=42),
        x_train_scaled,
        x_test_scaled,
        y_train,
        y_test,
    )
    models.append(metrics)

    predictions["XGBoost"], metrics = run_model(
        "XGBoost",
        XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        x_train,
        x_test,
        y_train,
        y_test,
    )
    models.append(metrics)

    predictions["CatBoost"], metrics = run_model(
        "CatBoost",
        CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            random_seed=42,
            verbose=False,
            thread_count=-1,
        ),
        x_train,
        x_test,
        y_train,
        y_test,
    )
    models.append(metrics)

    comparison_metrics_df = pd.DataFrame(models)
    print("\nFinal model comparison:")
    print(comparison_metrics_df.to_string(index=False))

    best_model = comparison_metrics_df.loc[
        comparison_metrics_df["F1 Score"].idxmax(), "Model"
    ]
    print(f"\nBest model by F1 Score: {best_model}")
    print(f"Target threshold: {threshold:.4f}")
    print(f"Training rows: {len(x_train)} | Test rows: {len(x_test)}")

    print(f"\nClassification report: {best_model}")
    print(
        classification_report(
            y_test,
            predictions[best_model],
            labels=[0, 1],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
