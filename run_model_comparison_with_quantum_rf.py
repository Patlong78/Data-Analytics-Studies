"""
Run model comparison against the current primary dataset.
Supports the original GPED plant-level data and the newer carbon-budget CSV.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent


def select_dataset_path(root: Path) -> Path:
    dataset_path = os.getenv("DATASET_PATH")
    if dataset_path:
        candidate = Path(dataset_path).expanduser().resolve()
        if candidate.exists():
            return candidate

    output_names = {
        "model_comparison_with_quantum_rf.csv",
        "quantum_rf_classification_report.csv",
        "quantum_rf_results.csv",
        "all_models_metrics_table.csv",
    }
    candidates = [
        p for p in root.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".xlsx", ".xls", ".csv"}
        and p.name not in output_names
    ]
    if not candidates:
        raise SystemExit("Dataset not found; place a raw .xlsx/.xls/.csv source file in the workspace root or set DATASET_PATH.")

    def score(path: Path) -> tuple[int, float]:
        name = path.name.lower()
        preference = 0
        if "global_power_emissions" in name or "gped" in name:
            preference += 30
        if "carbon" in name and "budget" in name:
            preference += 25
        if "annual" in name and "co2" in name:
            preference += 20
        if "emissions" in name:
            preference += 10
        if path.name.endswith(".csv"):
            preference += 5
        return (preference, path.stat().st_mtime)

    return max(candidates, key=score)


def read_primary_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        excel_file = pd.ExcelFile(path)
        sheet_names = excel_file.sheet_names
        sheet_name = "GPED_v1.0_Plant Level" if "GPED_v1.0_Plant Level" in sheet_names else sheet_names[0]
        df = pd.read_excel(path, sheet_name=sheet_name)
        if df.iloc[0].isnull().sum() < len(df.columns):
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
        return df

    return pd.read_csv(path)


def specificity_score(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn = cm.sum() - cm.sum(axis=0) - cm.sum(axis=1) + np.diag(cm)
    fp = cm.sum(axis=0) - np.diag(cm)
    class_specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0)
    support = cm.sum(axis=1)
    return np.sum(class_specificity * support) / support.sum() if support.sum() != 0 else 0.0


def build_classification_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "Fuel Types" not in df.columns:
        raise ValueError("Classification dataset is missing 'Fuel Types'.")

    df = df[~df["Fuel Types"].isin(["BIOMASS", "OTHER"])].copy()
    feature_cols = [
        "CO2 Emissions (Mg)", "SO2 Emissions (Mg)", "PM2.5 Emissions (Mg)",
        "NOx Emissions (Mg)", "Total Plant Installed Capacity (MW)"
    ]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").dropna()
    y = df.loc[X.index, "Fuel Types"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(kernel="rbf", probability=False, random_state=42),
        "Naive Bayes": GaussianNB(),
    }

    results = []
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.time() - start
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Specificity": specificity_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Sensitivity": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "Time (sec)": elapsed,
        })

    start = time.time()
    poly = PolynomialFeatures(degree=2, include_bias=False)
    scaler = StandardScaler()
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    X_poly_train_scaled = scaler.fit_transform(X_poly_train)
    X_poly_test_scaled = scaler.transform(X_poly_test)
    qr_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    qr_model.fit(X_poly_train_scaled, y_train)
    y_pred_qr = qr_model.predict(X_poly_test_scaled)
    elapsed = time.time() - start
    results.append({
        "Model": "Quantum Random Forest (poly+RF)",
        "Accuracy": accuracy_score(y_test, y_pred_qr),
        "Specificity": specificity_score(y_test, y_pred_qr),
        "Precision": precision_score(y_test, y_pred_qr, average="weighted", zero_division=0),
        "Sensitivity": recall_score(y_test, y_pred_qr, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, y_pred_qr, average="weighted", zero_division=0),
        "Time (sec)": elapsed,
    })

    results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    report = classification_report(y_test, y_pred_qr, zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(ROOT / "quantum_rf_classification_report.csv")
    return results_df


def build_regression_summary(df: pd.DataFrame) -> pd.DataFrame:
    target_candidates = ["Annual CO₂ emissions", "Annual CO2 emissions", "Annual CO2 Emissions"]
    target_name = next((c for c in target_candidates if c in df.columns), None)
    if target_name is None:
        raise ValueError("Regression dataset is missing a CO2 emissions target column.")

    df = df.copy()
    df[target_name] = pd.to_numeric(df[target_name], errors="coerce")
    feature_frames = []

    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        feature_frames.append(df[["Year"]])

    for col in ["Entity", "Country", "Country or region"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col.replace(" ", "_").replace("/", "_"))
            feature_frames.append(dummies)

    X = pd.concat(feature_frames, axis=1, sort=False).fillna(0)
    y = df[target_name]
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(kernel="rbf"),
    }

    results = []
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.time() - start
        results.append({
            "Model": name,
            "R²": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "Time (sec)": elapsed,
        })

    return pd.DataFrame(results).sort_values("R²", ascending=False).reset_index(drop=True)


def main() -> None:
    dataset_path = select_dataset_path(ROOT)
    df = read_primary_dataset(dataset_path)
    print(f"Using dataset: {dataset_path}")

    if "Fuel Types" in df.columns and any(col in df.columns for col in ["CO2 Emissions (Mg)", "SO2 Emissions (Mg)"]):
        results_df = build_classification_summary(df)
        out_path = ROOT / "model_comparison_with_quantum_rf.csv"
        results_df.to_csv(out_path, index=False)
        print(results_df.to_string(index=False))
        print(f"\nSaved classification summary to {out_path}")
    else:
        results_df = build_regression_summary(df)
        out_path = ROOT / "model_comparison_with_quantum_rf.csv"
        results_df.to_csv(out_path, index=False)
        print(results_df.to_string(index=False))
        print(f"\nSaved regression summary to {out_path}")


if __name__ == "__main__":
    main()
