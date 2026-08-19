"""
Run model comparison matching 03 - Read Dataset.ipynb and include Quantum RF.
Saves `model_comparison_with_quantum_rf.csv` in the workspace.
"""
import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).parent
xls_path = ROOT / "Global_Power_Emissions_Database_v1.0.xlsx"
if not xls_path.exists():
    raise SystemExit("Dataset not found; place Global_Power_Emissions_Database_v1.0.xlsx in the folder")

# Read data similar to notebook
df = pd.read_excel(xls_path, sheet_name="GPED_v1.0_Plant Level")
if df.iloc[0].isnull().sum() < len(df.columns):
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)

# remove some fuel types as in the notebook
if 'Fuel Types' in df.columns:
    df = df[~df['Fuel Types'].isin(['BIOMASS', 'OTHER'])]

# Prepare features exactly as Block 54
X = df[["CO2 Emissions (Mg)", "SO2 Emissions (Mg)", "PM2.5 Emissions (Mg)", 
        "NOx Emissions (Mg)", "Total Plant Installed Capacity (MW)"]].copy()
X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna()

y = df.loc[X.index, 'Fuel Types']

# Train-test split (stratify by y as in notebook)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# helper specificity
def specificity_score(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn = cm.sum() - cm.sum(axis=0) - cm.sum(axis=1) + np.diag(cm)
    fp = cm.sum(axis=0) - np.diag(cm)
    class_specificity = np.divide(tn, tn + fp, out=np.zeros_like(tn, dtype=float), where=(tn + fp) != 0)
    support = cm.sum(axis=1)
    return np.sum(class_specificity * support) / support.sum() if support.sum() != 0 else 0.0

# baseline models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Support Vector Machine': SVC(kernel='rbf', probability=False, random_state=42),
    'Naive Bayes': GaussianNB()
}

results = []
for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    end = time.time()

    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Specificity': specificity_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Sensitivity': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'Time (sec)': end - start
    })

# Quantum-inspired RF: polynomial features + RF
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
end = time.time()

results.append({
    'Model': 'Quantum Random Forest (poly+RF)',
    'Accuracy': accuracy_score(y_test, y_pred_qr),
    'Specificity': specificity_score(y_test, y_pred_qr),
    'Precision': precision_score(y_test, y_pred_qr, average='weighted', zero_division=0),
    'Sensitivity': recall_score(y_test, y_pred_qr, average='weighted', zero_division=0),
    'F1 Score': f1_score(y_test, y_pred_qr, average='weighted', zero_division=0),
    'Time (sec)': end - start
})

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False).reset_index(drop=True)
print(results_df.to_string(index=False))

# Save
out_path = ROOT / 'model_comparison_with_quantum_rf.csv'
results_df.to_csv(out_path, index=False)
print('\nSaved results to', out_path)

# Save classification report for QR model
report = classification_report(y_test, y_pred_qr, zero_division=0, output_dict=True)
pd.DataFrame(report).T.to_csv(ROOT / 'quantum_rf_classification_report.csv')
print('Saved classification report to quantum_rf_classification_report.csv')
