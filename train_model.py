from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "loan_prediction.csv"
MODEL_PATH = ROOT / "models" / "loan_model.joblib"

FEATURES = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
    "Credit_History", "Property_Area"
]
TARGET = "Loan_Status"
CATEGORICAL = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUMERICAL = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

def build_pipeline():
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    numerical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])
    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipe, CATEGORICAL),
        ("numerical", numerical_pipe, NUMERICAL)
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ))
    ])

def train_and_evaluate():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    data = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))

    data = data[required].copy()
    data[TARGET] = data[TARGET].astype(str).str.strip().str.upper()
    data = data[data[TARGET].isin(["Y", "N"])]
    if data.empty:
        raise ValueError("No valid Y/N values found in Loan_Status.")

    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions, labels=["N", "Y"])
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "confusion_matrix": cm.tolist(),
        "test_samples": int(len(y_test)),
        "train_samples": int(len(y_train)),
        "classification_report": classification_report(
            y_test, predictions, labels=["N", "Y"], zero_division=0
        )
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": model, "metrics": metrics}, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print("Confusion matrix (labels N, Y):")
    print(cm)
    print(metrics["classification_report"])
    return model, metrics

def load_or_train_model():
    if MODEL_PATH.exists():
        saved = joblib.load(MODEL_PATH)
        return saved["model"], saved["metrics"]
    return train_and_evaluate()

if __name__ == "__main__":
    train_and_evaluate()
