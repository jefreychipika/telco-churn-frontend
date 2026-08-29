"""
train_model.py
----------------
Rebuilds the churn model from the original notebook, but as a clean,
repeatable script instead of ad-hoc notebook cells.

Key differences from the original notebook version:
1. Instead of manually one-hot encoding columns and then trying to line
   up column names by hand inside the Streamlit app (which was fragile
   and only handled 3 of the ~19 input fields), we build a single
   scikit-learn Pipeline (preprocessing + model). The pipeline is saved
   as ONE object, so the FastAPI backend just calls `pipeline.predict()`
   on a raw dataframe with the original column names - no manual
   dummy-variable bookkeeping required.
2. Both Logistic Regression and Random Forest are trained and compared,
   exactly like the notebook did, and the better performing model
   (by F1-score on the churn class) is the one saved for production use.
3. Feature importances / coefficients are exported to a CSV so the
   Streamlit dashboard can display "what drives churn" without
   re-running training.

Run this script once (or whenever the CSV changes) to (re)generate:
    - churn_pipeline.pkl      -> the full trained pipeline (used by the API)
    - feature_importance.csv  -> top drivers of churn, for the dashboard
    - metrics.json            -> evaluation metrics, for the dashboard
"""

import json
import warnings

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. CONFIG - column groups
# ---------------------------------------------------------------------------
# These lists describe how each raw column from the CSV should be treated.
# Keeping them here (instead of hard-coded deep in the pipeline) makes it
# easy to see, at a glance, exactly what the model expects as input.

CSV_PATH = "telco_customer_churn.csv"

TARGET_COL = "Churn"
ID_COL = "customerID"

# Already numeric, used as-is
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

# Yes/No (or Male/Female) columns -> encoded as 0/1
BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

# Columns with 3+ categories -> one-hot encoded
CATEGORICAL_COLS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV and apply the same cleaning steps as the notebook.

    - TotalCharges arrives as text (a handful of blank strings for brand
      new customers), so we coerce it to numeric and drop the few rows
      that can't be converted.
    - customerID is dropped because it's a unique identifier, not a
      predictive feature.
    - gender / binary Yes-No columns are mapped to 0/1 here so that the
      downstream ColumnTransformer only has to deal with true multi-class
      categoricals.
    """
    df = pd.read_csv(csv_path)

    # Fix TotalCharges: blank strings -> NaN -> drop those rows
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Drop the identifier column - not useful for prediction
    df = df.drop(columns=[ID_COL])

    # Encode the target: Yes/No -> 1/0
    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    # Encode gender and the simple Yes/No columns
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df[col] = df[col].map({"Yes": 1, "No": 0})

    return df


def build_pipeline(model) -> Pipeline:
    """Wrap a classifier in a full preprocessing pipeline.

    The ColumnTransformer only needs to one-hot encode the multi-category
    columns; the numeric and already-binary-encoded columns pass through
    unchanged ('remainder=passthrough'). Because this lives inside a
    Pipeline, the exact same transformation is applied automatically at
    prediction time - callers just pass a raw dataframe with the original
    column names/values (e.g. Contract='Month-to-month').
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL_COLS,
            )
        ],
        remainder="passthrough",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def main():
    print("Loading and cleaning data...")
    df = load_and_clean_data(CSV_PATH)
    print(f"  -> {df.shape[0]} rows, {df.shape[1]} columns after cleaning")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------------
    # 2. Train both candidate models, exactly like the notebook explored
    # ------------------------------------------------------------------
    candidates = {
        "logistic_regression": build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=42
            )
        ),
    }

    results = {}
    for name, pipeline in candidates.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1_churn = f1_score(y_test, y_pred)  # F1 on the "Yes churn" class
        print(classification_report(y_test, y_pred))
        results[name] = {
            "pipeline": pipeline,
            "report": report,
            "f1_churn": f1_churn,
        }

    # ------------------------------------------------------------------
    # 3. Pick the winner (best F1-score on the churn class) and save it
    # ------------------------------------------------------------------
    best_name = max(results, key=lambda n: results[n]["f1_churn"])
    best_pipeline = results[best_name]["pipeline"]
    print(f"\nBest model: {best_name} (F1 on churn class = {results[best_name]['f1_churn']:.3f})")

    joblib.dump(best_pipeline, "churn_pipeline.pkl")
    print("Saved trained pipeline -> churn_pipeline.pkl")

    # Save a small metrics summary the dashboard can show to the user
    metrics_summary = {
        "best_model": best_name,
        "logistic_regression": results["logistic_regression"]["report"],
        "random_forest": results["random_forest"]["report"],
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print("Saved evaluation metrics -> metrics.json")

    # ------------------------------------------------------------------
    # 4. Feature importance / coefficients, for the "why" behind churn
    # ------------------------------------------------------------------
    feature_names = best_pipeline.named_steps["preprocess"].get_feature_names_out()
    model_step = best_pipeline.named_steps["model"]

    if hasattr(model_step, "feature_importances_"):
        importance_values = model_step.feature_importances_
    else:
        importance_values = model_step.coef_[0]

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance_values}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv("feature_importance.csv", index=False)
    print("Saved feature importance -> feature_importance.csv")


if __name__ == "__main__":
    main()
