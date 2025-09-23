"""
Training script for a personality classification model (Introvert/Extrovert).
Now loads processed datasets from data/processed/train.csv and data/processed/test.csv.
"""

import pandas as pd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Optional MLflow logging
try:
    import mlflow  # type: ignore
    _MLFLOW_AVAILABLE = True
except Exception:  # noqa: BLE001
    _MLFLOW_AVAILABLE = False


# --------------------------------------------------------------------------------------
# Data IO
# --------------------------------------------------------------------------------------

def _resolve_repo_root() -> Path:
    """Returns repo root independent of CWD."""
    return Path(__file__).resolve().parents[2]


def _resolve_processed_paths() -> tuple[Path, Path]:
    repo_root = _resolve_repo_root()
    train_path = repo_root / "data" / "processed" / "train.csv"
    test_path = repo_root / "data" / "processed" / "test.csv"
    return train_path, test_path


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads processed train/test CSVs created by code/datasets/prepare.py."""
    train_path, test_path = _resolve_processed_paths()
    print(f"Loading processed train from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Loading processed test from {test_path}...")
    test_df = pd.read_csv(test_path)

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    return train_df, test_df


# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------

def create_preprocessor():
    """Creates a preprocessing pipeline for numeric and categorical features."""
    print("Building preprocessing pipeline...")

    # Define numeric and categorical columns
    num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 
                'Friends_circle_size', 'Post_frequency']
    cat_cols = ['Stage_fear', 'Drained_after_socializing']

    # Pipeline for numeric features
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Pipeline for categorical features
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    # Combine pipelines
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ])

    return preprocess


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------

def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Trains a Random Forest model using a validation split from the train set.

    Returns: (fitted_model, test_ids, test_pred, test_proba)
    """
    print("Training model...")

    # Create preprocessing
    preprocess = create_preprocessor()

    # Prepare data
    target = "Personality"
    y = train_df[target].map({"Introvert": 0, "Extrovert": 1})

    # Remove id column from features
    X_full = train_df.drop(columns=[target, 'id'])
    X_test = test_df.drop(columns=['id']) if 'id' in test_df.columns else test_df.copy()
    test_ids = test_df['id'] if 'id' in test_df.columns else pd.Series(range(len(test_df)))

    # Split train into train/validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_full, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}")
    print(f"Test features shape: {X_test.shape}")

    # Create model
    rf_clf = Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # Train on train split
    rf_clf.fit(X_train, y_train)

    # Evaluate on validation
    rf_pred = rf_clf.predict(X_valid)
    rf_proba = rf_clf.predict_proba(X_valid)[:, 1]

    acc = accuracy_score(y_valid, rf_pred)
    roc = roc_auc_score(y_valid, rf_proba)

    print("\nValidation metrics (RandomForest):")
    print(classification_report(y_valid, rf_pred))
    print("Accuracy:", round(acc, 4))
    print("ROC-AUC:", round(roc, 4))

    # Train final model on all train data
    best_model = rf_clf
    best_model.fit(X_full, y)

    # Predict on test features (no labels expected in processed test)
    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    return best_model, test_ids, test_pred, test_proba


# --------------------------------------------------------------------------------------
# Model persistence
# --------------------------------------------------------------------------------------

def _resolve_root_models_path() -> Path:
    """Returns path to models/model.pkl at the repository root, independent of CWD."""
    repo_root = _resolve_repo_root()
    return repo_root / "models" / "model.pkl"


def save_model(model, model_path: str | None = None):
    """Saves the trained model into the root models/model.pkl path."""
    resolved_path = Path(model_path) if model_path else _resolve_root_models_path()
    print(f"Saving model to {resolved_path}...")

    # Create models directory if missing
    resolved_path.parent.mkdir(exist_ok=True)

    # Save model
    with open(resolved_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to {resolved_path}")


# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------

def main():
    """Main entrypoint."""
    print("=== Personality classification training ===\n")

    # Load processed data
    train_df, test_df = load_processed_data()

    # Train model
    model, test_ids, test_pred, test_proba = train_model(train_df, test_df)

    # Save model
    save_model(model)

    print("\n=== Training complete! ===")


if __name__ == "__main__":
    main()
