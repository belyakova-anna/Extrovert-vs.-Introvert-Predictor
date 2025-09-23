"""
Training script for a personality classification model (Introvert/Extrovert).
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
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def load_data(data_path="../datasets/data.csv"):
    """Loads data from a CSV file."""
    print("Loading data...")
    data = pd.read_csv(data_path)
    print(f"Data loaded. Shape: {data.shape}")
    return data


def prepare_data(data):
    """Prepares data for training."""
    print("Preparing data...")
    
    # Split data into train/test with 80/20 ratio
    y_all = data['Personality'].map({"Introvert":0, "Extrovert":1})
    
    train_data, test_data, y_train_split, y_test_split = train_test_split(
        data, y_all, test_size=0.2, stratify=y_all, random_state=42
    )
    
    # Create train and test datasets
    train = train_data.copy()
    test = test_data.copy()
    
    # For test-like features, drop the target column
    test = test.drop(columns=['Personality'])
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print("Class distribution in train:")
    print(train['Personality'].value_counts())
    
    return train, test, y_train_split, y_test_split


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


def train_model(train, test, y_train_split, y_test_split):
    """Trains a Random Forest model."""
    print("Training model...")
    
    # Create preprocessing
    preprocess = create_preprocessor()
    
    # Prepare data
    target = "Personality"
    y = train[target].map({"Introvert":0, "Extrovert":1})
    
    # Remove id column from features
    X_full = train.drop(columns=[target, 'id'])
    X_test = test.drop(columns=['id'])
    test_ids = test["id"]
    
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
    rf_proba = rf_clf.predict_proba(X_valid)[:,1]
    
    print("\nRandom Forest metrics on validation:")
    print(classification_report(y_valid, rf_pred))
    print("ROC-AUC:", round(roc_auc_score(y_valid, rf_proba), 4))
    
    # Train final model on all train data
    best_model = rf_clf
    best_model.fit(X_full, y)
    
    # Predict on test features
    test_proba = best_model.predict_proba(X_test)[:,1]
    test_pred = (test_proba >= 0.5).astype(int)
    
    # Evaluate on held-out test labels
    y_test_true = y_test_split
    
    print("\nEvaluation on test data:")
    print(classification_report(y_test_true, test_pred))
    print("ROC-AUC on test:", round(roc_auc_score(y_test_true, test_proba), 4))
    
    return best_model, test_ids, test_pred, test_proba


def _resolve_root_models_path() -> Path:
    """Returns path to models/model.pkl at the repository root, independent of CWD."""
    current_file = Path(__file__).resolve()
    # Репозиторий: .../introverts_and_extroverts
    # Этот файл: .../introverts_and_extroverts/code/models/train.py
    # Корень репо = parent(parent(current_file)) -> .../introverts_and_extroverts
    repo_root = current_file.parents[2]
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


def main():
    """Main entrypoint."""
    print("=== Personality classification training ===\n")
    
    # Load data
    data = load_data()
    
    # Prepare data
    train, test, y_train_split, y_test_split = prepare_data(data)
    
    # Train model
    model, test_ids, test_pred, test_proba = train_model(train, test, y_train_split, y_test_split)
    
    # Save model
    save_model(model)
    
    print("\n=== Training complete! ===")


if __name__ == "__main__":
    main()
