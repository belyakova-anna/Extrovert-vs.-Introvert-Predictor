"""
Prepare dataset: load raw CSV, clean, split into train/test, save processed files.
Reuses the existing logic from code/models/train.py for consistency.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# Paths resolved relative to repo root
CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parents[2]
RAW_DATA_PATH = REPO_ROOT / "data" / "raw" / "data.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
TRAIN_OUT = PROCESSED_DIR / "train.csv"
TEST_OUT = PROCESSED_DIR / "test.csv"


def load_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    print(f"Loading raw data from {path}...")
    data = pd.read_csv(path)
    print(f"Loaded: {data.shape}")
    return data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove obvious missing values and basic anomalies consistent with training.

    - Drop rows with missing target `Personality`
    - Basic NA handling for features by dropping rows with any NA
    - Remove impossible/negative values for numeric columns if present
    """
    df = df.copy()

    # Ensure target exists for splitting logic
    if "Personality" not in df.columns:
        raise ValueError("Column 'Personality' is required in the raw dataset.")

    # Drop rows with missing target
    df = df.dropna(subset=["Personality"])  # keep rows with defined labels

    # Define feature columns as used in training
    numeric_cols = [
        "Time_spent_Alone",
        "Social_event_attendance",
        "Going_outside",
        "Friends_circle_size",
        "Post_frequency",
    ]
    categorical_cols = [
        "Stage_fear",
        "Drained_after_socializing",
    ]

    # Drop rows with any NA in used columns to keep it simple and consistent
    used_cols = ["id", "Personality"] + numeric_cols + categorical_cols
    existing_used_cols = [c for c in used_cols if c in df.columns]
    df = df.dropna(subset=existing_used_cols)

    # Remove basic anomalies: negative values in numeric columns
    for col in numeric_cols:
        if col in df.columns:
            df = df[df[col] >= 0]

    return df


def split_and_save(df: pd.DataFrame) -> None:
    print("Splitting into train/test (80/20, stratified)...")
    y_all = df["Personality"].map({"Introvert": 0, "Extrovert": 1})

    train_df, test_df, y_train_split, y_test_split = train_test_split(
        df, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    # For the test-like features, align with training by dropping target
    test_df = test_df.copy()
    if "Personality" in test_df.columns:
        test_df = test_df.drop(columns=["Personality"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving train to {TRAIN_OUT} with shape {train_df.shape}...")
    train_df.to_csv(TRAIN_OUT, index=False)

    print(f"Saving test to {TEST_OUT} with shape {test_df.shape}...")
    test_df.to_csv(TEST_OUT, index=False)


def main() -> None:
    print("=== Preparing dataset ===")
    data = load_data()
    data = clean_data(data)
    split_and_save(data)
    print("=== Done. Processed files written to data/processed ===")


if __name__ == "__main__":
    main()
