import os
import pandas as pd
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATASET_NAME = "doanquanvietnamca/liar-dataset"


def download_dataset(dataset_name=DATASET_NAME, data_dir=DATA_DIR):
    """
    Downloads the LIAR dataset from Kaggle.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading dataset '{dataset_name}' to '{data_dir}'...")
    api.dataset_download_files(dataset_name, path=data_dir, unzip=True)
    print("Download complete.")


def load_dataset(data_dir=DATA_DIR, download=False):
    """
    Loads the LIAR dataset from a local path or downloads it if specified.

    The LIAR dataset has 6 classes:
    - pants-fire
    - false
    - barely-true
    - half-true
    - mostly-true
    - true

    We map these to a binary classification:
    - `pants-fire`, `false`, `barely-true`, `half-true` are mapped to 0 (fake).
    - `mostly-true`, `true` are mapped to 1 (real).
    """
    if download:
        download_dataset(data_dir=data_dir)

    train_path = os.path.join(data_dir, "train.tsv")
    test_path = os.path.join(data_dir, "test.tsv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Dataset files not found. Please run with --download to fetch them from Kaggle."
        )

    train_df = pd.read_csv(train_path, sep='\t', header=None)
    test_df = pd.read_csv(test_path, sep='\t', header=None)

    # Assign column names based on the dataset description
    columns = [
        "id", "label", "statement", "subject", "speaker", "job_title",
        "state_info", "party_affiliation", "barely_true_counts",
        "false_counts", "half_true_counts", "mostly_true_counts",
        "pants_on_fire_counts", "context"
    ]
    train_df.columns = columns
    test_df.columns = columns

    # Combine train and test sets for a larger dataset
    df = pd.concat([train_df, test_df], ignore_index=True)

    # Map labels to binary
    label_mapping = {
        "pants-fire": 0,
        "false": 0,
        "barely-true": 0,
        "half-true": 0,
        "mostly-true": 1,
        "true": 1
    }
    df["label"] = df["label"].map(label_mapping)

    # Preprocess text
    df["text"] = df["statement"].apply(
        lambda x: x.lower() if isinstance(x, str) else ""
    )

    return df[["text", "label"]]