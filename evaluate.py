"""Evaluate synthetic EMG tabular features with a held-out raw-data split."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 42


def load_samples(path):
    data = pd.read_csv(path)
    if data["Target"].isna().any() or set(data["Target"]) != {0, 1}:
        raise ValueError("Target must contain both binary labels with no missing values.")
    features = data.drop(columns="Target").apply(pd.to_numeric)
    if features.duplicated().any():
        raise ValueError("Repeated feature rows must be resolved before splitting.")
    return features, data["Target"].astype(int)


def legacy_audit(path):
    """Describe the overlap in the archived notebook's pre-resampled split."""
    data = pd.read_csv(path)
    train, test = train_test_split(data, test_size=0.30, random_state=SEED, stratify=data["Target"])
    train_rows = set(train.drop(columns="Target").itertuples(index=False, name=None))
    test_rows = test.drop(columns="Target").itertuples(index=False, name=None)
    return {
        "processed_rows": len(data),
        "test_rows": len(test),
        "test_rows_matching_a_training_feature_row": sum(row in train_rows for row in test_rows),
    }


def evaluate(path, legacy_path=None):
    features, labels = load_samples(path)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.20, stratify=labels, random_state=SEED,
    )
    methods = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"), RandomForestClassifier(random_state=SEED),
        ),
    }
    scores = {}
    for name, model in methods.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        tn, fp, fn, tp = (int(x) for x in matrix.ravel())
        scores[name] = {
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp),
            "confusion_matrix": matrix.tolist(),
        }
    result = {
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seed": SEED,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "train_rows": len(y_train),
        "test_rows": len(y_test),
        "test_positive": int(y_test.sum()),
        "missing_feature_values": int(features.isna().sum().sum()),
        "models": scores,
    }
    if legacy_path is not None:
        result["legacy_split_audit"] = legacy_audit(legacy_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "preterm_birth_dataset.csv")
    parser.add_argument("--audit-legacy", type=Path, help="Optional archived pre-resampled CSV")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = json.dumps(evaluate(args.data, args.audit_legacy), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
