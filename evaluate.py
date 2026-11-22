"""Evaluate synthetic EMG tabular features with a held-out raw-data split."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

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


def wilson_interval(successes, trials):
    """Two-sided 95% Wilson score interval for a binomial proportion."""
    if type(successes) is not int or type(trials) is not int or trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Binomial counts require 0 <= successes <= trials and trials > 0.")
    z = NormalDist().inv_cdf(0.975)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    variance = proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    radius = z * np.sqrt(variance) / denominator
    return [float(max(0, centre - radius)), float(min(1, centre + radius))]


def paired_auc_interval(labels, reference, candidate, n_resamples=2000, seed=SEED):
    """Bootstrap the AUC difference while preserving paired rows and class counts."""
    labels, reference, candidate = (np.asarray(values) for values in (labels, reference, candidate))
    if labels.ndim != 1 or reference.shape != labels.shape or candidate.shape != labels.shape:
        raise ValueError("Labels and both score arrays must have the same one-dimensional shape.")
    if set(labels) != {0, 1} or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Both label classes and finite prediction scores are required.")
    if type(n_resamples) is not int or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer.")
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(n_resamples):
        indices = np.concatenate((
            rng.choice(positive, len(positive)), rng.choice(negative, len(negative)),
        ))
        differences.append(
            roc_auc_score(labels[indices], candidate[indices])
            - roc_auc_score(labels[indices], reference[indices])
        )
    return {
        "difference": float(roc_auc_score(labels, candidate) - roc_auc_score(labels, reference)),
        "confidence_interval": np.quantile(differences, [0.025, 0.975]).tolist(),
        "confidence_level": 0.95,
        "method": "stratified paired percentile bootstrap",
        "n_resamples": n_resamples,
        "seed": seed,
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
    test_scores = {}
    for name, model in methods.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        probabilities = model.predict_proba(x_test)[:, 1]
        test_scores[name] = probabilities
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        tn, fp, fn, tp = (int(x) for x in matrix.ravel())
        scores[name] = {
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
            "sensitivity": tp / (tp + fn),
            "specificity": tn / (tn + fp),
            "confusion_matrix": matrix.tolist(),
            "binomial_confidence": {
                "method": "Wilson score",
                "confidence_level": 0.95,
                "sensitivity": {"successes": tp, "trials": tp + fn,
                                "interval": wilson_interval(tp, tp + fn)},
                "specificity": {"successes": tn, "trials": tn + fp,
                                "interval": wilson_interval(tn, tn + fp)},
            },
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
        "paired_auc_comparison": {
            "reference": "logistic",
            "candidate": "random_forest",
            **paired_auc_interval(y_test, test_scores["logistic"], test_scores["random_forest"]),
        },
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
