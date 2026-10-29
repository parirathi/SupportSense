"""
Model Training & Evaluation Script for SupportSense.

Splits data into train and held-out test sets.
Strictly evaluates test performance (never reports training set metrics as test results).
Saves the trained pipeline and records detailed evaluation metrics in data/metrics.json.
"""

import os
import json
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from model.classifier import SupportSenseClassifier, DEFAULT_MODEL_PATH


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "support_feedback.csv")
METRICS_PATH = os.path.join(BASE_DIR, "data", "metrics.json")


def train_and_evaluate(
    data_path: str = DATA_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    metrics_path: str = METRICS_PATH,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Trains TF-IDF + LogisticRegression model and evaluates strictly on test data."""
    print("=" * 65)
    print("SupportSense: Training Sentiment Classification Pipeline")
    print("=" * 65)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at '{data_path}'")

    df = pd.read_csv(data_path)
    df.dropna(subset=["text", "sentiment"], inplace=True)
    df["text"] = df["text"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

    # Filter only valid classes
    valid_classes = {"negative", "neutral", "positive"}
    df = df[df["sentiment"].isin(valid_classes)]

    print(f"Dataset successfully loaded from: {data_path}")
    print(f"Total labeled samples: {len(df)}")
    print("Class distribution:")
    for cls, count in df["sentiment"].value_counts().items():
        print(f"  - {cls.capitalize():<10}: {count} samples ({count / len(df) * 100:.1f}%)")

    # Stratified Train-Test Split
    X = df["text"].tolist()
    y = df["sentiment"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("\nDataset Split:")
    print(f"  - Training samples: {len(X_train)} ({(1 - test_size) * 100:.0f}%)")
    print(f"  - Held-out test samples: {len(X_test)} ({test_size * 100:.0f}%)")

    # Train Classifier
    classifier = SupportSenseClassifier(model_path=model_path)
    classifier.fit(X_train, y_train)

    # Evaluate ONLY on held-out test data
    y_pred = [classifier.predict(x) for x in X_test]

    labels = sorted(list(valid_classes))
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    class_report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    print("\n" + "=" * 65)
    print("HELD-OUT TEST SET EVALUATION RESULTS (Unseen Data Only)")
    print("=" * 65)
    print(f"  Accuracy:          {acc * 100:.2f}%")
    print(f"  Macro Precision:   {prec_macro * 100:.2f}%")
    print(f"  Macro Recall:      {rec_macro * 100:.2f}%")
    print(f"  Macro F1-Score:    {f1_macro * 100:.2f}%")
    print(f"  Weighted F1-Score: {f1_weighted * 100:.2f}%")
    print("-" * 65)
    print("Per-Class Metrics:")
    for lbl in labels:
        metrics = class_report[lbl]
        print(
            f"  {lbl.capitalize():<10} | Precision: {metrics['precision'] * 100:6.2f}% | "
            f"Recall: {metrics['recall'] * 100:6.2f}% | F1: {metrics['f1-score'] * 100:6.2f}% | "
            f"Support: {int(metrics['support'])}"
        )
    print("=" * 65)

    # Save trained pipeline
    saved_model_path = classifier.save()
    print(f"\nModel pipeline saved to: {saved_model_path}")

    # Prepare detailed evaluation record
    evaluation_record = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_architecture": "TF-IDF Vectorizer (ngram 1-2, sublinear_tf) + LogisticRegression (class_weight='balanced')",
        "dataset_name": "SupportSense Customer Feedback Demonstration Dataset",
        "dataset_license": "MIT License (Synthetic representative domain benchmark)",
        "split": {
            "total_samples": len(df),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "test_ratio": test_size,
            "random_state": random_state,
            "stratified": True,
        },
        "held_out_test_metrics": {
            "accuracy": round(acc, 4),
            "macro_precision": round(prec_macro, 4),
            "macro_recall": round(rec_macro, 4),
            "macro_f1": round(f1_macro, 4),
            "weighted_precision": round(prec_weighted, 4),
            "weighted_recall": round(rec_weighted, 4),
            "weighted_f1": round(f1_weighted, 4),
            "labels": labels,
            "per_class": {
                lbl: {
                    "precision": round(class_report[lbl]["precision"], 4),
                    "recall": round(class_report[lbl]["recall"], 4),
                    "f1_score": round(class_report[lbl]["f1-score"], 4),
                    "support": int(class_report[lbl]["support"]),
                }
                for lbl in labels
            },
            "confusion_matrix": conf_matrix,
        },
        "limitations": [
            "Dataset is a compact demonstration set (150 representative tickets across positive, neutral, negative categories).",
            "Domain specific to SaaS, billing, and technical customer support queries; out-of-domain conversational text may produce uncertain confidence.",
            "Subtle sarcasm, mixed sentiment paragraphs, and colloquial slang might be misclassified.",
            "Heuristic priority flags are rule-driven keyword/sentiment combinations, NOT a dedicated supervised urgency classifier.",
        ],
    }

    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_record, f, indent=2)

    print(f"Evaluation metrics recorded in: {metrics_path}")
    return evaluation_record


if __name__ == "__main__":
    train_and_evaluate()
