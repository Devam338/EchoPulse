from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from echopulse.config import load_config
from echopulse.dataset import build_feature_dataframe
from echopulse.modeling import (
    TrainedBundle,
    build_ensemble,
    build_rf_pipeline,
    build_svm_pipeline,
    evaluate_model,
    save_bundle,
)
from echopulse.utils.io import ensure_dir, write_json
from echopulse.utils.plots import save_confusion_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EchoPulse ML models.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing class subfolders of audio files.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML.")
    parser.add_argument("--model", type=str, default="all", choices=["rf", "svm", "ensemble", "all"])
    parser.add_argument("--output_dir", type=str, default="models")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    artifacts_dir = ensure_dir("artifacts")

    df = build_feature_dataframe(args.data_dir, config)
    X = df.filter(regex=r"^f_").to_numpy(dtype=np.float32)
    y_labels = df["label"].to_numpy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)

    split_kwargs = {
        "test_size": config["training"]["test_size"],
        "random_state": config["training"]["random_state"],
    }
    if config["training"].get("stratify", True):
        split_kwargs["stratify"] = y

    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

    rf_pipeline = build_rf_pipeline(config)
    svm_pipeline = build_svm_pipeline(config)
    ensemble_pipeline = build_ensemble(config)

    models_to_run = []
    if args.model in {"rf", "all"}:
        models_to_run.append(("rf", rf_pipeline))
    if args.model in {"svm", "all"}:
        models_to_run.append(("svm", svm_pipeline))
    if args.model in {"ensemble", "all"}:
        models_to_run.append(("ensemble", ensemble_pipeline))

    metrics_summary = {}
    class_names = list(label_encoder.classes_)

    for model_name, model in models_to_run:
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, label_encoder)
        metrics_summary[model_name] = {
            "accuracy": float(metrics["accuracy"]),
            "classification_report": metrics["classification_report"],
        }
        save_confusion_matrix(
            metrics["confusion_matrix"],
            class_names,
            artifacts_dir / f"{model_name}_confusion_matrix.png",
            title=f"EchoPulse {model_name.upper()} Confusion Matrix",
        )
        print(f"\n=== {model_name.upper()} ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(metrics["classification_report"])

    rf_pipeline.fit(X_train, y_train)
    svm_pipeline.fit(X_train, y_train)
    ensemble_pipeline.fit(X_train, y_train)

    bundle = TrainedBundle(
        label_encoder=label_encoder,
        rf_pipeline=rf_pipeline,
        svm_pipeline=svm_pipeline,
        ensemble_pipeline=ensemble_pipeline,
        classes_=class_names,
    )
    save_bundle(bundle, str(output_dir / "model_bundle.joblib"))
    write_json(output_dir / "metrics_summary.json", metrics_summary)
    print(f"\nSaved model bundle to {output_dir / 'model_bundle.joblib'}")


if __name__ == "__main__":
    main()
