from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


@dataclass
class TrainedBundle:
    label_encoder: LabelEncoder
    rf_pipeline: Pipeline
    svm_pipeline: Pipeline
    ensemble_pipeline: VotingClassifier
    classes_: list[str]


def build_rf_pipeline(config: dict) -> Pipeline:
    rf_cfg = config["random_forest"]
    return Pipeline(
        steps=[
            ("rf", RandomForestClassifier(
                n_estimators=rf_cfg["n_estimators"],
                max_depth=rf_cfg["max_depth"],
                min_samples_split=rf_cfg["min_samples_split"],
                min_samples_leaf=rf_cfg["min_samples_leaf"],
                random_state=config["training"]["random_state"],
                n_jobs=-1,
            ))
        ]
    )


def build_svm_pipeline(config: dict) -> Pipeline:
    svm_cfg = config["svm"]
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=svm_cfg["C"],
                kernel=svm_cfg["kernel"],
                gamma=svm_cfg["gamma"],
                probability=svm_cfg["probability"],
                random_state=config["training"]["random_state"],
            ))
        ]
    )


def build_ensemble(config: dict) -> VotingClassifier:
    rf = RandomForestClassifier(
        n_estimators=config["random_forest"]["n_estimators"],
        max_depth=config["random_forest"]["max_depth"],
        min_samples_split=config["random_forest"]["min_samples_split"],
        min_samples_leaf=config["random_forest"]["min_samples_leaf"],
        random_state=config["training"]["random_state"],
        n_jobs=-1,
    )
    svm = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=config["svm"]["C"],
                kernel=config["svm"]["kernel"],
                gamma=config["svm"]["gamma"],
                probability=config["svm"]["probability"],
                random_state=config["training"]["random_state"],
            ))
        ]
    )
    return VotingClassifier(estimators=[("rf", rf), ("svm", svm)], voting="soft")


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, label_encoder: LabelEncoder) -> dict[str, Any]:
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(
        y_test,
        preds,
        target_names=label_encoder.inverse_transform(sorted(np.unique(y_test))),
        zero_division=0,
    )
    cm = confusion_matrix(y_test, preds)
    return {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": cm,
        "predictions": preds,
    }


def save_bundle(bundle: TrainedBundle, path: str) -> None:
    joblib.dump(bundle, path)


def load_bundle(path: str) -> TrainedBundle:
    return joblib.load(path)
