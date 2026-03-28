from __future__ import annotations

import argparse
import json

import numpy as np

from echopulse.audio import load_audio
from echopulse.config import load_config
from echopulse.features import extract_mfcc_features
from echopulse.modeling import load_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-file inference with EchoPulse.")
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--bundle_path", type=str, default="models/model_bundle.joblib")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default="ensemble", choices=["rf", "svm", "ensemble"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    bundle = load_bundle(args.bundle_path)

    audio_cfg = config["audio"]
    signal = load_audio(
        args.audio_path,
        sample_rate=audio_cfg["sample_rate"],
        duration_seconds=audio_cfg["duration_seconds"],
    )
    features = extract_mfcc_features(
        signal,
        sample_rate=audio_cfg["sample_rate"],
        n_mfcc=audio_cfg["n_mfcc"],
        n_fft=audio_cfg["n_fft"],
        hop_length=audio_cfg["hop_length"],
    ).reshape(1, -1)

    model = {
        "rf": bundle.rf_pipeline,
        "svm": bundle.svm_pipeline,
        "ensemble": bundle.ensemble_pipeline,
    }[args.model]

    pred_encoded = model.predict(features)[0]
    pred_label = bundle.label_encoder.inverse_transform([pred_encoded])[0]

    output = {"predicted_label": pred_label}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        output["probabilities"] = {
            label: float(prob)
            for label, prob in zip(bundle.classes_, probabilities)
        }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
