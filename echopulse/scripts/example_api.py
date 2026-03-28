from __future__ import annotations

from flask import Flask, jsonify, request

from echopulse.audio import load_audio
from echopulse.config import load_config
from echopulse.features import extract_mfcc_features
from echopulse.modeling import load_bundle

app = Flask(__name__)
config = load_config("config.yaml")
bundle = load_bundle("models/model_bundle.joblib")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    audio_path = payload.get("audio_path")
    model_name = payload.get("model", "ensemble")

    if not audio_path:
        return jsonify({"error": "audio_path is required"}), 400

    audio_cfg = config["audio"]
    signal = load_audio(
        audio_path,
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
    }.get(model_name, bundle.ensemble_pipeline)

    pred_encoded = model.predict(features)[0]
    pred_label = bundle.label_encoder.inverse_transform([pred_encoded])[0]
    return jsonify({"predicted_label": pred_label, "model": model_name})


if __name__ == "__main__":
    app.run(debug=True)
