from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from echopulse.audio import load_audio, add_noise, shift_pitch
from echopulse.features import extract_mfcc_features

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


@dataclass
class SampleRecord:
    path: str
    label: str
    variant: str


def iter_audio_files(data_dir: str | Path) -> Iterable[tuple[Path, str]]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        label = class_dir.name
        for file_path in sorted(class_dir.rglob("*")):
            if file_path.suffix.lower() in AUDIO_EXTENSIONS:
                yield file_path, label


def build_feature_dataframe(data_dir: str | Path, config: dict) -> pd.DataFrame:
    audio_cfg = config["audio"]
    aug_cfg = config["augmentation"]

    records: list[SampleRecord] = []
    features: list[np.ndarray] = []

    for file_path, label in iter_audio_files(data_dir):
        signal = load_audio(
            str(file_path),
            sample_rate=audio_cfg["sample_rate"],
            duration_seconds=audio_cfg["duration_seconds"],
        )

        variants: list[tuple[str, np.ndarray]] = []
        if aug_cfg.get("generate_original", True):
            variants.append(("original", signal))
        if aug_cfg.get("enabled", True) and aug_cfg.get("generate_noise", True):
            variants.append(("noise", add_noise(signal, aug_cfg["noise_factor"])))
        if aug_cfg.get("enabled", True) and aug_cfg.get("generate_pitch", True):
            variants.append(("pitch", shift_pitch(signal, audio_cfg["sample_rate"], aug_cfg["pitch_steps"])))

        for variant_name, variant_signal in variants:
            feature_vector = extract_mfcc_features(
                variant_signal,
                sample_rate=audio_cfg["sample_rate"],
                n_mfcc=audio_cfg["n_mfcc"],
                n_fft=audio_cfg["n_fft"],
                hop_length=audio_cfg["hop_length"],
            )
            features.append(feature_vector)
            records.append(SampleRecord(str(file_path), label, variant_name))

    if not features:
        raise ValueError("No supported audio files were found in the data directory.")

    feature_matrix = np.vstack(features)
    feature_columns = [f"f_{i}" for i in range(feature_matrix.shape[1])]
    df = pd.DataFrame(feature_matrix, columns=feature_columns)
    df["path"] = [r.path for r in records]
    df["label"] = [r.label for r in records]
    df["variant"] = [r.variant for r in records]
    return df
