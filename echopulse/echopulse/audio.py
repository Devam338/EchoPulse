from __future__ import annotations

import numpy as np
import librosa


def load_audio(audio_path: str, sample_rate: int, duration_seconds: int) -> np.ndarray:
    target_length = sample_rate * duration_seconds
    signal, _ = librosa.load(audio_path, sr=sample_rate, mono=True)

    if len(signal) < target_length:
        pad_width = target_length - len(signal)
        signal = np.pad(signal, (0, pad_width), mode="constant")
    else:
        signal = signal[:target_length]

    return signal.astype(np.float32)


def add_noise(signal: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
    noise = np.random.randn(len(signal)).astype(np.float32)
    augmented = signal + noise_factor * noise
    return np.clip(augmented, -1.0, 1.0)


def shift_pitch(signal: np.ndarray, sample_rate: int, n_steps: float = 2.0) -> np.ndarray:
    return librosa.effects.pitch_shift(signal, sr=sample_rate, n_steps=n_steps).astype(np.float32)
