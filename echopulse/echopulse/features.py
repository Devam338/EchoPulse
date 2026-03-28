from __future__ import annotations

import numpy as np
import librosa


def extract_mfcc_features(
    signal: np.ndarray,
    sample_rate: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate(
        [
            mfcc.mean(axis=1),
            mfcc.std(axis=1),
            delta.mean(axis=1),
            delta.std(axis=1),
            delta2.mean(axis=1),
            delta2.std(axis=1),
        ]
    )
    return features.astype(np.float32)
