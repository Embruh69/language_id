"""
inference.py
------------
Lightweight, real-time inference wrapper for the language ID model.

Design goals for per-word latency:
  • Skip noisereduce (too slow for ~0.3 s clips) — use simple bandpass instead.
  • Skip librosa.pyin (slow pitch tracker) — zero-pad its 2 dims so the scaler
    shape still matches the training-time vector.
  • Everything else (MFCC + deltas, chroma, spectral contrast, mel-band energy,
    ZCR, RMS, tempo) is fast enough for real-time use (~80–150 ms total).
"""

from __future__ import annotations

import time
import threading
import numpy as np
import librosa
import cv2
import joblib
import tensorflow as tf
from dataclasses import dataclass


# ── Constants (must match training config exactly) ────────────────────────────
TARGET_SR      = 16_000
CLIP_DURATION  = 10.0
N_SAMPLES      = int(TARGET_SR * CLIP_DURATION)
N_MFCC         = 40
N_MELS         = 128
N_FFT          = 2048
HOP_LEN        = 512
IMG_H, IMG_W   = 128, 128
N_CHANNELS     = 3


@dataclass
class Prediction:
    label: str
    probs: dict          # {language: float}
    latency_ms: float
    audio_duration_s: float


class LanguageIDInference:
    """
    Thread-safe inference wrapper using TensorFlow/Keras.

    Parameters
    ----------
    artefacts_dir : str
        Directory produced by the training notebook, containing:
        - language_id_model.keras
        - scaler.pkl
        - label_encoder.pkl
    """

    def __init__(self, artefacts_dir: str):
        self.model = tf.keras.models.load_model(
            f"{artefacts_dir}/language_id_model.keras",
            compile=False,
        )

        self.scaler   = joblib.load(f"{artefacts_dir}/scaler.pkl")
        self.le       = joblib.load(f"{artefacts_dir}/label_encoder.pkl")
        self._vec_len = self.scaler.n_features_in_
        self._lock    = threading.Lock()

        # Warm up — first TF call is always slow
        _img = np.zeros((1, IMG_H, IMG_W, N_CHANNELS), dtype=np.float32)
        _vec = np.zeros((1, self._vec_len),             dtype=np.float32)
        self.model.predict([_img, _vec], verbose=0)

    # ── Audio helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _light_clean(y: np.ndarray) -> np.ndarray:
        """
        Fast cleaning for short word clips.
          1. Bandpass 80–8000 Hz (keeps speech, drops rumble/hiss)
          2. Trim silence
          3. Fix length to N_SAMPLES (pad/truncate)
          4. Peak normalise
        """
        from scipy.signal import butter, sosfilt

        nyq = TARGET_SR / 2.0
        sos = butter(4, [80 / nyq, 8000 / nyq], btype="band", output="sos")
        y   = sosfilt(sos, y).astype(np.float32)

        y, _ = librosa.effects.trim(y, top_db=25)

        if len(y) == 0:
            y = np.zeros(N_SAMPLES, dtype=np.float32)
        elif len(y) < N_SAMPLES:
            y = np.pad(y, (0, N_SAMPLES - len(y)))
        else:
            y = y[:N_SAMPLES]

        peak = np.max(np.abs(y))
        return (y / peak).astype(np.float32) if peak > 1e-6 else y

    @staticmethod
    def _feature_vector_fast(y: np.ndarray) -> np.ndarray:
        """
        Full feature vector, skipping librosa.pyin (too slow for real-time).
        The 2 pitch dims are zero-padded so the scaler shape stays consistent.
        """
        sr    = TARGET_SR
        feats = []

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                     n_fft=N_FFT, hop_length=HOP_LEN)
        d1   = librosa.feature.delta(mfcc)
        d2   = librosa.feature.delta(mfcc, order=2)
        for m in (mfcc, d1, d2):
            feats.extend(np.mean(m, axis=1))
            feats.extend(np.std(m,  axis=1))

        chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                              n_fft=N_FFT, hop_length=HOP_LEN)
        feats.extend(np.mean(chroma, axis=1))
        feats.extend(np.std(chroma,  axis=1))

        sc = librosa.feature.spectral_contrast(y=y, sr=sr,
                                                n_fft=N_FFT, hop_length=HOP_LEN)
        feats.extend(np.mean(sc, axis=1))
        feats.extend(np.std(sc,  axis=1))

        mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                                 n_fft=N_FFT, hop_length=HOP_LEN)
        mel_db = librosa.power_to_db(mel)
        for band in np.array_split(mel_db, 6, axis=0):
            feats.append(float(np.mean(band)))
            feats.append(float(np.std(band)))

        zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LEN)
        feats += [float(np.mean(zcr)), float(np.std(zcr))]

        rms = librosa.feature.rms(y=y, hop_length=HOP_LEN)
        feats += [float(np.mean(rms)), float(np.std(rms))]

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats.append(float(tempo))

        # Zero-pad the 2 pitch dims (mean F0, voiced fraction)
        feats += [0.0, 0.0]

        return np.array(feats, dtype=np.float32)

    @staticmethod
    def _spectrogram_image(y: np.ndarray) -> np.ndarray:
        mel    = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS,
                                                 n_fft=N_FFT, hop_length=HOP_LEN)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        d1     = librosa.feature.delta(mel_db)
        d2     = librosa.feature.delta(mel_db, order=2)

        def norm(x):
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-8)

        c0 = cv2.resize(norm(mel_db), (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        c1 = cv2.resize(norm(d1),     (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        c2 = cv2.resize(norm(d2),     (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)
        return np.stack([c0, c1, c2], axis=-1).astype(np.float32)

    # ── Public API ────────────────────────────────────────────────────────────

    def predict_waveform(self, y: np.ndarray, sr: int = TARGET_SR) -> Prediction:
        """
        Predict language from a raw waveform array.

        Parameters
        ----------
        y  : float32 ndarray — raw PCM samples
        sr : int — sample rate of y

        Returns
        -------
        Prediction dataclass
        """
        t0 = time.perf_counter()

        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

        audio_duration = len(y) / TARGET_SR
        y = self._light_clean(y)

        vec = self._feature_vector_fast(y)
        if len(vec) < self._vec_len:
            vec = np.pad(vec, (0, self._vec_len - len(vec)))
        elif len(vec) > self._vec_len:
            vec = vec[:self._vec_len]

        vec_sc = self.scaler.transform(vec.reshape(1, -1)).astype(np.float32)
        img    = self._spectrogram_image(y)[np.newaxis]  # (1, H, W, 3)

        with self._lock:
            raw_probs = self.model.predict([img, vec_sc], verbose=0)[0]

        label = self.le.inverse_transform([int(np.argmax(raw_probs))])[0]
        probs = dict(zip(self.le.classes_.tolist(), raw_probs.tolist()))

        latency_ms = (time.perf_counter() - t0) * 1000
        return Prediction(label=label, probs=probs,
                          latency_ms=latency_ms,
                          audio_duration_s=audio_duration)
