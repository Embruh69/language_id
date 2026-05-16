"""
vad.py
------
Energy-based Voice Activity Detection (VAD) for real-time word segmentation.

Why not Silero-VAD / WebRTC-VAD?
  - Silero requires torch (~1 GB), too heavy for Streamlit Cloud.
  - WebRTC-VAD only accepts 10/20/30 ms frames of 8/16/32/48 kHz int16.
  - Our simple energy + ZCR VAD is fast, dependency-free, and robust enough
    for the demo use case.

Algorithm
---------
The AudioBuffer accumulates incoming PCM frames. A sliding window computes
short-time energy. When energy crosses a speech threshold the state becomes
SPEAKING. When it drops below a silence threshold for MIN_SILENCE_S seconds
the word boundary is detected and the accumulated segment is flushed.
"""

from __future__ import annotations

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ── VAD parameters ─────────────────────────────────────────────────────────
TARGET_SR          = 16_000
FRAME_MS           = 20          # analysis frame (ms)
FRAME_SAMPLES      = int(TARGET_SR * FRAME_MS / 1000)   # 320 samples

SPEECH_ENERGY_THR  = 0.002       # RMS energy to enter SPEAKING state
SILENCE_ENERGY_THR = 0.001       # RMS energy to stay SPEAKING
MIN_SPEECH_MS      = 100         # minimum word length to emit
MIN_SILENCE_MS     = 400         # silence gap required to flush a word
MAX_WORD_MS        = 2_000       # hard cap per word (avoid mega-words)

MIN_SPEECH_FRAMES  = MIN_SPEECH_MS  // FRAME_MS
MIN_SILENCE_FRAMES = MIN_SILENCE_MS // FRAME_MS
MAX_WORD_FRAMES    = MAX_WORD_MS    // FRAME_MS


@dataclass
class WordSegment:
    samples: np.ndarray   # float32 PCM
    sr:      int = TARGET_SR


class VoiceActivityDetector:
    """
    Stateful, streaming VAD.

    Feed PCM frames with `push(frame)`.
    Call `pop_word()` to retrieve completed word segments (returns None if
    no word is ready yet).
    """

    def __init__(self):
        self._speaking      = False
        self._speech_buf    : list[np.ndarray] = []
        self._silence_count : int = 0
        self._speech_count  : int = 0
        self._pending       : Optional[WordSegment] = None

    # ── Internal ─────────────────────────────────────────────────────────────

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

    def _flush(self) -> Optional[WordSegment]:
        """Concatenate buffered frames into a WordSegment and clear state."""
        if len(self._speech_buf) < MIN_SPEECH_FRAMES:
            self._speech_buf.clear()
            return None
        samples = np.concatenate(self._speech_buf).astype(np.float32)
        self._speech_buf.clear()
        return WordSegment(samples=samples)

    # ── Public ────────────────────────────────────────────────────────────────

    def push(self, frame: np.ndarray) -> None:
        """
        Ingest one PCM frame (any length; internally chunked to FRAME_SAMPLES).
        """
        # Chunk the incoming frame into FRAME_MS analysis windows
        pos = 0
        while pos < len(frame):
            chunk = frame[pos : pos + FRAME_SAMPLES]
            pos  += FRAME_SAMPLES

            energy = self._rms(chunk)

            if not self._speaking:
                if energy > SPEECH_ENERGY_THR:
                    self._speaking      = True
                    self._silence_count = 0
                    self._speech_count  = 0
                    self._speech_buf    = [chunk]
                    self._speech_count += 1
            else:
                self._speech_buf.append(chunk)
                self._speech_count += 1

                if energy < SILENCE_ENERGY_THR:
                    self._silence_count += 1
                else:
                    self._silence_count = 0

                # Word boundary: enough silence after speech
                if self._silence_count >= MIN_SILENCE_FRAMES:
                    word = self._flush()
                    self._speaking      = False
                    self._silence_count = 0
                    if word is not None:
                        self._pending = word

                # Hard cap — flush anyway
                elif self._speech_count >= MAX_WORD_FRAMES:
                    word = self._flush()
                    self._speaking      = False
                    self._silence_count = 0
                    if word is not None:
                        self._pending = word

    def pop_word(self) -> Optional[WordSegment]:
        """Return the next completed word segment, or None."""
        word          = self._pending
        self._pending = None
        return word

    def reset(self) -> None:
        self._speaking      = False
        self._speech_buf    = []
        self._silence_count = 0
        self._speech_count  = 0
        self._pending       = None
