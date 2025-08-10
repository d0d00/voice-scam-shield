from __future__ import annotations

import numpy as np


class EnergyVAD:
    """Simple energy-based VAD with hangover.

    Parameters
    -----------
    sample_rate: int
        Sampling rate of audio, expected 16000.
    frame_ms: float
        Frame size in milliseconds for analysis.
    threshold_db: float
        Energy threshold in dBFS to consider speech active.
    hangover_ms: float
        Time to keep VAD active after speech drops below threshold.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: float = 20.0,
        threshold_db: float = -45.0,
        hangover_ms: float = 200.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_len = int(sample_rate * (frame_ms / 1000.0))
        self.threshold_db = float(threshold_db)
        self.hangover_frames = int(round(hangover_ms / frame_ms))
        self._hang = 0

    def is_speech(self, samples: np.ndarray) -> bool:
        if samples.size == 0:
            return False
        # Ensure 1D float32
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)
        # Frame-wise check using last frame
        frame = samples[-self.frame_len :] if samples.size >= self.frame_len else samples
        # RMS to dBFS
        rms = np.sqrt(np.mean(np.square(frame))) + 1e-9
        db = 20.0 * np.log10(rms)
        active = db > self.threshold_db
        if active:
            self._hang = self.hangover_frames
            return True
        if self._hang > 0:
            self._hang -= 1
            return True
        return False


