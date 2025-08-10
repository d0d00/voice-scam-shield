from __future__ import annotations

from typing import Optional
import os
from pathlib import Path
import logging
import numpy as np

try:
    import torch
    import torchaudio
except Exception:  # optional runtime deps
    torch = None  # type: ignore
    torchaudio = None  # type: ignore


class AASISTScorer:
    """Load and run AASIST anti-spoofing model if available.

    Expects 16 kHz mono float32 in [-1, 1]. Returns probability of synthetic voice [0, 1].
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu", target_samples: int = 64600) -> None:
        self.available = False
        self.device = device
        self.model = None
        self._checkpoint = checkpoint_path
        self.target_samples = int(target_samples)
        self._logger = logging.getLogger("vss")
        if torch is None:
            self._logger.info("AASIST unavailable: torch not installed")
            return
        try:
            if checkpoint_path:
                ckpt = checkpoint_path
                if not os.path.isabs(ckpt):
                    base = Path(__file__).resolve().parents[2]  # .../hacknation
                    abs_path = base / ckpt
                else:
                    abs_path = Path(ckpt)
                if not abs_path.exists():
                    self._logger.warning("AASIST checkpoint missing at %s; spoofing disabled", abs_path)
                    return
                self.model = torch.jit.load(str(abs_path), map_location=self.device)
                self.model.eval()
                self.available = True
            else:
                self._logger.info("AASIST checkpoint not set; spoofing disabled")
        except Exception as e:
            self.model = None
            self.available = False
            self._logger.error("AASIST load failed: %s", e)

    @torch.no_grad() if torch is not None else (lambda f: f)  # type: ignore
    def score(self, samples: np.ndarray, sample_rate: int = 16000) -> float:
        # Guard rails: return safe 0.0 when unavailable
        if samples is None or getattr(samples, "size", 0) == 0:
            return 0.0
        if self.model is None or torch is None:
            return 0.0
        try:
            x = np.asarray(samples, dtype=np.float32)
            # Ensure length exactly target_samples: pad or take last segment
            ts = self.target_samples
            if x.shape[0] < ts:
                pad = np.zeros(ts - x.shape[0], dtype=np.float32)
                x = np.concatenate([x, pad], axis=0)
            elif x.shape[0] > ts:
                x = x[-ts:]

            t = torch.from_numpy(x).to(self.device)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            logits = self.model(t)
            # Some scripted models return (logits, extras)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            # Assume 2-class logits [bonafide, spoof]
            if hasattr(torch, "softmax") and hasattr(logits, "ndim") and logits.ndim == 2 and logits.shape[1] == 2:
                probs = torch.softmax(logits, dim=1)
                prob = float(probs[0, 1].item())
            else:
                prob = float(torch.sigmoid(logits.squeeze()).mean().item())
            if not (prob == prob):  # NaN
                return 0.0
            return float(max(0.0, min(1.0, prob)))
        except Exception as e:
            self._logger.warning("AASIST score error: %s", e)
            return 0.0


