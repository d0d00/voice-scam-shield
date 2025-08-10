from __future__ import annotations

from typing import Optional, Tuple
import numpy as np



class WhisperStreamer:
    """Thin wrapper around faster-whisper for short streaming chunks.

    Designed for 16 kHz mono Float32 audio arrays. Maintains a running
    partial transcript and last detected language.
    """

    def __init__(self, model_size: str = "tiny", device: str = "cpu", compute_type: str = "int8") -> None:
        # Lazy import with graceful fallback if faster-whisper is unavailable
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except Exception:
            WhisperModel = None  # type: ignore
        self._WhisperModel = WhisperModel
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self.model = None
        self.partial_transcript: str = ""
        self.last_language: Optional[str] = None
        self.available: bool = False
        self.fallback_used: Optional[str] = None  # compute_type actually used, if fallback occurred
        self._last_chunk_text: str = ""

    def _append_unique(self, new_text: str) -> None:
        if not new_text:
            return
        if not self.partial_transcript:
            self.partial_transcript = new_text
            return
        # Deduplicate by overlapping suffix/prefix
        tail = self.partial_transcript[-80:]
        if new_text.startswith(tail):
            self.partial_transcript += new_text[len(tail) :]
            return
        # Find max overlap
        max_olap = 0
        for k in range(min(len(tail), len(new_text)), 10, -1):
            if tail[-k:] == new_text[:k]:
                max_olap = k
                break
        self.partial_transcript += (" " if max_olap == 0 else "") + new_text[max_olap:]

    def transcribe_chunk(self, audio: np.ndarray, sample_rate: int = 16000) -> Tuple[str, Optional[str]]:
        if audio is None or audio.size == 0:
            return "", self.last_language
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        # Initialize model on first use if available
        if self.model is None and self._WhisperModel is not None:
            try:
                # Try preferred compute type first, then fall back
                try_types = [self._compute_type, "int8_float16", "float16", "int8", "float32"]
                last_err = None
                for ct in try_types:
                    try:
                        self.model = self._WhisperModel(self._model_size, device=self._device, compute_type=ct)
                        self.fallback_used = ct if ct != self._compute_type else None
                        break
                    except Exception as e:  # keep trying
                        last_err = e
                        self.model = None
                if self.model is None and last_err:
                    raise last_err
            except Exception:
                self.model = None

        if self.model is None:
            # Fallback: no ASR available
            self.available = False
            return "", self.last_language
        else:
            self.available = True

        # faster-whisper expects float32 PCM in [-1, 1]
        # Ensure at least minimal duration before calling to avoid 'unavailable' and heavy VAD removal noise
        if audio.shape[0] < int(0.4 * sample_rate):
            return "", self.last_language
        kwargs = dict(
            beam_size=1,
            # Keep internal VAD disabled to avoid double-trimming; rely on our EnergyVAD
            vad_filter=False,
            language=self.last_language,
            temperature=0.0,
            initial_prompt=None,
        )
        # Some versions support condition_on_previous_text. Try, then fallback.
        try:
            try:
                segments, info = self.model.transcribe(audio, condition_on_previous_text=True, **kwargs)  # type: ignore
            except TypeError:
                segments, info = self.model.transcribe(audio, **kwargs)
        except Exception:
            # Any unexpected error: return empty safely
            return "", self.last_language
        text_parts = [seg.text for seg in segments]
        text = (" ".join(tp.strip() for tp in text_parts)).strip()
        if info is not None and getattr(info, "language", None):
            self.last_language = info.language

        if text:
            self._append_unique(text)

        return text, self.last_language


