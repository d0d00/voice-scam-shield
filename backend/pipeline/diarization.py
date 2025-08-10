from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch

try:
    from pyannote.audio import Pipeline  # type: ignore
except Exception:
    Pipeline = None  # type: ignore


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-9
    return v / n


class OnlineDiarizer:
    """Lightweight wrapper around pyannote speaker diarization.

    For streaming, we periodically run diarization on the last window_seconds
    of audio and select the dominant speaker in that window. We then return
    only that speaker's samples for downstream processing (ASR/spoof).
    """

    def __init__(self, hf_token: Optional[str], window_seconds: float = 5.0) -> None:
        self.available = False
        self.window_seconds = float(window_seconds)
        self._pipeline = None
        self._embedder = None
        self._user_embedding: Optional[np.ndarray] = None
        if hf_token and Pipeline is not None:
            try:
                # Recent versions use task-specific pipelines; fall back if needed
                # "pyannote/speaker-diarization" may require access.
                self._pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization", use_auth_token=hf_token
                )
                # Try to load a speaker embedding model for enrollment
                try:
                    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding  # type: ignore

                    self._embedder = PretrainedSpeakerEmbedding(
                        "speechbrain/spkrec-ecapa-voxceleb", device="cpu"
                    )
                except Exception:
                    self._embedder = None
                self.available = True
            except Exception:
                self._pipeline = None
                self.available = False

    def select_dominant_speaker(
        self, audio: np.ndarray, sample_rate: int
    ) -> Tuple[np.ndarray, Optional[str]]:
        if not self.available or audio.size == 0:
            return audio, None
        try:
            waveform = torch.from_numpy(audio).unsqueeze(0)  # 1 x T
            audio_file = {"waveform": waveform, "sample_rate": sample_rate}
            diarization = self._pipeline(audio_file)
            # Accumulate duration per speaker
            durations: dict[str, float] = {}
            for seg, _, speaker in diarization.itertracks(yield_label=True):
                durations[speaker] = durations.get(speaker, 0.0) + float(seg.duration)
            if not durations:
                return audio, None
            dominant = max(durations, key=durations.get)
            # Build mask for dominant speaker
            total_len = audio.shape[0]
            mask = np.zeros(total_len, dtype=np.float32)
            for seg, _, speaker in diarization.itertracks(yield_label=True):
                if speaker != dominant:
                    continue
                start = int(max(0, round(seg.start * sample_rate)))
                end = int(min(total_len, round(seg.end * sample_rate)))
                if end > start:
                    mask[start:end] = 1.0
            # Avoid empty mask
            if mask.sum() < sample_rate * 0.2:
                return audio, dominant
            return audio * mask, dominant
        except Exception:
            return audio, None

    def enroll_user(self, audio: np.ndarray, sample_rate: int) -> bool:
        """Create an enrollment embedding for the local user.

        Use a short clean mic-only segment. Returns True if enrollment stored.
        """
        if not self.available or self._embedder is None:
            return False
        try:
            wav = torch.from_numpy(audio).float().unsqueeze(0)
            if wav.shape[1] < int(sample_rate * 0.5):
                return False
            emb = self._embedder(wav)
            if hasattr(emb, "detach"):
                emb = emb.detach().cpu().numpy().squeeze()
            else:
                emb = np.array(emb).squeeze()
            self._user_embedding = _l2_normalize(emb.astype(np.float32))
            return True
        except Exception:
            return False

    def select_caller(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Optional[str]]:
        """Return the non-user speaker (caller) if user enrollment exists, else dominant speaker.
        """
        if not self.available:
            return audio, None
        try:
            waveform = torch.from_numpy(audio).unsqueeze(0)
            diarization = self._pipeline({"waveform": waveform, "sample_rate": sample_rate})
            # Compute embedding per speaker if user embedding is present
            if self._embedder is None or self._user_embedding is None:
                return self.select_dominant_speaker(audio, sample_rate)

            speakers = {}
            for seg, _, spk in diarization.itertracks(yield_label=True):
                start = int(max(0, round(seg.start * sample_rate)))
                end = int(min(audio.shape[0], round(seg.end * sample_rate)))
                if end <= start:
                    continue
                chunk = audio[start:end]
                wav = torch.from_numpy(chunk).float().unsqueeze(0)
                emb = self._embedder(wav)
                if hasattr(emb, "detach"):
                    emb = emb.detach().cpu().numpy().squeeze()
                else:
                    emb = np.array(emb).squeeze()
                speakers.setdefault(spk, []).append(_l2_normalize(emb.astype(np.float32)))

            if not speakers:
                return audio, None
            # Average embeddings per speaker
            spk_emb = {
                spk: _l2_normalize(np.mean(np.stack(v, axis=0), axis=0)) for spk, v in speakers.items()
            }
            # Pick the speaker with lowest cosine similarity to user embedding = caller
            ue = self._user_embedding
            scores = {spk: float(np.dot(emb, ue)) for spk, emb in spk_emb.items()}  # cosine since L2-normed
            caller = min(scores, key=scores.get)

            # Build mask for caller
            total_len = audio.shape[0]
            mask = np.zeros(total_len, dtype=np.float32)
            for seg, _, spk in diarization.itertracks(yield_label=True):
                if spk != caller:
                    continue
                start = int(max(0, round(seg.start * sample_rate)))
                end = int(min(total_len, round(seg.end * sample_rate)))
                if end > start:
                    mask[start:end] = 1.0
            if mask.sum() < sample_rate * 0.2:
                return audio, caller
            return audio * mask, caller
        except Exception:
            return audio, None


