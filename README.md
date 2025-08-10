# Voice Scam Shield

Multilingual, low-latency call guardian that detects scam intent and synthetic voices in real-time and provides discreet alerts.

## Quickstart

### Prerequisites
- macOS/Linux, Python 3.12, Node.js 18+
- Poetry (`pipx install poetry`) for backend

### Backend (FastAPI + WebSocket)
```
cd backend
poetry install

# Optional: configure env in repo root `.env`
# AASIST_CHECKPOINT_PATH=backend/models/aasist_scripted.pt
# PYANNOTE_TOKEN=...

poetry run uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Notes:
- If AASIST checkpoint is absent, spoof score gracefully falls back to near-zero.
- If PyAnnote token is unset/unavailable, diarization is disabled and mixed audio is used.

### Frontend (Next.js)
```
cd frontend
npm ci
npm run dev
```
Open `http://localhost:3000`. Click “Start Mic”. You should see:
- Transcript streaming
- Risk bar and per-criterion bars (Intent, Synthetic Voice, Heuristics)

### Demo flow
- Speak normally to see baseline SAFE.
- Say phrases implying credential/OTP/payment requests to raise intent and overall risk.
- Spoof score updates only while speech is active to avoid drift during silence.

## Configuration
- `backend/config.py` reads from `.env` (at repo root):
  - `AASIST_CHECKPOINT_PATH` (optional): TorchScript checkpoint for AASIST.
  - `PYANNOTE_TOKEN` (optional): enables diarization.
  - `ASR_MODEL_SIZE`: faster-whisper model size (default `small`).

## Repo Layout
- `backend/`: FastAPI app, pipelines (`asr_stream.py`, `intent.py`, `antispoof.py`, `fuse.py`), utils.
- `frontend/`: Next.js app, AudioWorklet, streaming UI.
- `docs/`: context and progress.
- `scripts/`: model fetch utilities.

## Notes
- Latency target < 2s; streaming frames at 20ms, ASR windows ~3s for robustness.
- Intent is weighted heavier in risk fusion for demo clarity.


