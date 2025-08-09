Voice Scam Shield — Hackathon MVP Plan (Coding Context)

Objective

Build a multilingual AI call-guardian that:
	•	Detects scam intent and synthetic (AI) voices in real time
	•	Works on phone or video calls (Twilio, WebRTC, Zoom SDK)
	•	Supports EN, ES, FR (bonus: 1–2 more)
	•	Gives discreet, low-latency alerts (<2 s) without disrupting conversation

⸻

Architecture Overview

Frontend (React/Next.js)
	•	WebRTC stream → backend via WebSocket
	•	UI: risk dial, status chip (Safe / Suspicious / Scam), transcript highlights, event timeline
	•	Toggle: discreet alert on/off (e.g., whispered TTS)
	•	Post-call report page

Backend (Python/FastAPI)
	•	WebSocket API for streaming audio
	•	Processing pipeline:
	1.	VAD (Silero or WebRTC VAD) → Diarization (pyannote) to isolate caller
	2.	ASR: Whisper small/medium streaming (multilingual)
	3.	Scam intent detection:
	•	Option A: API LLM (few-shot multilingual scam classifier)
	•	Option B: Local multilingual transformer (XLM-R base) with zero/few-shot prompt templates
	4.	Synthetic voice detection: AASIST or RawNet2 (pretrained on ASVspoof 2019/2021)
	5.	Heuristic features: background noise absence, overly clean audio, repeated OTP/payment mentions
	6.	Risk fusion: weighted sum of spoof_score, intent_score, heuristics → classification
	7.	Emit JSON events every 0.5–1 s: {risk, spoof_score, intent_score, rationale, tags}

⸻

Detection Components

1. Anti-Spoofing (Synthetic Voice)
	•	Model: AASIST (PyTorch)
	•	Input: 3 s window, 1 s hop, 16 kHz mono caller audio
	•	Output: probability [0–1] synthetic

2. Scam Intent
	•	Input: incremental ASR transcript chunks
	•	Features: sensitive info requests, urgency, payment/OTP keywords
	•	Approach:
	•	Few-shot prompt for EN/ES/FR classification
	•	Tags: CREDENTIAL_REQUEST, OTP_REQUEST, PAYMENT, LINK
	•	Output: probability [0–1] scam intent

3. Heuristics
	•	Noise floor variance
	•	Non-speech spectral texture
	•	Lack of backchanneling (“uh-huh”, “mm-hmm”)
	•	Output: small adjustment to risk score

4. Risk Fusion Formula
	•	Example code
        ```python
        risk = 0.5 * spoof_score + 0.4 * intent_score + 0.1 * heuristics
        if risk < 0.35 → SAFE
        elif risk < 0.65 → SUSPICIOUS
        else → SCAM
        ```
	•	Include top reasons in rationale

⸻

Data Sources
	•	ASVspoof 2019/2021 (synthetic voice)
	•	WaveFake / FakeAVCeleb (voice clones)
	•	Custom scam script set (EN/ES/FR)
	•	Small sample bank for latency/threshold tuning

⸻

Repo Structure

    voice-scam-shield/
    backend/
        app.py
        pipeline/
        vad.py
        diarization.py
        asr_stream.py
        antispoof.py
        intent.py
        fuse.py
        models/
        aasist.ckpt
        utils/
        audio_buffers.py
        lang.py
    frontend/
        (Next.js + Tailwind UI)
    deployments/
        docker-compose.yml
        Dockerfile.backend
        Dockerfile.frontend
    scripts/
        fetch_models.py
        demo_assets/
    docs/
        quickstart.md

⸻

Implementation Timeline (12 h)

Hour 0–1 — Setup repo, Docker, FastAPI WebSocket, dummy UI
Hour 1–3 — VAD + diarization; audio buffering
Hour 3–5 — Whisper ASR (streaming), intent classification (few-shot prompt)
Hour 5–7 — Load AASIST, spoof scoring windows
Hour 7–8 — Risk fusion, rationale aggregation, frontend updates
Hour 8–9 — Discreet TTS alerts (ElevenLabs API)
Hour 9–10 — Post-call report endpoint
Hour 10–12 — Multilingual tests, latency profiling, demo polish

⸻

Key Python Dependencies
	•	fastapi, uvicorn, websockets
	•	torch, torchaudio
	•	whisper or faster-whisper
	•	pyannote.audio (optional diarization)
	•	numpy, librosa
	•	silero-vad
	•	Pretrained AASIST model
