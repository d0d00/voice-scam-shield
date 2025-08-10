from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import asyncio
import numpy as np
import time
import uuid

from utils.audio_buffers import SlidingWindowBuffer
from utils.vad import EnergyVAD
from pipeline.asr_stream import WhisperStreamer
from pipeline.intent import score_intent
from pipeline.antispoof import AASISTScorer
from pipeline.fuse import fuse_scores
from pipeline.diarization import OnlineDiarizer
from config import settings
import logging


app = FastAPI(title="Voice Scam Shield")
logger = logging.getLogger("vss")
logging.basicConfig(level=logging.INFO)

# Optional global components (initialized once)
DIARIZER = OnlineDiarizer(hf_token=settings.pyannote_token, window_seconds=5.0)
if DIARIZER and DIARIZER.available:
    logger.info("Diarization enabled (pyannote)")
else:
    logger.info("Diarization disabled or unavailable; using mixed audio")
SPOOF_SCORER = AASISTScorer(checkpoint_path=settings.aasist_checkpoint_path, device="cpu")
if SPOOF_SCORER and SPOOF_SCORER.available:
    logger.info("AASIST enabled (%s)", settings.aasist_checkpoint_path)
else:
    logger.info("AASIST unavailable; using fallback spoof score")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/enroll")
async def enroll_user(body: bytes):
    """Enroll local user voice for diarization.

    Body: raw PCM16 mono 16k bytes.
    """
    if not DIARIZER or not DIARIZER.available:
        return JSONResponse(status_code=503, content={"ok": False, "error": "diarization_unavailable"})
    samples = pcm16le_bytes_to_float32(body)
    ok = DIARIZER.enroll_user(samples, sample_rate=16000)
    return {"ok": bool(ok)}


# In-memory report store (simple, non-persistent)
REPORTS: dict[str, dict] = {}


def pcm16le_bytes_to_float32(data: bytes) -> np.ndarray:
    if not data:
        return np.zeros(0, dtype=np.float32)
    arr = np.frombuffer(data, dtype=np.int16)
    # Normalize to [-1, 1]
    return (arr.astype(np.float32) / 32768.0).astype(np.float32)


@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid.uuid4())
    session = {
        "id": session_id,
        "start": time.time(),
        "end": None,
        "events": [],
        "transcript": "",
        "lang": None,
        "last_label": "SAFE",
    }
    # Send session id to client
    await ws.send_json({"session_id": session_id})
    buffer = SlidingWindowBuffer(capacity_samples=16000 * 6)  # 6s buffer
    vad = EnergyVAD(sample_rate=16000, frame_ms=20.0, threshold_db=-55.0, hangover_ms=300.0)
    asr = WhisperStreamer(model_size=settings.asr_model_size, device="cpu", compute_type="int8")
    # Warm-up ASR to initialize model early and avoid transient 'unavailable' logs
    try:
        _ = asr.transcribe_chunk(np.zeros(16000, dtype=np.float32), sample_rate=16000)
    except Exception:
        pass
    # Global DIARIZER and SPOOF_SCORER are initialized at startup

    # Streaming accumulation/smoothing state
    ema_intent = 0.0
    ema_spoof = 0.0
    ema_risk = 0.0
    sticky_intent = 0.0
    sticky_spoof = 0.0
    # Update every 0.5s, use gentle decay so evidence persists across the call
    DECAY_PER_TICK = 0.98  # ~17s half-life at 0.5s tick
    ALPHA_INTENT = 0.4
    ALPHA_SPOOF = 0.3
    ALPHA_RISK = 0.3
    last_intent_eval_len = 0
    last_intent_score = 0.0
    last_intent_tags: list[str] = []

    last_rx_level = 0.0
    frames_received = 0
    first_frame_logged = False

    async def emit_loop():
        nonlocal ema_intent, ema_spoof, ema_risk, sticky_intent, sticky_spoof
        nonlocal last_intent_eval_len, last_intent_score, last_intent_tags
        # Emit status every 500ms
        try:
            while True:
                await asyncio.sleep(0.5)
                # Heartbeat/status to ensure client sees periodic messages even if downstream fails
                try:
                    await ws.send_json({
                        "session_id": session_id,
                        "tick": True,
                        "rx_level": float(last_rx_level),
                        "buffer_size": int(buffer.size()),
                        "frames_received": int(frames_received),
                    })
                except Exception:
                    pass
                try:
                    recent = buffer.get_recent(16000 * 1)  # last 1s
                    # For demo reliability, treat any non-empty audio as active if VAD says False but we have samples
                    is_active = vad.is_speech(recent) or (recent.size > 0 and np.max(np.abs(recent)) > 1e-4)
                    heuristics = 0.1 if is_active else 0.0

                    # ASR on last 3s window for more reliable decoding in demo conditions
                    recent_asr = buffer.get_recent(16000 * 3)
                    # Use diarization if available to focus on caller speech
                    if DIARIZER and DIARIZER.available:
                        dia_audio, _ = DIARIZER.select_caller(recent_asr, sample_rate=16000)
                    else:
                        dia_audio = recent_asr
                    # Always run ASR (we smooth results downstream)
                    text, lang = asr.transcribe_chunk(dia_audio, sample_rate=16000)
                    if not asr.available:
                        logger.info("ASR unavailable; using empty transcript (fallback)")
                    elif asr.fallback_used:
                        logger.info("ASR compute_type fallback in use: %s", asr.fallback_used)

                    # Intent (optionally refined by LLM) over full call context, not just recent fragment
                    full_text = asr.partial_transcript or text
                    # Re-evaluate intent only when transcript grows materially
                    if full_text and len(full_text) >= (last_intent_eval_len or 0) + 40:
                        intent_res = score_intent(full_text, api_key=settings.openai_api_key)
                        last_intent_eval_len = len(full_text)
                        last_intent_score = float(intent_res.score)
                        last_intent_tags = list(intent_res.tags)
                    else:
                        from types import SimpleNamespace
                        intent_res = SimpleNamespace(score=(last_intent_score or 0.0), tags=(last_intent_tags or []), rationale="cached")  # type: ignore
                    # Use a 3s window for spoof; gate by speech activity to avoid drift during silence
                    recent_spoof = buffer.get_recent(16000 * 3)
                    spoof_in = dia_audio if (DIARIZER and DIARIZER.available) else recent_spoof
                    raw_spoof = SPOOF_SCORER.score(spoof_in) if SPOOF_SCORER and SPOOF_SCORER.available else 0.05
                    spoof = raw_spoof if is_active else 0.0

                    tags = []
                    if is_active:
                        tags.append("VAD_ACTIVE")
                    tags.extend(intent_res.tags)

                    # Update smoothers and sticky evidence accumulators
                    ema_intent = ALPHA_INTENT * float(intent_res.score) + (1.0 - ALPHA_INTENT) * (ema_intent or 0.0)
                    ema_spoof = ALPHA_SPOOF * float(spoof) + (1.0 - ALPHA_SPOOF) * (ema_spoof or 0.0)
                    sticky_intent = max(ema_intent, (sticky_intent or 0.0) * DECAY_PER_TICK)
                    sticky_spoof = max(ema_spoof, (sticky_spoof or 0.0) * DECAY_PER_TICK)

                    # Fuse using sticky values to accumulate risk across the conversation
                    fusion = fuse_scores(spoof=sticky_spoof, intent=sticky_intent, heuristics=heuristics, tags=tags)
                    # Smooth the displayed risk to avoid flicker
                    ema_risk = ALPHA_RISK * float(fusion.risk) + (1.0 - ALPHA_RISK) * (ema_risk or 0.0)

                    # Derive label from smoothed risk using same thresholds
                    if ema_risk < 0.35:
                        label = "SAFE"
                    elif ema_risk < 0.65:
                        label = "SUSPICIOUS"
                        
                    else:
                        label = "SCAM"

                    payload = {
                        "risk": float(ema_risk),
                        "spoof": float(sticky_spoof),
                        "intent": float(sticky_intent),
                        "heuristics": float(heuristics),
                        "label": label,
                        "rationale": f"intent={sticky_intent:.2f}, spoof={sticky_spoof:.2f}, heuristics={heuristics:.2f}",
                        "tags": fusion.tags,
                        "partial_transcript": asr.partial_transcript[-400:],
                        "lang": lang,
                        "asr_available": asr.available,
                        "asr_fallback_used": asr.fallback_used,
                        "diar_available": bool(DIARIZER and DIARIZER.available),
                        "session_id": session_id,
                        "rx_level": float(last_rx_level),
                        "buffer_size": int(buffer.size()),
                        "frames_received": int(frames_received),
                    }
                    await ws.send_json(payload)

                    # Update session
                    session["last_label"] = label
                    if lang:
                        session["lang"] = lang
                    session["transcript"] = asr.partial_transcript
                    session["events"].append({
                        "t": time.time(),
                        "risk": float(ema_risk),
                        "label": label,
                        "tags": fusion.tags,
                        "intent": float(sticky_intent),
                        "spoof": float(sticky_spoof),
                    })
                except Exception as e:
                    logger.exception("emit_loop error")
        except WebSocketDisconnect:
            return

    emit_task = asyncio.create_task(emit_loop())
    try:
        while True:
            # Expect 16kHz mono PCM16 frames from the browser
            frame = await ws.receive_bytes()
            samples = pcm16le_bytes_to_float32(frame)
            # Track receive level for diagnostics
            if samples.size > 0:
                last_rx_level = float(np.sqrt(float(np.mean(np.square(samples)))) + 1e-9)
                if not first_frame_logged and last_rx_level > 0:
                    logger.info("received first audio frame (rms=%.6f)", last_rx_level)
                    first_frame_logged = True
                frames_received += 1
            buffer.push(samples)
    except WebSocketDisconnect:
        emit_task.cancel()
        try:
            await emit_task
        except Exception:
            pass
        session["end"] = time.time()
        REPORTS[session_id] = session
        return


@app.get("/report/{session_id}")
def get_report(session_id: str):
    report = REPORTS.get(session_id)
    if not report:
        return JSONResponse(status_code=404, content={"error": "not found"})
    return report