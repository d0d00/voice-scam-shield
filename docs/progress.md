# Voice Scam Shield – Progress Log

- Date: now

## Completed
- Frontend: Mic capture hook `frontend/lib/useMicStream.ts` with robust resampling to 16 kHz, PCM16 framing, and WS streaming.
- Frontend: AudioWorklet `frontend/public/worklets/pcm-processor.js` for stable 20 ms frames.
- Frontend: UI updated to start/stop mic and stream to backend.
- Backend: Added `utils/audio_buffers.py` circular sliding buffer.
- Backend: Added `utils/vad.py` simple energy-based VAD with hangover.
- Backend: Wired `/ws/audio` to accept PCM16 frames, buffer them, run VAD, and emit periodic status with baseline risk fusion placeholder.
- Backend: Added `pipeline/asr_stream.py` (faster-whisper wrapper), `pipeline/intent.py` (heuristic EN/ES/FR tags + optional LLM refinement), `pipeline/antispoof.py` (placeholder heuristics), `pipeline/fuse.py` (risk fusion). Integrated into WS loop; emitting `partial_transcript`, `lang`, `tags`.
 - Frontend: Added `app/components/StatusChip.tsx`. UI shows label, tags, and partial transcript from WS events.
 - Backend: Config option `asr_model_size` in `backend/config.py`. WS now includes `asr_available` and `asr_fallback_used` to surface when fallback/no model is active; frontend logs a console warning only for developers.
 - Backend: ASR quality improvements: longer context (2s window) with overlap, deduped partial transcript assembly, language conditioning when available.
 - Backend: Integrated optional diarization (`pyannote`) to select dominant speaker for ASR/anti-spoof when token available.
 - Backend: Replaced anti-spoof heuristic with AASIST scorer placeholder class; ready to load checkpoint when available.
 - Backend: Added `scripts/fetch_models.py` to download AASIST TorchScript checkpoint; config `AASIST_CHECKPOINT_PATH` to enable. Backend logs whether ASR/diar/AASIST fallbacks are active.

- Demo readiness fixes:
  - Backend: Disabled Whisper internal VAD (`vad_filter=False`) to prevent removal of all audio causing empty transcripts; rely on our own VAD.
  - Frontend: Fixed mic streaming to WebSocket by passing a live `wsRef` into `useMicStream`, ensuring frames are sent after the socket connects.

- Backend: Added call-level accumulation and smoothing:
  - Intent now evaluated on full running transcript, not only last chunk, with cached re-eval to limit API usage.
  - Added exponential moving average and sticky (decaying max) accumulators for `intent` and `spoof` so evidence persists across the call.
  - Smoothed `risk` with EMA to reduce flicker; thresholds unchanged.
  - WS payload now includes `intent`, `spoof`, and `heuristics` fields alongside total `risk`.
- Frontend: Added separate mini bars for `Intent`, `Synthetic Voice`, and `Heuristics` below the total risk bar.
 - Frontend: Added developer diagnostics (WS msgs, TX/RX level, buffer size) for debugging during demo; can be hidden later.

## Next
- Stabilize: measure latency and CPU; adjust ASR model size if needed. Optional: add start/stop ASR toggle.
- Backend: Replace anti-spoof heuristics with AASIST when time allows; add `scripts/fetch_models.py`.
 - Evaluate diarization (pyannote) if audio contains multiple speakers and we need caller-only segments (token available).
 - Optional: Improve ASR latency without reducing model size (currently using `small` for transcript quality).
 - Provide AASIST checkpoint URL or place file at `backend/models/aasist_scripted.pt`; set `AASIST_CHECKPOINT_PATH` accordingly.
 - Add user enrollment step to diarizer for robust caller/user separation; currently uses automatic selection of non-user via similarity when enrolled.

- Fine-tune accumulation parameters for demo:
  - Adjust EMA alphas and decay to balance responsiveness vs. persistence.
  - Consider keyword windowing for "cool-down" on benign content instead of global decay.
- Frontend polish (after 1 & 2):
  - Prettier layout, color legend, responsive grid; add per-criterion numeric labels and tooltips with rationale/tags.

## Notes
- Keep 20 ms frames at 16 kHz for consistent downstream timing.
- Current VAD is energy-only; swap to silero for robustness if time allows.
- ASR transcript quality is currently not great (possibly fallback/no model). Re-evaluate after full implementation and consider upgrading model size or device, or optimizing chunking.
 - Accumulation design: sticky decay keeps risk from immediately dropping to 0 after a suspicious segment; full transcript context improves intent detection when user imitates calls.
 - Spoof gating: spoof scoring is applied only during active speech to avoid drift during silence.
 - Risk fusion: intent weighted heavier (0.6) than spoof (0.3) and heuristics (0.1) for demo clarity.
