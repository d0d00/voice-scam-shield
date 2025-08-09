from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

app = FastAPI(title="Voice Scam Shield")

@app.get("/health")
def health():
    return {"ok": True}

@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    try:
        # Expect small binary PCM frames from the browser
        while True:
            frame = await ws.receive_bytes()
            # TODO: push to ring buffer, run VAD/ASR/intent/spoof, then emit event
            await ws.send_json({"risk": 0.1, "spoof": 0.05, "intent": 0.1,
                                "rationale": "warmup", "tags":[]})
    except WebSocketDisconnect:
        pass