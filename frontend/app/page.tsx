"use client";
import { useEffect, useRef, useState } from "react";

export default function Home() {
  const wsRef = useRef<WebSocket | null>(null);
  const [risk, setRisk] = useState(0);
  const [status, setStatus] = useState("disconnected");

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/audio");
    wsRef.current = ws;
    ws.onopen = () => setStatus("connected");
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (typeof msg.risk === "number") setRisk(Math.round(msg.risk * 100));
      } catch {}
    };
    ws.onclose = () => setStatus("disconnected");
    return () => ws.close();
  }, []);

  // For now, send fake “audio” bytes every 500ms
  useEffect(() => {
    const id = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        // placeholder bytes; replace with PCM from WebAudio later
        const buf = new Uint8Array([0,1,2,3]);
        wsRef.current.send(buf);
      }
    }, 500);
    return () => clearInterval(id);
  }, []);

  return (
    <main className="p-8">
      <h1 className="text-2xl font-semibold mb-4">Voice Scam Shield</h1>
      <div className="mb-2">WebSocket: <b>{status}</b></div>
      <div className="mb-4">
        Risk: <b>{risk}</b>/100
        <div className="w-64 h-4 bg-gray-200 rounded">
          <div
            className="h-4 bg-red-500 rounded"
            style={{ width: `${risk}%` }}
          />
        </div>
      </div>
      <p className="text-sm text-gray-600">Hooked to backend /ws/audio. Replace dummy bytes with mic PCM soon.</p>
    </main>
  );
}