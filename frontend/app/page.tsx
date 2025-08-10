"use client";
import { useEffect, useRef, useState } from "react";
import { useMicStream } from "../lib/useMicStream";
import { StatusChip } from "./components/StatusChip";
import { ProgressBar } from "./components/ProgressBar";

export default function Home() {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [label, setLabel] = useState<string>("SAFE");
  const [transcript, setTranscript] = useState<string>("");
  const [sessionId, setSessionId] = useState<string>("");
  const [intent, setIntent] = useState<number>(0);
  const [spoof, setSpoof] = useState<number>(0);
  const [heuristics, setHeuristics] = useState<number>(0);
  const [risk, setRisk] = useState<number>(0);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/audio");
    wsRef.current = ws;
    ws.onopen = () => setConnected(true);
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (typeof msg.session_id === "string" && !sessionId) setSessionId(msg.session_id);
        if (typeof msg.risk === "number") setRisk(Math.round(msg.risk * 100));
        if (typeof msg.intent === "number") setIntent(Math.round(msg.intent * 100));
        if (typeof msg.spoof === "number") setSpoof(Math.round(msg.spoof * 100));
        if (typeof msg.heuristics === "number") setHeuristics(Math.round(msg.heuristics * 100));
        if (typeof msg.label === "string") setLabel(msg.label);
        if (typeof msg.partial_transcript === "string") setTranscript(msg.partial_transcript);
      } catch {}
    };
    ws.onclose = () => setConnected(false);
    return () => ws.close();
  }, []);

  const { active, error, start, stop } = useMicStream({ wsRef, frameMs: 20 });

  const startDisabled = !connected;
  const statusColor = label === "SCAM" ? "bg-red-600" : label === "SUSPICIOUS" ? "bg-yellow-500" : "bg-emerald-600";

  return (
    <div className="min-h-screen bg-[#0b1220] text-white">
      <main className="max-w-5xl mx-auto px-6 py-8">
        <h2 className="text-3xl font-semibold mb-6">Call Status</h2>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-3">
            <StatusChip label={label} />
          </div>

          <div className="lg:col-span-3">
            <div className="mb-2 text-sm text-gray-300">Risk Score</div>
            <ProgressBar value={risk} color={statusColor} heightClass="h-2.5" />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>Low Risk</span>
              <span>{risk}</span>
            </div>
          </div>

          <div className="lg:col-span-3 mt-2">
            <div className="text-lg font-medium mb-3">Risk Breakdown</div>
            <div className="space-y-4 max-w-3xl">
              <div>
                <div className="flex justify-between text-sm text-gray-300 mb-1">
                  <span>Intent Risk</span>
                  <span>{intent}</span>
                </div>
                <ProgressBar value={intent} color="bg-amber-500" />
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-300 mb-1">
                  <span>Synthetic Voice Detection</span>
                  <span>{spoof}</span>
                </div>
                <ProgressBar value={spoof} color="bg-violet-500" />
              </div>
              <div>
                <div className="flex justify-between text-sm text-gray-300 mb-1">
                  <span>Heuristic Analysis</span>
                  <span>{heuristics}</span>
                </div>
                <ProgressBar value={heuristics} color="bg-sky-500" />
              </div>
            </div>
          </div>

          <div className="lg:col-span-3 mt-4">
            <div className="flex items-center gap-3">
              <button
                className={`px-4 py-2 rounded-md text-sm font-medium ${active ? "bg-gray-600" : "bg-blue-600 hover:bg-blue-500"}`}
                onClick={() => (active ? stop() : start())}
                disabled={startDisabled}
              >
                {active ? "Stop Recording" : "Start Recording"}
              </button>
              {error && <span className="text-red-400 text-sm">{error}</span>}
            </div>
          </div>

          <div className="lg:col-span-3 mt-6">
            <div className="text-lg font-medium mb-2">Real-time Transcript</div>
            <div className="text-sm bg-[#0f172a] border border-gray-800 rounded-lg p-4 min-h-24 whitespace-pre-wrap">
              {transcript || <span className="text-gray-500">Transcript will appear here as the conversation progresses...</span>}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}