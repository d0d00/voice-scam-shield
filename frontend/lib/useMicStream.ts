import { useCallback, useEffect, useRef, useState } from "react";
import type { MutableRefObject } from "react";

// Linear resampler from arbitrary inputRate -> 16kHz
function resampleTo16k(input: Float32Array, inputRate: number): Float32Array {
  const targetRate = 16000;
  if (inputRate === targetRate) return input;
  const ratio = inputRate / targetRate;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcPos = i * ratio;
    const i0 = Math.floor(srcPos);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const t = srcPos - i0;
    out[i] = input[i0] * (1 - t) + input[i1] * t;
  }
  return out;
}

function floatToPCM16(float32: Float32Array): Uint8Array {
  const buffer = new ArrayBuffer(float32.length * 2);
  const view = new DataView(buffer);
  let offset = 0;
  for (let i = 0; i < float32.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, float32[i]));
    s = s < 0 ? s * 0x8000 : s * 0x7fff;
    view.setInt16(offset, s, true);
  }
  return new Uint8Array(buffer);
}

type UseMicStreamOptions = {
  wsRef?: MutableRefObject<WebSocket | null>;
  frameMs?: number; // preferred output frame size in ms at 16kHz
};

export function useMicStream({ wsRef, frameMs = 20 }: UseMicStreamOptions) {
  const [active, setActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [framesSent, setFramesSent] = useState(0);
  const [lastSentAt, setLastSentAt] = useState<number | null>(null);
  const [txLevel, setTxLevel] = useState(0);
  const workletReadyRef = useRef(false);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const inputRateRef = useRef<number>(48000);
  const batchBufferRef = useRef<Float32Array>(new Float32Array(0));

  const start = useCallback(async () => {
    if (active) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true }, video: false });

      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 48000 });
      audioCtxRef.current = ctx;
      inputRateRef.current = ctx.sampleRate;
      // Ensure context is running after user gesture
      if (ctx.state !== "running") {
        await ctx.resume();
      }

      const source = ctx.createMediaStreamSource(stream);
      // Try AudioWorklet first
      let usingWorklet = false;
      if (!workletReadyRef.current) {
        try {
          await ctx.audioWorklet.addModule("/worklets/pcm-processor.js");
          workletReadyRef.current = true;
        } catch (e: any) {
          // ignore; will fall back
        }
      }

      const frameSamples = Math.round(ctx.sampleRate * 0.02); // 20ms native frames
      let worklet: AudioWorkletNode | null = null;
      try {
        if (workletReadyRef.current) {
          worklet = new AudioWorkletNode(ctx, "pcm-processor", {
            processorOptions: { frameSamples },
          });
          workletRefCleanup(worklet);
          source.connect(worklet);
          worklet.connect(ctx.destination);
          usingWorklet = true;
        }
      } catch {
        usingWorklet = false;
        worklet = null;
      }

      const handleFrame = (frameF32In: Float32Array, inRate: number) => {
        const frameF32 = frameF32In;
        const sRate = inRate || inputRateRef.current;
        if (!frameF32 || frameF32.length === 0) return;

        // Resample to 16kHz
        const resampled = resampleTo16k(frameF32, sRate);

        // Batch into desired 16kHz frame size
        const desiredSamples = Math.round((frameMs / 1000) * 16000);
        const prev = batchBufferRef.current;
        const concat = new Float32Array(prev.length + resampled.length);
        concat.set(prev, 0);
        concat.set(resampled, prev.length);

        let offset = 0;
        while (concat.length - offset >= desiredSamples) {
          const chunk = concat.subarray(offset, offset + desiredSamples);
          // measure TX RMS for debugging
          let s = 0;
          for (let i = 0; i < chunk.length; i++) s += chunk[i] * chunk[i];
          const rms = Math.sqrt(s / chunk.length);
          setTxLevel(rms);
          const pcm = floatToPCM16(chunk);
          const currentWs = wsRef?.current;
          if (currentWs && currentWs.readyState === WebSocket.OPEN) {
            currentWs.send(pcm);
            setFramesSent((n) => n + 1);
            setLastSentAt(Date.now());
          }
          offset += desiredSamples;
        }

        const remaining = concat.length - offset;
        const next = new Float32Array(remaining);
        next.set(concat.subarray(offset));
        batchBufferRef.current = next;
      };

      if (usingWorklet && worklet) {
        worklet.port.onmessage = (event: MessageEvent) => {
          const { type, payload, sampleRate } = event.data || {};
          if (type !== "frame" || !payload) return;
          let frameF32: Float32Array | null = null;
          if (payload instanceof Float32Array) frameF32 = payload;
          else if (payload instanceof ArrayBuffer) frameF32 = new Float32Array(payload);
          else if (ArrayBuffer.isView(payload)) frameF32 = new Float32Array((payload as ArrayBufferView).buffer);
          if (!frameF32 || frameF32.length === 0) return;
          handleFrame(frameF32, (sampleRate as number) || inputRateRef.current);
        };
        workletNodeRef.current = worklet;
      } else {
        // Fallback: ScriptProcessorNode (deprecated, but reliable for Safari)
        const proc = ctx.createScriptProcessor(1024, 1, 1);
        proc.onaudioprocess = (ev: AudioProcessingEvent) => {
          const inputBuf = ev.inputBuffer.getChannelData(0);
          // Copy to avoid holding onto the internal buffer
          const copy = new Float32Array(inputBuf.length);
          copy.set(inputBuf);
          handleFrame(copy, ctx.sampleRate);
        };
        source.connect(proc);
        proc.connect(ctx.destination);
      }
      setActive(true);
    } catch (e: any) {
      console.error(e);
      setError(e?.message || "Failed to start mic stream");
      await stop();
    }
  }, [active, wsRef, frameMs]);

  const stop = useCallback(async () => {
    try {
      const ctx = audioCtxRef.current;
      workletNodeRef.current?.disconnect();
      workletNodeRef.current = null;
      if (ctx && ctx.state !== "closed") {
        await ctx.close();
      }
    } finally {
      audioCtxRef.current = null;
      batchBufferRef.current = new Float32Array(0);
      setActive(false);
    }
  }, []);

  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  return { active, error, start, stop, framesSent, lastSentAt, txLevel };
}

function workletRefCleanup(node: AudioWorkletNode) {
  node.onprocessorerror = (err: any) => {
    // eslint-disable-next-line no-console
    console.error("AudioWorklet processor error", err);
  };
}


