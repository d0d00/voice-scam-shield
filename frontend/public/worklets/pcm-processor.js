// Minimal AudioWorkletProcessor that batches input into fixed-size frames
// and posts Float32Array buffers to the main thread.

class PCMProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.frameSamples = Math.max(1, opts.frameSamples || Math.round(sampleRate * 0.02)); // default 20ms
    this._buffer = new Float32Array(0);
  }

  process(inputs) {
    const input = inputs && inputs[0] && inputs[0][0];
    if (!input) return true;

    // Append to buffer
    const prev = this._buffer;
    const concatenated = new Float32Array(prev.length + input.length);
    concatenated.set(prev, 0);
    concatenated.set(input, prev.length);
    this._buffer = concatenated;

    // Flush full frames
    let offset = 0;
    while (this._buffer.length - offset >= this.frameSamples) {
      const frame = this._buffer.subarray(offset, offset + this.frameSamples);
      // Transfer ArrayBuffer for maximal compatibility across realms
      const copy = new Float32Array(frame.length);
      copy.set(frame);
      const ab = copy.buffer;
      this.port.postMessage({ type: 'frame', sampleRate, payload: ab }, [ab]);
      offset += this.frameSamples;
    }

    if (offset > 0) {
      const remaining = this._buffer.length - offset;
      const next = new Float32Array(remaining);
      next.set(this._buffer.subarray(offset));
      this._buffer = next;
    }

    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);


