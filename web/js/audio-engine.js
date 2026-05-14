/**
 * audio-engine.js — microphone capture using the Web Audio API.
 *
 * Captures mic input in real time and emits AudioFrame objects (raw
 * PCM samples + timestamp) to its onAudioFrame callback. The
 * PositionTracker consumes these.
 *
 * STATUS: working capture, no DSP yet. Phase 3 wires this output into
 * the chroma extractor + Whisper. For now the AudioFrame stream
 * exists but tracker.consume() ignores it.
 */

const FRAME_SIZE = 2048; // ~46ms at 44.1kHz; tradeoff: latency vs FFT resolution

export class AudioEngine {
  constructor() {
    this.running = false;
    this.onAudioFrame = null;
    this._ctx = null;
    this._stream = null;
    this._node = null;
  }

  async start() {
    if (this.running) return;

    this._stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,    // we want raw vocal, not VoIP-processed
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
      },
    });

    this._ctx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 16000, // Whisper expects 16kHz; resampled by the browser
      latencyHint: 'interactive',
    });
    const source = this._ctx.createMediaStreamSource(this._stream);

    // Use a ScriptProcessorNode for broad compatibility; the modern
    // alternative (AudioWorkletNode) is preferable in production -- it
    // runs off the main thread -- but ScriptProcessor is simpler to
    // ship as a single static file and works on iOS Safari today.
    this._node = this._ctx.createScriptProcessor(FRAME_SIZE, 1, 1);
    this._node.onaudioprocess = (e) => {
      if (!this.onAudioFrame) return;
      const input = e.inputBuffer.getChannelData(0);
      // Copy so the consumer can hold onto it past this callback.
      const samples = new Float32Array(input.length);
      samples.set(input);
      this.onAudioFrame({
        samples,
        sampleRate: this._ctx.sampleRate,
        timestamp: this._ctx.currentTime,
      });
    };
    source.connect(this._node);
    this._node.connect(this._ctx.destination);

    this.running = true;
  }

  async stop() {
    if (!this.running) return;
    this.running = false;
    if (this._node) { try { this._node.disconnect(); } catch {} this._node = null; }
    if (this._stream) {
      this._stream.getTracks().forEach(t => t.stop());
      this._stream = null;
    }
    if (this._ctx) { await this._ctx.close().catch(() => {}); this._ctx = null; }
  }
}
