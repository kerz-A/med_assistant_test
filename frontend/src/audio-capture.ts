export type OnChunkCallback = (chunk: ArrayBuffer) => void;

export class AudioCapture {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private workletNode: AudioWorkletNode | null = null;
  private _onChunk: OnChunkCallback | null = null;
  private browserSampleRate = 48000;
  private readonly targetSampleRate = 16000;

  set onChunk(cb: OnChunkCallback) {
    this._onChunk = cb;
  }

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        channelCount: 1,
        sampleRate: { ideal: this.targetSampleRate },
      },
    });

    this.context = new AudioContext({ sampleRate: this.targetSampleRate });
    this.browserSampleRate = this.context.sampleRate;

    await this.context.audioWorklet.addModule("/audio-processor.js");

    this.sourceNode = this.context.createMediaStreamSource(this.stream);
    this.workletNode = new AudioWorkletNode(this.context, "audio-capture-processor");

    this.workletNode.port.onmessage = (event: MessageEvent) => {
      if (event.data.type === "audio-chunk") {
        const float32: Float32Array = event.data.samples;
        const resampled = this.resampleIfNeeded(float32);
        const int16 = this.float32ToInt16(resampled);
        this._onChunk?.(int16.buffer as ArrayBuffer);
      }
    };

    this.sourceNode.connect(this.workletNode);
    this.workletNode.connect(this.context.destination);
  }

  stop(): void {
    this.workletNode?.disconnect();
    this.sourceNode?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    void this.context?.close();
    this.workletNode = null;
    this.sourceNode = null;
    this.stream = null;
    this.context = null;
  }

  private resampleIfNeeded(samples: Float32Array): Float32Array {
    if (this.browserSampleRate === this.targetSampleRate) {
      return samples;
    }

    const ratio = this.browserSampleRate / this.targetSampleRate;
    const newLength = Math.round(samples.length / ratio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const srcIndex = i * ratio;
      const lo = Math.floor(srcIndex);
      const hi = Math.min(lo + 1, samples.length - 1);
      const frac = srcIndex - lo;
      result[i] = samples[lo] * (1 - frac) + samples[hi] * frac;
    }

    return result;
  }

  private float32ToInt16(float32: Float32Array): Int16Array {
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
      const clamped = Math.max(-1, Math.min(1, float32[i]));
      int16[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    }
    return int16;
  }
}
