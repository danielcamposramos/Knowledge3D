import type { RpnOpHandler, RpnValue } from './engine';
import { popNumber } from './engine';

export interface AudioBufferLike {
  duration: number;
  sampleRate: number;
  numberOfChannels: number;
  length: number;
  getChannelData(channel: number): Float32Array;
}

class SimpleAudioBuffer implements AudioBufferLike {
  public duration: number;
  public sampleRate: number;
  public numberOfChannels: number;
  public length: number;
  private channels: Float32Array[];

  constructor(sampleRate: number, channels: Float32Array[]) {
    this.sampleRate = Math.max(1, Math.floor(sampleRate));
    this.channels = channels.map((channel) => new Float32Array(channel));
    this.numberOfChannels = this.channels.length;
    this.length = this.channels[0]?.length ?? 0;
    this.duration = this.length / this.sampleRate;
  }

  getChannelData(channel: number): Float32Array {
    return this.channels[channel] ?? new Float32Array(this.length);
  }
}

function isAudioBufferLike(value: unknown): value is AudioBufferLike {
  return typeof value === 'object'
    && value !== null
    && 'duration' in value
    && 'sampleRate' in value
    && 'numberOfChannels' in value
    && 'length' in value
    && 'getChannelData' in value;
}

function popAudioBuffer(stack: RpnValue[]): AudioBufferLike {
  const value = stack.pop();
  if (!isAudioBufferLike(value)) {
    throw new Error('Expected audio buffer');
  }
  return value;
}

function popStringLike(stack: RpnValue[]): string {
  const value = stack.pop();
  if (value === undefined) {
    throw new Error('RPN underflow');
  }
  if (typeof value === 'number') {
    return String(value);
  }
  if (typeof value === 'object' && value !== null && 'str' in value) {
    return String((value as { str: string }).str);
  }
  throw new Error('Expected string-like value');
}

function cloneBuffer(buffer: AudioBufferLike): SimpleAudioBuffer {
  const channels: Float32Array[] = [];
  for (let index = 0; index < buffer.numberOfChannels; index += 1) {
    channels.push(new Float32Array(buffer.getChannelData(index)));
  }
  return new SimpleAudioBuffer(buffer.sampleRate, channels);
}

function createAudioBuffer(sampleRate: number, channels: Float32Array[]): AudioBufferLike {
  return new SimpleAudioBuffer(sampleRate, channels);
}

function monoMix(buffer: AudioBufferLike): AudioBufferLike {
  if (buffer.numberOfChannels <= 1) {
    return cloneBuffer(buffer);
  }
  const mixed = new Float32Array(buffer.length);
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    const data = buffer.getChannelData(channel);
    for (let index = 0; index < buffer.length; index += 1) {
      mixed[index] += data[index] / buffer.numberOfChannels;
    }
  }
  return createAudioBuffer(buffer.sampleRate, [mixed]);
}

function peakAmplitude(buffer: AudioBufferLike): number {
  let peak = 0;
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    const data = buffer.getChannelData(channel);
    for (let index = 0; index < data.length; index += 1) {
      const amplitude = Math.abs(data[index]);
      if (amplitude > peak) {
        peak = amplitude;
      }
    }
  }
  return peak;
}

function rmsEnergy(buffer: AudioBufferLike): number {
  let sum = 0;
  let count = 0;
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    const data = buffer.getChannelData(channel);
    for (let index = 0; index < data.length; index += 1) {
      sum += data[index] * data[index];
      count += 1;
    }
  }
  return count > 0 ? Math.sqrt(sum / count) : 0;
}

function resample(buffer: AudioBufferLike, sampleRate: number): AudioBufferLike {
  const targetRate = Math.max(1, Math.floor(sampleRate));
  if (targetRate === buffer.sampleRate) {
    return cloneBuffer(buffer);
  }
  const targetLength = Math.max(1, Math.round(buffer.duration * targetRate));
  const channels: Float32Array[] = [];
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    const input = buffer.getChannelData(channel);
    const output = new Float32Array(targetLength);
    for (let index = 0; index < targetLength; index += 1) {
      const position = (index * (input.length - 1)) / Math.max(1, targetLength - 1);
      const left = Math.floor(position);
      const right = Math.min(input.length - 1, left + 1);
      const mix = position - left;
      output[index] = input[left] * (1 - mix) + input[right] * mix;
    }
    channels.push(output);
  }
  return createAudioBuffer(targetRate, channels);
}

function applyGain(buffer: AudioBufferLike, gainScale: number): AudioBufferLike {
  const channels: Float32Array[] = [];
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    const input = buffer.getChannelData(channel);
    const output = new Float32Array(input.length);
    for (let index = 0; index < input.length; index += 1) {
      output[index] = input[index] * gainScale;
    }
    channels.push(output);
  }
  return createAudioBuffer(buffer.sampleRate, channels);
}

function trim(buffer: AudioBufferLike, startSec: number, endSec: number): AudioBufferLike {
  const startIndex = Math.max(0, Math.floor(startSec * buffer.sampleRate));
  const endIndex = Math.max(startIndex, Math.min(buffer.length, Math.floor(endSec * buffer.sampleRate)));
  const channels: Float32Array[] = [];
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    channels.push(buffer.getChannelData(channel).slice(startIndex, endIndex));
  }
  return createAudioBuffer(buffer.sampleRate, channels);
}

function concatBuffers(a: AudioBufferLike, b: AudioBufferLike): AudioBufferLike {
  const channelCount = Math.max(a.numberOfChannels, b.numberOfChannels);
  const channels: Float32Array[] = [];
  for (let channel = 0; channel < channelCount; channel += 1) {
    const left = channel < a.numberOfChannels ? a.getChannelData(channel) : new Float32Array(a.length);
    const right = channel < b.numberOfChannels ? b.getChannelData(channel) : new Float32Array(b.length);
    const output = new Float32Array(left.length + right.length);
    output.set(left, 0);
    output.set(right, left.length);
    channels.push(output);
  }
  return createAudioBuffer(a.sampleRate, channels);
}

function applyFade(buffer: AudioBufferLike, durationSec: number, mode: 'in' | 'out'): AudioBufferLike {
  const fadeSamples = Math.max(1, Math.floor(durationSec * buffer.sampleRate));
  const channels: Float32Array[] = [];
  for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
    const input = buffer.getChannelData(channel);
    const output = new Float32Array(input);
    if (mode === 'in') {
      const limit = Math.min(fadeSamples, output.length);
      for (let index = 0; index < limit; index += 1) {
        output[index] *= index / Math.max(1, limit - 1);
      }
    } else {
      const limit = Math.min(fadeSamples, output.length);
      for (let index = 0; index < limit; index += 1) {
        const sourceIndex = output.length - limit + index;
        output[sourceIndex] *= 1 - (index / Math.max(1, limit - 1));
      }
    }
    channels.push(output);
  }
  return createAudioBuffer(buffer.sampleRate, channels);
}

function mixBuffers(a: AudioBufferLike, b: AudioBufferLike, ratio: number): AudioBufferLike {
  const channelCount = Math.max(a.numberOfChannels, b.numberOfChannels);
  const length = Math.max(a.length, b.length);
  const mixRatio = Math.max(0, Math.min(1, ratio));
  const channels: Float32Array[] = [];
  for (let channel = 0; channel < channelCount; channel += 1) {
    const left = channel < a.numberOfChannels ? a.getChannelData(channel) : new Float32Array(a.length);
    const right = channel < b.numberOfChannels ? b.getChannelData(channel) : new Float32Array(b.length);
    const output = new Float32Array(length);
    for (let index = 0; index < length; index += 1) {
      const leftValue = index < left.length ? left[index] : 0;
      const rightValue = index < right.length ? right[index] : 0;
      output[index] = leftValue * (1 - mixRatio) + rightValue * mixRatio;
    }
    channels.push(output);
  }
  return createAudioBuffer(a.sampleRate, channels);
}

export const audioOps: Record<string, RpnOpHandler> = {
  AUDIO_LOAD: (stack, context) => {
    const source = popStringLike(stack);
    const buffer = context.audio?.load?.(source);
    if (!isAudioBufferLike(buffer)) {
      throw new Error('AUDIO_LOAD requires context.audio.load(source)');
    }
    stack.push(buffer);
  },
  AUDIO_PROBE: (stack) => {
    const buffer = popAudioBuffer(stack);
    stack.push({
      duration: buffer.duration,
      sampleRate: buffer.sampleRate,
      channels: buffer.numberOfChannels,
      length: buffer.length,
    });
  },
  AUDIO_RESAMPLE: (stack) => {
    const targetRate = popNumber(stack);
    const buffer = popAudioBuffer(stack);
    stack.push(resample(buffer, targetRate));
  },
  AUDIO_MONO: (stack) => {
    stack.push(monoMix(popAudioBuffer(stack)));
  },
  AUDIO_GAIN: (stack) => {
    const gainDb = popNumber(stack);
    const buffer = popAudioBuffer(stack);
    stack.push(applyGain(buffer, Math.pow(10, gainDb / 20)));
  },
  AUDIO_NORMALIZE: (stack) => {
    const buffer = popAudioBuffer(stack);
    const peak = peakAmplitude(buffer);
    stack.push(peak > 0 ? applyGain(buffer, 1 / peak) : cloneBuffer(buffer));
  },
  AUDIO_TRIM: (stack) => {
    const endSec = popNumber(stack);
    const startSec = popNumber(stack);
    const buffer = popAudioBuffer(stack);
    stack.push(trim(buffer, startSec, endSec));
  },
  AUDIO_CONCAT: (stack) => {
    const b = popAudioBuffer(stack);
    const a = popAudioBuffer(stack);
    stack.push(concatBuffers(a, b));
  },
  AUDIO_FADE_IN: (stack) => {
    const durationSec = popNumber(stack);
    const buffer = popAudioBuffer(stack);
    stack.push(applyFade(buffer, durationSec, 'in'));
  },
  AUDIO_FADE_OUT: (stack) => {
    const durationSec = popNumber(stack);
    const buffer = popAudioBuffer(stack);
    stack.push(applyFade(buffer, durationSec, 'out'));
  },
  AUDIO_MIX: (stack) => {
    const ratio = popNumber(stack);
    const b = popAudioBuffer(stack);
    const a = popAudioBuffer(stack);
    stack.push(mixBuffers(a, b, ratio));
  },
  AUDIO_PEAK: (stack) => {
    stack.push(peakAmplitude(popAudioBuffer(stack)));
  },
  AUDIO_RMS: (stack) => {
    stack.push(rmsEnergy(popAudioBuffer(stack)));
  },
  AUDIO_DURATION: (stack) => {
    stack.push(popAudioBuffer(stack).duration);
  },
  AUDIO_PLAY: (stack, context) => {
    const buffer = popAudioBuffer(stack);
    const source = context.audio?.play?.(buffer) ?? null;
    if (context.audio) {
      context.audio.currentSource = source ?? null;
    }
  },
  AUDIO_STOP: (_stack, context) => {
    context.audio?.currentSource?.stop?.();
    if (context.audio) {
      context.audio.currentSource = null;
    }
  },
};

export { createAudioBuffer };
