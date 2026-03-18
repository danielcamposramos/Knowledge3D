import { audioOps, createAudioRpnEngine } from '../src/rpn';
import { createAudioBuffer } from '../src/rpn/audioOps';

describe('audioOps', () => {
  test('AUDIO_PROBE pushes metadata', () => {
    const buf = createAudioBuffer(16000, [new Float32Array(80000)]);
    const stack = [buf];
    audioOps.AUDIO_PROBE(stack, { matrixStack: [] });
    expect(stack.pop()).toEqual({
      duration: 5,
      sampleRate: 16000,
      channels: 1,
      length: 80000,
    });
  });

  test('AUDIO_PEAK finds maximum amplitude', () => {
    const data = new Float32Array([0.1, -0.5, 0.3, -0.8, 0.2]);
    const buf = createAudioBuffer(16000, [data]);
    const stack = [buf];
    audioOps.AUDIO_PEAK(stack, { matrixStack: [] });
    expect(stack.pop()).toBeCloseTo(0.8);
  });

  test('AUDIO_RMS computes root mean square', () => {
    const data = new Float32Array([1, 1, 1, 1]);
    const buf = createAudioBuffer(16000, [data]);
    const stack = [buf];
    audioOps.AUDIO_RMS(stack, { matrixStack: [] });
    expect(stack.pop()).toBeCloseTo(1.0);
  });

  test('AUDIO_DURATION pushes seconds', () => {
    const buf = createAudioBuffer(8000, [new Float32Array(100000)]);
    const stack = [buf];
    audioOps.AUDIO_DURATION(stack, { matrixStack: [] });
    expect(stack.pop()).toBe(12.5);
  });

  test('audio engine registers analysis ops', () => {
    const engine = createAudioRpnEngine();
    engine.context.audio = {
      load: () => createAudioBuffer(16000, [new Float32Array(16000)]),
    };
    expect(engine.execute('STR_demo AUDIO_LOAD AUDIO_DURATION')).toEqual([1]);
  });
});
