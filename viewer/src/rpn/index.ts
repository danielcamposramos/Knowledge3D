import { audioOps } from './audioOps';
import { DomBuilder } from './domBuilder';
import { domOps } from './domOps';
import { RpnEngine } from './engine';
import { MeshBuilder, rpnToMesh } from './meshBuilder';
import { mat4Ops } from './mat4Ops';
import { meshOps } from './meshOps';
import { PathBuilder, rpnToSVG } from './pathBuilder';
import { pathOps } from './pathOps';
import { RPN } from './math';

export function createVisualRpnEngine(): RpnEngine {
  const engine = new RpnEngine();
  engine.context.path = new PathBuilder();
  engine.context.mesh = new MeshBuilder();
  engine.registerModule(mat4Ops);
  engine.registerModule(pathOps);
  engine.registerModule(meshOps);
  return engine;
}

export function createDomRpnEngine(): RpnEngine {
  const engine = new RpnEngine();
  engine.context.dom = new DomBuilder();
  engine.registerModule(domOps);
  return engine;
}

export function createAudioRpnEngine(): RpnEngine {
  const engine = new RpnEngine();
  engine.registerModule(audioOps);
  return engine;
}

export function createFullRpnEngine(): RpnEngine {
  const engine = new RpnEngine();
  engine.context.path = new PathBuilder();
  engine.context.mesh = new MeshBuilder();
  engine.context.dom = new DomBuilder();
  engine.registerModule(mat4Ops);
  engine.registerModule(pathOps);
  engine.registerModule(meshOps);
  engine.registerModule(domOps);
  engine.registerModule(audioOps);
  return engine;
}

export function rpnToDOM(program: string): HTMLElement {
  const engine = createDomRpnEngine();
  engine.execute(program);
  return engine.context.dom!.toDOM();
}

export { RpnEngine, PathBuilder, MeshBuilder, DomBuilder, RPN, audioOps, domOps, rpnToSVG, rpnToMesh };
export type { RPNOptions, RPNToken } from './math';
export type { RpnContext, RpnOpHandler, RpnValue } from './engine';
