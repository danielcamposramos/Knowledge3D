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

export { RpnEngine, PathBuilder, MeshBuilder, RPN, rpnToSVG, rpnToMesh };
export type { RPNOptions, RPNToken } from './math';
export type { RpnContext, RpnOpHandler, RpnValue } from './engine';
