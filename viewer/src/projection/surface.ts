import * as THREE from 'three';

export interface ProjectionConfig {
  anchor: THREE.Vector3;
  bounds: THREE.Vector3;
  mode: 'holographic' | 'stellarium';
}

function fitObjectToBounds(object: THREE.Object3D, bounds: THREE.Vector3): THREE.Object3D {
  const wrapper = new THREE.Group();
  wrapper.add(object);
  object.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(object);
  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  box.getSize(size);
  box.getCenter(center);

  const ratios = [bounds.x, bounds.y, bounds.z].map((limit, index) => {
    const component = size.getComponent(index);
    return component > 1e-6 ? limit / component : Number.POSITIVE_INFINITY;
  });
  const scale = Math.min(...ratios.filter(Number.isFinite));
  const safeScale = Number.isFinite(scale) && scale > 0 ? scale : 1;
  object.scale.multiplyScalar(safeScale);
  object.position.sub(center.multiplyScalar(safeScale));
  return wrapper;
}

function setObjectOpacity(object: THREE.Object3D, opacity: number): void {
  object.traverse((child: THREE.Object3D) => {
    const candidate = child as THREE.Mesh | THREE.Points | THREE.LineSegments;
    const materialValue = (candidate as any).material;
    if (!materialValue) {
      return;
    }
    const materials = Array.isArray(materialValue) ? materialValue : [materialValue];
    for (const material of materials) {
      if (!material) continue;
      material.transparent = opacity < 0.999 || material.transparent;
      material.opacity = opacity;
      material.needsUpdate = true;
    }
  });
}

export class ProjectionSurface {
  readonly group: THREE.Group;
  readonly config: ProjectionConfig;
  private contentRoot: THREE.Group;
  private content: THREE.Object3D | null = null;
  private targetOpacity = 0;
  private currentOpacity = 0;
  private glow: THREE.LineSegments | null = null;
  private _visible = false;

  constructor(config: ProjectionConfig) {
    this.config = config;
    this.group = new THREE.Group();
    this.group.position.copy(config.anchor);
    this.group.visible = false;
    this.contentRoot = new THREE.Group();
    this.group.add(this.contentRoot);

    if (config.mode === 'holographic') {
      const frame = new THREE.EdgesGeometry(
        new THREE.BoxGeometry(config.bounds.x, 0.02, config.bounds.z),
      );
      this.glow = new THREE.LineSegments(
        frame,
        new THREE.LineBasicMaterial({
          color: 0x00ddff,
          transparent: true,
          opacity: 0.0,
        }),
      );
      this.glow.position.set(0, -0.02, 0);
      this.group.add(this.glow);
    }
  }

  get visible(): boolean {
    return this._visible;
  }

  setContent(object: THREE.Object3D): void {
    this.clear();
    this.content = fitObjectToBounds(object, this.config.bounds);
    this.contentRoot.add(this.content);
    setObjectOpacity(this.content, this.currentOpacity);
    this.group.visible = this._visible || this.currentOpacity > 0.001;
  }

  clear(): void {
    if (this.content) {
      this.contentRoot.remove(this.content);
      this.content = null;
    }
  }

  show(): void {
    this._visible = true;
    this.targetOpacity = 1;
    this.group.visible = true;
  }

  hide(): void {
    this._visible = false;
    this.targetOpacity = 0;
  }

  update(delta: number): void {
    const blend = Math.min(1, Math.max(0, delta * 6));
    this.currentOpacity += (this.targetOpacity - this.currentOpacity) * blend;
    if (this.content) {
      setObjectOpacity(this.content, this.currentOpacity);
    }
    if (this.glow) {
      const glowMaterial = this.glow.material as THREE.LineBasicMaterial;
      glowMaterial.opacity = this.currentOpacity * 0.8;
      glowMaterial.needsUpdate = true;
    }
    if (!this._visible && this.currentOpacity < 0.01) {
      this.group.visible = false;
    } else if (this._visible) {
      this.group.visible = true;
    }
  }
}
