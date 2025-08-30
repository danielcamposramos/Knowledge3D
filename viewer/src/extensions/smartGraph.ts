import * as THREE from 'three';

export class AISuggestionManager {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private overlay: HTMLDivElement | null = null;
  private canvas: HTMLCanvasElement;
  private records: ReadonlyArray<{ id: string; embedding: number[]; vector: [number,number,number]; metadata: any }> = [];

  constructor(scene: THREE.Scene, camera: THREE.PerspectiveCamera, canvas: HTMLCanvasElement) {
    this.scene = scene;
    this.camera = camera;
    this.canvas = canvas;
  }

  setRecords(records: ReadonlyArray<{ id: string; embedding: number[]; vector: [number,number,number]; metadata: any }>) {
    this.records = records;
  }

  async showSuggestions(node: { id: string; vector: [number,number,number]; metadata?: any }, onClick: (targetId: string) => void) {
    const pt = new THREE.Vector3(node.vector[0], node.vector[1], node.vector[2]);
    const screen = this.projectToScreen(pt);
    const resp = await this.mockApi(node);
    this.renderRadial(screen.x, screen.y, resp.suggestions, onClick);
  }

  hide() {
    if (this.overlay) { document.body.removeChild(this.overlay); this.overlay = null; }
  }

  private projectToScreen(p: THREE.Vector3): { x: number; y: number } {
    const rect = this.canvas.getBoundingClientRect();
    const v = p.clone().project(this.camera);
    const x = (v.x * 0.5 + 0.5) * rect.width + rect.left;
    const y = (-v.y * 0.5 + 0.5) * rect.height + rect.top;
    return { x, y };
  }

  private async mockApi(node: { id: string; metadata?: any }): Promise<{ suggestions: { id: string; label: string; reason: string }[] }> {
    // simulate latency and suggest by simple heuristics (label substrings or neighbors by id hash)
    await new Promise(r => setTimeout(r, 400));
    const label = (node.metadata?.label as string) || node.id;
    const candidates = this.records.slice(0, Math.min(this.records.length, 128));
    const picks = candidates.filter(r => r.id !== node.id).slice(0, 8).map(r => ({ id: r.id, label: (r.metadata?.label as string)||r.id, reason: `related to ${label}` }));
    return { suggestions: picks };
  }

  private renderRadial(cx: number, cy: number, items: { id: string; label: string; reason: string }[], onClick: (targetId: string) => void) {
    this.hide();
    const div = document.createElement('div');
    div.style.position = 'fixed';
    div.style.left = '0';
    div.style.top = '0';
    div.style.width = '100vw';
    div.style.height = '100vh';
    div.style.pointerEvents = 'none';
    div.style.zIndex = '2000';
    const radius = 90;
    const angleStep = (Math.PI * 2) / Math.max(1, items.length);
    items.forEach((it, i) => {
      const a = i * angleStep - Math.PI/2;
      const x = cx + Math.cos(a) * radius;
      const y = cy + Math.sin(a) * radius;
      const b = document.createElement('button');
      b.textContent = it.label;
      b.title = it.reason;
      b.style.position = 'absolute';
      b.style.left = `${Math.round(x)}px`;
      b.style.top = `${Math.round(y)}px`;
      b.style.transform = 'translate(-50%, -50%)';
      b.style.pointerEvents = 'auto';
      b.onclick = (e) => { e.stopPropagation(); onClick(it.id); this.hide(); };
      div.appendChild(b);
    });
    div.onclick = () => this.hide();
    document.body.appendChild(div);
    this.overlay = div;
  }
}

export class DynamicLayerManager {
  private source: ReadonlyArray<{ id: string; vector: [number,number,number]; color?: [number,number,number]; metadata: any }> = [];
  private enabled: Set<string> = new Set();
  private layers: Set<string> = new Set();

  setRecords(records: ReadonlyArray<{ id: string; vector: [number,number,number]; color?: [number,number,number]; metadata: any }>) {
    this.source = records;
    this.layers = new Set();
    for (const r of records) {
      const l = (r.metadata?.layer as string) || (Array.isArray(r.metadata?.tags) ? (r.metadata.tags[0] as string) : undefined);
      if (l) this.layers.add(l);
    }
    this.enabled = new Set(this.layers);
  }

  getLayers(): string[] { return Array.from(this.layers); }
  isEnabled(name: string): boolean { return this.enabled.has(name); }
  toggle(name: string, on?: boolean) { if (on === undefined) { if (this.enabled.has(name)) this.enabled.delete(name); else this.enabled.add(name); } else { if (on) this.enabled.add(name); else this.enabled.delete(name); } }

  buildGeometry(): THREE.BufferGeometry {
    const points = this.source.filter(r => {
      const l = (r.metadata?.layer as string) || (Array.isArray(r.metadata?.tags) ? (r.metadata.tags[0] as string) : undefined);
      return !l || this.enabled.has(l);
    });
    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);
    for (let i=0;i<points.length;i++){
      positions.set(points[i].vector, i*3);
      const c = points[i].color || [0.6,0.7,0.9];
      colors.set(c, i*3);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    return g;
  }
}

export class LODRenderer {
  private base: THREE.BufferGeometry;
  private material: THREE.PointsMaterial;
  private points: THREE.Points | null = null; // base points for intersections
  private pBase: THREE.Points | null = null;
  private pMid: THREE.Points | null = null;
  private pLow: THREE.Points | null = null;
  private low: THREE.BufferGeometry | null = null;
  private mid: THREE.BufferGeometry | null = null;
  private centroid: THREE.Vector3 = new THREE.Vector3();
  private lastLevel: 0|1|2 = 0;
  private pixNear = 6; // pixel radius threshold to use base
  private pixMid = 4;  // pixel radius threshold to use mid
  private hysteresis = 1.5; // pixel band to avoid flicker
  private fadeWidth = 2.0; // pixels for cross-fade

  constructor(base: THREE.BufferGeometry, material?: THREE.PointsMaterial) {
    this.base = base;
    this.material = material || new THREE.PointsMaterial({ size: 0.1, vertexColors: true });
    this.computeCentroid();
    this.prepareLevels();
  }

  attach(scene: THREE.Scene): THREE.Points {
    // Prepare materials per level (enable transparency for fades)
    const mBase = this.material.clone(); mBase.transparent = true; mBase.opacity = 1.0;
    const mMid  = this.material.clone(); mMid.transparent  = true; mMid.opacity  = 0.0;
    const mLow  = this.material.clone(); mLow.transparent  = true; mLow.opacity  = 0.0;
    this.pBase = new THREE.Points(this.base, mBase);
    this.pMid  = new THREE.Points(this.mid  || this.base, mMid);
    this.pLow  = new THREE.Points(this.low  || this.base, mLow);
    scene.add(this.pBase, this.pMid, this.pLow);
    // Keep reference for raycasting (base is fine for interaction)
    this.points = this.pBase;
    return this.points;
  }

  update(camera: THREE.PerspectiveCamera) {
    if (!this.pBase || !this.pMid || !this.pLow) return;
    // Screen-space pixel radius for bounding sphere
    const bs = (this.base.boundingSphere || this.computeCentroid()).clone();
    const d = camera.position.distanceTo(bs.center);
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const pixelsPerUnit = (0.5 * window.innerHeight) / Math.tan(fov / 2);
    const pixelRadius = (bs.radius / Math.max(1e-6, d)) * pixelsPerUnit * (window.devicePixelRatio || 1);
    // Determine target level with hysteresis on pixel thresholds
    let level: 0|1|2 = this.lastLevel;
    const nearHi = this.pixNear + this.hysteresis, nearLo = this.pixNear - this.hysteresis;
    const midHi  = this.pixMid  + this.hysteresis, midLo  = this.pixMid  - this.hysteresis;
    if (this.lastLevel === 0) {
      if (pixelRadius < nearLo) level = 1;
    } else if (this.lastLevel === 1) {
      if (pixelRadius < midLo) level = 2;
      else if (pixelRadius > nearHi) level = 0;
    } else { // lastLevel 2
      if (pixelRadius > midHi) level = 1;
    }
    // Cross-fade between adjacent levels over fadeWidth pixels
    const fade = (t: number) => Math.min(1, Math.max(0, t));
    let aBase = 0, aMid = 0, aLow = 0;
    if (pixelRadius >= this.pixNear) {
      aBase = 1; aMid = 0; aLow = 0;
      if (pixelRadius < this.pixNear + this.fadeWidth) {
        const k = fade((this.pixNear + this.fadeWidth - pixelRadius) / this.fadeWidth);
        aMid = 1 - k; aBase = k;
      }
      level = 0;
    } else if (pixelRadius >= this.pixMid) {
      aMid = 1; aBase = 0; aLow = 0;
      const upperBlend = this.pixNear;
      const lowerBlend = this.pixMid;
      if (pixelRadius > upperBlend - this.fadeWidth) {
        const k = fade((upperBlend - pixelRadius) / this.fadeWidth);
        aBase = 1 - k; aMid = k;
      } else if (pixelRadius < lowerBlend + this.fadeWidth) {
        const k = fade((pixelRadius - lowerBlend) / this.fadeWidth);
        aLow = 1 - k; aMid = k;
      }
      level = 1;
    } else {
      aLow = 1; aMid = 0; aBase = 0;
      if (pixelRadius > this.pixMid - this.fadeWidth) {
        const k = fade((pixelRadius - (this.pixMid - this.fadeWidth)) / this.fadeWidth);
        aMid = 1 - k; aLow = k;
      }
      level = 2;
    }
    this.lastLevel = level;
    (this.pBase.material as THREE.PointsMaterial).opacity = aBase;
    (this.pMid.material as THREE.PointsMaterial).opacity = aMid;
    (this.pLow.material as THREE.PointsMaterial).opacity = aLow;
  }

  setBase(geom: THREE.BufferGeometry) {
    this.base = geom;
    this.base.computeBoundingSphere();
    this.computeCentroid();
    this.prepareLevels();
    if (this.pBase) this.pBase.geometry = this.base;
    if (this.pMid)  this.pMid.geometry  = this.mid  || this.base;
    if (this.pLow)  this.pLow.geometry  = this.low  || this.base;
  }

  private computeCentroid() {
    const pos = this.base.getAttribute('position') as THREE.BufferAttribute;
    const n = pos.count;
    const c = new THREE.Vector3();
    for (let i=0;i<n;i++){ c.x += pos.getX(i); c.y += pos.getY(i); c.z += pos.getZ(i); }
    c.multiplyScalar(1/Math.max(1,n));
    this.centroid.copy(c);
  }

  private prepareLevels() {
    const pos = this.base.getAttribute('position') as THREE.BufferAttribute;
    const col = this.base.getAttribute('color') as THREE.BufferAttribute | null;
    const n = pos.count;
    const makeDecimated = (stride: number) => {
      const m = Math.max(1, Math.floor(n/stride));
      const positions = new Float32Array(m*3);
      const colors = col ? new Float32Array(m*3) : null;
      let k = 0;
      for (let i=0;i<n;i+=stride){
        positions[k*3+0] = pos.getX(i);
        positions[k*3+1] = pos.getY(i);
        positions[k*3+2] = pos.getZ(i);
        if (colors) {
          colors[k*3+0] = col!.getX(i);
          colors[k*3+1] = col!.getY(i);
          colors[k*3+2] = col!.getZ(i);
        }
        k++;
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      if (colors) g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      return g;
    };
    this.mid = makeDecimated(2);
    this.mid.computeBoundingSphere();
    this.low = makeDecimated(8);
    this.low.computeBoundingSphere();
  }
}

export default { AISuggestionManager, DynamicLayerManager, LODRenderer };
