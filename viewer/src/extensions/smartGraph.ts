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
  private points: THREE.Points | null = null;
  private low: THREE.BufferGeometry | null = null;
  private mid: THREE.BufferGeometry | null = null;
  private centroid: THREE.Vector3 = new THREE.Vector3();

  constructor(base: THREE.BufferGeometry, material?: THREE.PointsMaterial) {
    this.base = base;
    this.material = material || new THREE.PointsMaterial({ size: 0.1, vertexColors: true });
    this.computeCentroid();
    this.prepareLevels();
  }

  attach(scene: THREE.Scene): THREE.Points {
    this.points = new THREE.Points(this.base, this.material);
    scene.add(this.points);
    return this.points;
  }

  update(camera: THREE.PerspectiveCamera) {
    if (!this.points) return;
    const d = camera.position.distanceTo(this.centroid);
    const targetGeom = d > 20 ? (this.low || this.base) : (d > 10 ? (this.mid || this.base) : this.base);
    if (this.points.geometry !== targetGeom) {
      this.points.geometry = targetGeom;
    }
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
    this.low = makeDecimated(8);
  }
}

export default { AISuggestionManager, DynamicLayerManager, LODRenderer };

