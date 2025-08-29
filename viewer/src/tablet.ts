import * as THREE from 'three';

type TabletMode = 'ai' | 'human';

export type TabletStatus = {
  ws: 'connected' | 'disconnected' | 'error';
  queue: number;
  house?: string;
  nodes?: number;
  info?: string;
  mode?: TabletMode;
};

export class Tablet3D {
  public object: THREE.Mesh;
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private tex: THREE.CanvasTexture;
  private status: TabletStatus = { ws: 'disconnected', queue: 0, mode: 'ai' };
  private overlay: HTMLDivElement | null = null;

  constructor() {
    // screen canvas
    this.canvas = document.createElement('canvas');
    this.canvas.width = 768; // good resolution for crisp text
    this.canvas.height = 480;
    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('2D context not available');
    this.ctx = ctx;
    this.tex = new THREE.CanvasTexture(this.canvas);
    this.tex.minFilter = THREE.LinearFilter;
    this.tex.magFilter = THREE.LinearFilter;

    // tablet geometry (approx 16:10)
    const w = 1.2; // meters
    const h = w * (10 / 16);
    const geom = new THREE.PlaneGeometry(w, h);
    const mat = new THREE.MeshBasicMaterial({ map: this.tex, transparent: false });
    this.object = new THREE.Mesh(geom, mat);
    this.object.position.set(-1.2, 0.6, 0); // place to the left of origin
    this.object.rotation.y = 0.4;

    this.renderScreen();
  }

  setStatus(update: Partial<TabletStatus>) {
    this.status = { ...this.status, ...update };
    this.renderScreen();
  }

  toggleFocus() {
    if (this.overlay) { this.hideFocus(); return; }
    this.showFocus();
  }

  showFocus() {
    if (this.overlay) return;
    const div = document.createElement('div');
    div.style.position = 'fixed';
    div.style.left = '0';
    div.style.top = '0';
    div.style.width = '100vw';
    div.style.height = '100vh';
    div.style.background = 'rgba(0,0,0,0.9)';
    div.style.zIndex = '1000';
    div.style.color = '#fff';
    div.style.fontFamily = 'system-ui, sans-serif';
    div.style.display = 'flex';
    div.style.flexDirection = 'column';
    div.style.padding = '16px';

    const row = document.createElement('div');
    row.style.display = 'flex';
    row.style.justifyContent = 'space-between';
    row.style.alignItems = 'center';

    const title = document.createElement('div');
    title.textContent = `K3D Tablet — ${this.status.mode ?? 'ai'}`;
    title.style.fontSize = '18px';
    title.style.fontWeight = 'bold';
    row.appendChild(title);

    const controls = document.createElement('div');
    const btnExit = document.createElement('button');
    btnExit.textContent = 'Exit Focus';
    btnExit.onclick = () => this.hideFocus();
    controls.appendChild(btnExit);
    const btnInstall = document.createElement('button');
    btnInstall.textContent = 'Install App…';
    btnInstall.style.marginLeft = '8px';
    btnInstall.onclick = () => alert('App store placeholder — to be wired');
    controls.appendChild(btnInstall);
    row.appendChild(controls);

    const content = document.createElement('div');
    content.style.flex = '1';
    content.style.marginTop = '12px';
    content.style.overflow = 'auto';
    content.style.border = '1px solid #444';
    content.style.padding = '12px';
    content.style.background = '#111';

    const pre = document.createElement('pre');
    pre.textContent = this.describe();
    content.appendChild(pre);

    div.appendChild(row);
    div.appendChild(content);
    document.body.appendChild(div);
    this.overlay = div;
  }

  hideFocus() {
    if (!this.overlay) return;
    document.body.removeChild(this.overlay);
    this.overlay = null;
  }

  private describe(): string {
    const s = this.status;
    return [
      `mode: ${s.mode ?? 'ai'}`,
      `ws: ${s.ws}  queue: ${s.queue}`,
      `house: ${s.house ?? '—'}`,
      `nodes: ${s.nodes ?? '—'}`,
      `info: ${s.info ?? ''}`,
      '',
      'Apps:',
      '- Console (planned) — stream explain traces',
      '- Notes (planned) — scratchpad synced to house',
      '- Graph (planned) — tiny neighbor map of focus label',
    ].join('\n');
  }

  private renderScreen() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    ctx.fillStyle = '#101214';
    ctx.fillRect(0, 0, w, h);
    // header
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 24px system-ui, sans-serif';
    ctx.fillText('K3D Tablet', 20, 36);
    ctx.font = '14px system-ui, sans-serif';
    ctx.fillText(`mode: ${this.status.mode ?? 'ai'}`, 20, 60);
    // status
    const ws = this.status.ws;
    ctx.fillStyle = ws === 'connected' ? '#7CFC00' : (ws === 'error' ? '#ff6b6b' : '#ffd166');
    ctx.fillRect(20, 76, 12, 12);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(`ws: ${ws}  queue: ${this.status.queue}`, 40, 86);
    // house line
    ctx.fillText(`house: ${this.status.house ?? '—'}`, 20, 110);
    ctx.fillText(`nodes: ${this.status.nodes ?? '—'}`, 20, 132);
    // info box
    ctx.fillStyle = '#182026';
    ctx.fillRect(20, 150, w - 40, h - 170);
    ctx.strokeStyle = '#2c3e50';
    ctx.strokeRect(20, 150, w - 40, h - 170);
    ctx.fillStyle = '#e0e0e0';
    ctx.font = '13px system-ui, sans-serif';
    const info = this.describe().split('\n');
    let y = 170;
    for (const line of info.slice(0, 12)) {
      ctx.fillText(line, 28, y);
      y += 18;
    }
    this.tex.needsUpdate = true;
  }
}

