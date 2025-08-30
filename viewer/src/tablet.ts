import * as THREE from 'three';
import { ConsoleApp, NotesApp, RpnApp, WebApp, CalendarApp, MailApp, EmbeddingsApp, GraphApp, GalaxyApp, StatsApp, LayersApp, DoorsApp, DiaryApp, ControlApp, type TabletApp } from './apps';

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
  private apps: TabletApp[] = [new ConsoleApp(), new NotesApp(), new RpnApp(), new WebApp(), new CalendarApp(), new MailApp(), new EmbeddingsApp(), new GraphApp(), new GalaxyApp(), new StatsApp(), new LayersApp(), new DoorsApp(), new DiaryApp(), new ControlApp()];
  private activeApp = 'console';
  private emitter: ((ev: { type: string; payload?: any; kind?: string }) => void) | null = null;
  private localHandler: ((ev: { type: string; payload?: any }) => void) | null = null;

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

  pushExplain(line: string) {
    const app = this.apps.find(a => a.id === 'console') as ConsoleApp | undefined;
    app?.push(line);
    this.renderScreen();
  }

  setDataset(records: ReadonlyArray<{ id: string; vector: [number,number,number]; embedding: number[]; metadata: Record<string, unknown> }>) {
    for (const app of this.apps) app.setContext?.({ records: records as any, publish: (ev) => this.publish(ev) });
  }

  setFocusLabel(label: string) {
    for (const app of this.apps) app.onEvent?.({ type: 'focus', payload: { label } });
    this.renderScreen();
  }

  private publish(ev: { type: string; payload?: any }) {
    // fan-out to apps
    for (const app of this.apps) app.onEvent?.(ev);
    // local handler for in-app actions (e.g., applyLayers)
    try { this.localHandler?.(ev); } catch {}
    // emit upstream for live logging (e.g., agentic browser events)
    try {
      this.emitter?.({ ...ev, kind: ev.type });
    } catch {}
    this.renderScreen();
  }

  setEmitter(fn: (ev: { type: string; payload?: any; kind?: string }) => void) {
    this.emitter = fn;
  }

  setLocalHandler(fn: (ev: { type: string; payload?: any }) => void) {
    this.localHandler = fn;
  }

  // Allow external dispatch of tablet events (e.g., from commands)
  dispatch(ev: { type: string; payload?: any }) {
    this.publish(ev);
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
    // Tabs for apps
    const tabs = document.createElement('div');
    tabs.style.display = 'flex';
    tabs.style.gap = '6px';
    for (const app of this.apps) {
      const b = document.createElement('button');
      b.textContent = app.title;
      b.onclick = () => { this.activeApp = app.id; app.openOverlay(contentArea); this.renderScreen(); };
      tabs.appendChild(b);
    }
    const contentArea = document.createElement('div');
    contentArea.style.marginTop = '8px';
    contentArea.style.minHeight = '60vh';
    const initial = this.apps.find(a => a.id === this.activeApp) || this.apps[0];
    initial.openOverlay(contentArea);
    content.appendChild(tabs);
    content.appendChild(contentArea);

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
      ...this.apps.map(a => `- ${a.title}`),
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
    // tabs and app area on canvas
    // tabs
    ctx.fillStyle = '#2c3e50';
    ctx.fillRect(20, 146, w - 40, 24);
    ctx.fillStyle = '#ffffff';
    ctx.font = '13px system-ui, sans-serif';
    let x = 24;
    for (const app of this.apps) {
      const label = ` ${app.title} `;
      const m = ctx.measureText(label);
      const sel = app.id === this.activeApp;
      ctx.fillStyle = sel ? '#3a506b' : '#2c3e50';
      ctx.fillRect(x - 2, 146, m.width + 10, 24);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x, 162);
      x += m.width + 14;
    }
    // app area
    ctx.fillStyle = '#182026';
    ctx.fillRect(20, 174, w - 40, h - 194);
    ctx.strokeStyle = '#2c3e50';
    ctx.strokeRect(20, 174, w - 40, h - 194);
    const app = this.apps.find(a => a.id === this.activeApp);
    if (app) app.renderCanvas(ctx, { x: 24, y: 180, w: w - 48, h: h - 206 });
    this.tex.needsUpdate = true;
  }
}
