import * as THREE from 'three';
import { openStore } from './cache';
import { RPN } from './rpn';
import type { K3DRecord } from './loadK3D';

export interface TabletApp {
  id: string;
  title: string;
  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }): void;
  openOverlay(el: HTMLDivElement): void;
  onEvent?(ev: { type: string; payload?: any }): void;
  setContext?(ctx: { records: ReadonlyArray<K3DRecord> }): void;
}

export class ConsoleApp implements TabletApp {
  id = 'console';
  title = 'Console';
  private logs: string[] = [];
  push(line: string) {
    this.logs.push(line);
    if (this.logs.length > 500) this.logs.shift();
  }
  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }) {
    ctx.fillStyle = '#000000';
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.fillStyle = '#e0e0e0';
    ctx.font = '12px monospace';
    const lines = this.logs.slice(-12);
    let y = rect.y + 16;
    for (const ln of lines) {
      ctx.fillText(ln.slice(0, 100), rect.x + 8, y);
      y += 14;
    }
  }
  openOverlay(el: HTMLDivElement) {
    el.innerHTML = '';
    const pre = document.createElement('pre');
    pre.textContent = this.logs.join('\n');
    pre.style.whiteSpace = 'pre-wrap';
    pre.style.color = '#ddd';
    const btn = document.createElement('button');
    btn.textContent = 'Clear';
    btn.onclick = () => { this.logs = []; pre.textContent = ''; };
    el.appendChild(btn);
    el.appendChild(pre);
  }
}

export class NotesApp implements TabletApp {
  id = 'notes';
  title = 'Notes';
  private store = openStore<any>('k3d-tablet', 'notes');
  private notes: { id: number; text: string; ts: number }[] = [];
  private loaded = false;
  private async ensureLoaded() {
    if (this.loaded) return;
    const arr = (await this.store.get('all')) || [];
    this.notes = Array.isArray(arr) ? arr : [];
    this.loaded = true;
  }
  private async save() { await this.store.put('all', this.notes); }
  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }) {
    ctx.fillStyle = '#101010';
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.fillStyle = '#ffec99';
    ctx.fillRect(rect.x + 8, rect.y + 8, rect.w - 16, rect.h - 16);
    ctx.fillStyle = '#333';
    ctx.font = '12px system-ui';
    ctx.fillText(`Notes: ${this.notes.length}`, rect.x + 16, rect.y + 28);
  }
  async openOverlay(el: HTMLDivElement) {
    await this.ensureLoaded();
    el.innerHTML = '';
    const list = document.createElement('div');
    const ta = document.createElement('textarea');
    ta.rows = 6; ta.style.width = '100%'; ta.placeholder = 'Write a note...';
    const btnAdd = document.createElement('button'); btnAdd.textContent = 'Add';
    btnAdd.onclick = async () => {
      const text = ta.value.trim(); if (!text) return;
      this.notes.push({ id: Date.now(), text, ts: Date.now() }); ta.value = '';
      await this.save();
      renderList();
    };
    el.appendChild(ta); el.appendChild(btnAdd); el.appendChild(list);
    const renderList = () => {
      list.innerHTML = '';
      for (const n of this.notes.slice().reverse()) {
        const p = document.createElement('div');
        p.style.padding = '8px'; p.style.marginTop = '6px'; p.style.background = '#222'; p.style.color = '#ddd';
        p.textContent = new Date(n.ts).toLocaleString() + ': ' + n.text;
        list.appendChild(p);
      }
    };
    renderList();
  }
}

export class RpnApp implements TabletApp {
  id = 'rpn';
  title = 'RPN Calc';
  private rpn = new RPN();
  private stack: number[] = [];
  private push(x: number) { this.stack.push(x); if (this.stack.length > 32) this.stack.shift(); }
  private applyOp(op: string) {
    try {
      const res = this.rpn.eval([op]);
      if (typeof res === 'number') this.push(res);
    } catch {}
  }
  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }) {
    ctx.fillStyle = '#0b0d0f';
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.fillStyle = '#e0f7fa';
    ctx.font = '12px monospace';
    const view = this.stack.slice(-6);
    let y = rect.y + 18;
    for (let i = view.length - 1; i >= 0; i--) {
      ctx.fillText((view[i] as any).toString().slice(0, 16), rect.x + 10, y);
      y += 14;
    }
    ctx.fillStyle = '#90caf9';
    ctx.fillText('[RPN] keys in Focus mode', rect.x + 10, rect.y + rect.h - 8);
  }
  openOverlay(el: HTMLDivElement) {
    el.innerHTML = '';
    const input = document.createElement('input'); input.placeholder = 'number or op';
    const btnEnter = document.createElement('button'); btnEnter.textContent = 'Enter';
    btnEnter.onclick = () => { const v = parseFloat(input.value); if (!Number.isNaN(v)) this.push(v); input.value=''; render(); };
    const ops = ['+', '-', '*', '/', 'swap', 'drop', 'dup', 'sin', 'cos', 'tan', 'sqrt', 'pow'];
    const panel = document.createElement('div'); panel.style.display = 'flex'; panel.style.flexWrap = 'wrap'; panel.style.gap = '6px';
    for (const op of ops) { const b = document.createElement('button'); b.textContent = op; b.onclick = () => { this.applyOp(op); render(); }; panel.appendChild(b); }
    const out = document.createElement('pre'); out.style.color = '#ddd';
    const render = () => { out.textContent = this.stack.slice(-12).map((v,i)=>`${i}: ${v}`).reverse().join('\n'); };
    el.appendChild(input); el.appendChild(btnEnter); el.appendChild(panel); el.appendChild(out); render();
  }
}

export class WebApp implements TabletApp {
  id = 'web';
  title = 'Web';
  private store = openStore<any>('k3d-tablet', 'web');
  private lastUrl: string = '';
  private lastTitle: string = '';
  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }) {
    ctx.fillStyle = '#121212';
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.fillStyle = '#f5f5f5';
    ctx.font = '12px system-ui';
    const t = this.lastTitle || this.lastUrl || '(no page)';
    ctx.fillText(t.slice(0, 48), rect.x + 8, rect.y + 18);
  }
  async openOverlay(el: HTMLDivElement) {
    el.innerHTML = '';
    const url = document.createElement('input'); url.style.width='70%'; url.placeholder='https://…'; if (this.lastUrl) url.value = this.lastUrl;
    const go = document.createElement('button'); go.textContent = 'Go';
    const view = document.createElement('div'); view.style.marginTop = '8px'; view.style.background='#111'; view.style.color='#ddd'; view.style.padding='8px'; view.style.height='60vh'; view.style.overflow='auto';
    const fetchText = async (u: string) => {
      try {
        const res = await fetch(u, { mode: 'cors' });
        const html = await res.text();
        const text = html.replace(/<script[\s\S]*?<\/script>/g,'').replace(/<style[\s\S]*?<\/style>/g,'').replace(/<[^>]+>/g,'');
        view.textContent = text.slice(0, 8000);
        this.lastUrl = u; this.lastTitle = (html.match(/<title>([^<]+)<\/title>/i)?.[1] || '').trim();
        await this.store.put('last', { url: this.lastUrl, title: this.lastTitle, ts: Date.now() });
      } catch (e) {
        view.textContent = 'Fetch failed (CORS or offline).';
      }
    };
    go.onclick = () => { const u = url.value.trim(); if (!u) return; void fetchText(u); };
    el.appendChild(url); el.appendChild(go); el.appendChild(view);
    const saved = await this.store.get('last'); if (saved?.url) { this.lastUrl = saved.url; this.lastTitle = saved.title || ''; url.value = saved.url; }
  }
}

export class CalendarApp implements TabletApp {
  id = 'calendar';
  title = 'Calendar';
  private store = openStore<any>('k3d-tablet', 'calendar');
  private events: { id: number; title: string; start: string; end?: string; desc?: string }[] = [];
  private async load() { this.events = (await this.store.get('all')) || []; if (!Array.isArray(this.events)) this.events = []; }
  private async save() { await this.store.put('all', this.events); }
  async openOverlay(el: HTMLDivElement) {
    await this.load();
    el.innerHTML = '';
    const t = document.createElement('input'); t.placeholder = 'Title';
    const s = document.createElement('input'); s.placeholder = 'Start (YYYY-MM-DD)';
    const d = document.createElement('input'); d.placeholder = 'Description'; d.style.width = '60%';
    const add = document.createElement('button'); add.textContent = 'Add';
    const list = document.createElement('div'); list.style.marginTop = '8px';
    add.onclick = async () => { if (!t.value || !s.value) return; this.events.push({ id: Date.now(), title: t.value, start: s.value, desc: d.value }); await this.save(); render(); };
    el.appendChild(t); el.appendChild(s); el.appendChild(d); el.appendChild(add); el.appendChild(list);
    const render = () => { list.innerHTML = this.events.slice().reverse().map(e=>`${e.start} — ${e.title}${e.desc?': '+e.desc:''}`).join('\n'); list.style.whiteSpace='pre-wrap'; list.style.color='#ddd'; };
    render();
  }
  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }) {
    ctx.fillStyle = '#0e0f10'; ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.fillStyle = '#fff'; ctx.font = '12px system-ui';
    ctx.fillText(`Events: ${this.events.length}`, rect.x + 8, rect.y + 18);
  }
}

export class MailApp implements TabletApp {
  id = 'mail'; title = 'Email';
  private store = openStore<any>('k3d-tablet', 'mail');
  private msgs: { id: number; from: string; to: string; subj: string; body: string; ts: number }[] = [];
  private async load() { this.msgs = (await this.store.get('all')) || []; if (!Array.isArray(this.msgs)) this.msgs = []; }
  private async save() { await this.store.put('all', this.msgs); }
  async openOverlay(el: HTMLDivElement) {
    await this.load(); el.innerHTML='';
    const to = document.createElement('input'); to.placeholder='to';
    const subj = document.createElement('input'); subj.placeholder='subject'; subj.style.width='50%';
    const body = document.createElement('textarea'); body.rows=6; body.style.width='100%';
    const send = document.createElement('button'); send.textContent='Save Draft';
    const list = document.createElement('div'); list.style.marginTop='8px';
    send.onclick = async ()=>{ this.msgs.push({ id: Date.now(), from: 'me', to: to.value, subj: subj.value, body: body.value, ts: Date.now() }); await this.save(); render(); };
    el.appendChild(to); el.appendChild(subj); el.appendChild(send); el.appendChild(body); el.appendChild(list);
    const render = () => { list.innerHTML = this.msgs.slice().reverse().map(m=>`[${new Date(m.ts).toLocaleString()}] ${m.to}: ${m.subj}`).join('\n'); list.style.whiteSpace='pre-wrap'; list.style.color='#ddd'; };
    render();
  }
  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }) { ctx.fillStyle='#0a0a0a'; ctx.fillRect(rect.x, rect.y, rect.w, rect.h); ctx.fillStyle='#fff'; ctx.font='12px system-ui'; ctx.fillText(`Messages: ${this.msgs.length}`, rect.x+8, rect.y+18); }
}

export class EmbeddingsApp implements TabletApp {
  id = 'peek'; title = 'Embeddings Peek';
  private records: ReadonlyArray<K3DRecord> = [];
  private focus: string = '';
  private results: { label: string; sim: number }[] = [];
  setContext(ctx: { records: ReadonlyArray<K3DRecord> }) { this.records = ctx.records; }
  onEvent(ev: { type: string; payload?: any }) {
    if (ev.type === 'focus' && ev.payload?.label) {
      this.focus = String(ev.payload.label);
      this.compute();
    }
  }
  private cosine(a: number[], b: number[]): number {
    let dot = 0, na = 0, nb = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) { const x=a[i], y=b[i]; dot += x*y; na += x*x; nb += y*y; }
    return (na>0 && nb>0) ? (dot / (Math.sqrt(na)*Math.sqrt(nb))) : 0;
  }
  private compute() {
    if (!this.records.length || !this.focus) { this.results = []; return; }
    const idx = this.records.findIndex(r => ((r.metadata?.label as string) || r.id) === this.focus);
    if (idx < 0) { this.results = []; return; }
    const ref = this.records[idx].embedding as number[];
    const cap = Math.min(5000, this.records.length);
    const sims: { i: number; s: number }[] = [];
    for (let i = 0; i < cap; i++) {
      if (i === idx) continue;
      const s = this.cosine(ref, this.records[i].embedding as number[]);
      sims.push({ i, s });
    }
    sims.sort((a,b)=>b.s-a.s);
    this.results = sims.slice(0, 16).map(x => ({ label: (this.records[x.i].metadata?.label as string) || this.records[x.i].id, sim: x.s }));
  }
  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }) {
    ctx.fillStyle = '#0a0d10'; ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.fillStyle = '#fff'; ctx.font = '12px system-ui';
    ctx.fillText(`Focus: ${(this.focus||'—').slice(0,48)}`, rect.x+8, rect.y+18);
    let y = rect.y + 36;
    ctx.font = '12px monospace';
    for (const r of this.results) { ctx.fillText(`${r.sim.toFixed(3)}  ${r.label.slice(0,48)}`, rect.x+8, y); y += 14; }
  }
  openOverlay(el: HTMLDivElement) {
    el.innerHTML='';
    const input = document.createElement('input'); input.placeholder = 'label or id'; input.style.width='60%'; if (this.focus) input.value = this.focus;
    const btn = document.createElement('button'); btn.textContent = 'Peek';
    const out = document.createElement('pre'); out.style.color='#ddd'; out.style.whiteSpace='pre-wrap'; out.style.marginTop='8px';
    const go = () => { this.focus = input.value.trim(); this.compute(); out.textContent = this.results.map(r=>`${r.sim.toFixed(3)}  ${r.label}`).join('\n'); };
    btn.onclick = go; el.appendChild(input); el.appendChild(btn); el.appendChild(out);
    if (this.focus) go();
  }
}

export class GraphApp implements TabletApp {
  id = 'graph'; title = 'Mini-Map';
  private records: ReadonlyArray<K3DRecord> = [];
  private focus: string = '';
  setContext(ctx: { records: ReadonlyArray<K3DRecord> }) { this.records = ctx.records; }
  onEvent(ev: { type: string; payload?: any }) { if (ev.type === 'focus' && ev.payload?.label) this.focus = String(ev.payload.label); }
  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }) {
    ctx.fillStyle = '#050608'; ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    if (!this.records.length) return;
    // project vectors to 2D by XY; normalize bounds
    const cap = Math.min(this.records.length, 4096);
    let minx=Infinity, miny=Infinity, maxx=-Infinity, maxy=-Infinity;
    for (let i=0;i<cap;i++){ const v=this.records[i].vector; if(!v) continue; if(v[0]<minx)minx=v[0]; if(v[0]>maxx)maxx=v[0]; if(v[1]<miny)miny=v[1]; if(v[1]>maxy)maxy=v[1]; }
    const sx = (x:number)=> rect.x + ((x-minx)/(maxx-minx||1)) * rect.w;
    const sy = (y:number)=> rect.y + ((y-miny)/(maxy-miny||1)) * rect.h;
    // draw points
    ctx.fillStyle = '#88aaff';
    for (let i=0;i<cap;i++){ const v=this.records[i].vector as [number,number,number]; const x=sx(v[0]), y=sy(v[1]); ctx.fillRect(x, y, 1, 1); }
    // highlight focus if present
    if (this.focus) {
      const i = this.records.findIndex(r => ((r.metadata?.label as string) || r.id) === this.focus);
      if (i>=0) { const v=this.records[i].vector as [number,number,number]; ctx.strokeStyle='#ffcc00'; ctx.strokeRect(sx(v[0])-3, sy(v[1])-3, 6, 6); }
    }
  }
  openOverlay(el: HTMLDivElement) {
    el.innerHTML=''; const p = document.createElement('p'); p.textContent='Mini-map of XY positions (cap 4096) with focus square.'; p.style.color='#ddd'; el.appendChild(p);
  }
}
