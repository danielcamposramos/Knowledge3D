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
  setContext?(ctx: { records: ReadonlyArray<K3DRecord>; publish?: (ev: { type: string; payload?: any }) => void }): void;
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
  title = 'Agentic Browser';
  private store = openStore<any>('k3d-tablet', 'web');
  private lastUrl: string = '';
  private lastTitle: string = '';
  private publish: ((ev: { type: string; payload?: any }) => void) | null = null;

  setContext(ctx: { records: ReadonlyArray<K3DRecord>; publish?: (ev: { type: string; payload?: any }) => void }) {
    this.publish = ctx.publish || null;
  }

  private log(kind: string, data: Record<string, unknown>) {
    try { this.publish?.({ type: kind, payload: data }); } catch {}
  }

  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }) {
    ctx.fillStyle = '#121212';
    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.fillStyle = '#f5f5f5';
    ctx.font = '12px system-ui';
    const t = this.lastTitle || this.lastUrl || '(no page)';
    ctx.fillText(t.slice(0, 48), rect.x + 8, rect.y + 18);
  }

  private async fetchDirect(u: string, iframe: HTMLIFrameElement, view: HTMLDivElement) {
    try {
      const res = await fetch(u, { mode: 'cors' });
      const html = await res.text();
      const text = html.replace(/<script[\s\S]*?<\/script>/g,'').replace(/<style[\s\S]*?<\/style>/g,'').replace(/<[^>]+>/g,'');
      iframe.style.display='none'; view.style.display='block';
      view.textContent = text.slice(0, 8000);
      this.lastUrl = u; this.lastTitle = (html.match(/<title>([^<]+)<\/title>/i)?.[1] || '').trim();
      await this.store.put('last', { url: this.lastUrl, title: this.lastTitle, ts: Date.now() });
      this.log('browser_visit', { engine: 'direct', url: u, title: this.lastTitle, len: text.length });
    } catch (e) {
      // Fallback: iframe (may be blocked by X-Frame-Options)
      iframe.style.display='block'; view.style.display='none'; iframe.src = u;
      this.lastUrl = u; this.lastTitle = '';
      await this.store.put('last', { url: this.lastUrl, title: this.lastTitle, ts: Date.now() });
      this.log('browser_iframe', { url: u });
    }
  }

  private async wikiSearch(q: string): Promise<{ title: string; snippet: string }[]> {
    const api = 'https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&origin=*';
    const url = `${api}&srsearch=${encodeURIComponent(q)}`;
    const res = await fetch(url);
    const data = await res.json();
    const arr = (data?.query?.search || []) as any[];
    return arr.map((it) => ({ title: it.title as string, snippet: (it.snippet || '').replace(/<[^>]+>/g,'') }));
  }

  private async wikiSummary(title: string): Promise<{ title: string; extract: string; url: string } | null> {
    const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(title)}?redirect=true`;
    const res = await fetch(url, { headers: { 'accept': 'application/json' } });
    if (!res.ok) return null;
    const data = await res.json();
    const extract = (data?.extract || '') as string;
    const pageUrl = (data?.content_urls?.desktop?.page || `https://en.wikipedia.org/wiki/${encodeURIComponent(title)}`) as string;
    return { title: data?.title || title, extract, url: pageUrl };
  }

  async openOverlay(el: HTMLDivElement) {
    el.innerHTML = '';
    // Controls
    const url = document.createElement('input'); url.style.width='50%'; url.placeholder='https://… or Wikipedia topic'; if (this.lastUrl) url.value = this.lastUrl;
    const go = document.createElement('button'); go.textContent = 'Go';
    const wikiBtn = document.createElement('button'); wikiBtn.textContent = 'Wikipedia Search'; wikiBtn.style.marginLeft = '8px';
    const iframe = document.createElement('iframe'); iframe.style.width='100%'; iframe.style.height='55vh'; iframe.style.border='1px solid #444'; iframe.style.display='none'; iframe.setAttribute('sandbox','allow-scripts allow-forms allow-popups allow-popups-to-escape-sandbox');
    const view = document.createElement('div'); view.style.marginTop = '8px'; view.style.background='#111'; view.style.color='#ddd'; view.style.padding='8px'; view.style.height='55vh'; view.style.overflow='auto';
    const results = document.createElement('div'); results.style.marginTop='8px'; results.style.color='#ddd';

    const doGo = async () => { const u = url.value.trim(); if (!u) return; await this.fetchDirect(u, iframe, view); };
    go.onclick = () => { void doGo(); };

    wikiBtn.onclick = async () => {
      const q = url.value.trim(); if (!q) return;
      results.innerHTML = 'Searching…';
      try {
        const items = await this.wikiSearch(q);
        this.log('browser_search', { engine: 'wikipedia', query: q, count: items.length });
        if (!items.length) { results.textContent = 'No results.'; return; }
        results.innerHTML = '';
        for (const it of items.slice(0, 12)) {
          const row = document.createElement('div'); row.style.padding='6px 0'; row.style.borderBottom='1px solid #333';
          const a = document.createElement('a'); a.href = `https://en.wikipedia.org/wiki/${encodeURIComponent(it.title)}`; a.textContent = it.title; a.target = '_blank';
          const p = document.createElement('div'); p.textContent = it.snippet;
          const open = document.createElement('button'); open.textContent = 'Open summary'; open.style.marginLeft='8px';
          open.onclick = async () => {
            const s = await this.wikiSummary(it.title);
            if (s) {
              iframe.style.display='none'; view.style.display='block';
              view.textContent = `Wikipedia — ${s.title}\n\n${s.extract}\n\nURL: ${s.url}`;
              this.lastUrl = s.url; this.lastTitle = s.title; await this.store.put('last', { url: this.lastUrl, title: this.lastTitle, ts: Date.now() });
              this.log('browser_visit', { engine: 'wikipedia', url: s.url, title: s.title, len: s.extract.length });
            }
          };
          row.appendChild(a); row.appendChild(open); row.appendChild(p);
          results.appendChild(row);
        }
      } catch (e) {
        results.textContent = 'Search failed.';
      }
    };

    el.appendChild(url); el.appendChild(go); el.appendChild(wikiBtn);
    el.appendChild(iframe); el.appendChild(view); el.appendChild(results);
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
  private publish: ((ev: { type: string; payload?: any }) => void) | null = null;
  setContext(ctx: { records: ReadonlyArray<K3DRecord>; publish?: (ev: { type: string; payload?: any }) => void }) { this.records = ctx.records; this.publish = ctx.publish || null; }
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
    const btnHi = document.createElement('button'); btnHi.textContent = 'Highlight in Mini-Map'; btnHi.style.marginLeft = '8px';
    const out = document.createElement('pre'); out.style.color='#ddd'; out.style.whiteSpace='pre-wrap'; out.style.marginTop='8px';
    const go = () => { this.focus = input.value.trim(); this.compute(); out.textContent = this.results.map(r=>`${r.sim.toFixed(3)}  ${r.label}`).join('\n'); };
    btn.onclick = go; btnHi.onclick = () => { if (this.publish) this.publish({ type: 'highlightNeighbors', payload: { labels: this.results.map(r=>r.label) } }); };
    el.appendChild(input); el.appendChild(btn); el.appendChild(btnHi); el.appendChild(out);
    if (this.focus) go();
  }
}

export class GraphApp implements TabletApp {
  id = 'graph'; title = 'Mini-Map';
  private records: ReadonlyArray<K3DRecord> = [];
  private focus: string = '';
  private highlight: Set<string> = new Set();
  setContext(ctx: { records: ReadonlyArray<K3DRecord> }) { this.records = ctx.records; }
  onEvent(ev: { type: string; payload?: any }) {
    if (ev.type === 'focus' && ev.payload?.label) this.focus = String(ev.payload.label);
    if (ev.type === 'highlightNeighbors' && Array.isArray(ev.payload?.labels)) this.highlight = new Set<string>(ev.payload.labels as string[]);
    if (ev.type === 'clearHighlight') this.highlight.clear();
  }
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
    for (let i=0;i<cap;i++){
      const rec = this.records[i];
      const v=rec.vector as [number,number,number]; const x=sx(v[0]), y=sy(v[1]);
      const lab = (rec.metadata?.label as string) || rec.id;
      if (this.highlight.has(lab)) { ctx.fillStyle='#3df5c7'; ctx.fillRect(x-1, y-1, 3, 3); }
      else { ctx.fillStyle = '#88aaff'; ctx.fillRect(x, y, 1, 1); }
    }
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

export class GalaxyApp implements TabletApp {
  id = 'galaxy'; title = 'Galaxy Context';
  private records: ReadonlyArray<K3DRecord> = [];
  private focus: string = '';
  private rings = 4; // number of expansion rings beyond focus
  private base = 12; // base nodes per ring, grows by phi^level
  private phi = 1.61803398875;
  private layout: { ring: number; label: string; sim: number }[] = [];
  private frozen = false;

  setContext(ctx: { records: ReadonlyArray<K3DRecord> }) { this.records = ctx.records; this.compute(); }
  onEvent(ev: { type: string; payload?: any }) { if (ev.type === 'focus' && ev.payload?.label) { this.focus = String(ev.payload.label); if (!this.frozen) this.compute(); } }

  private cosine(a: number[], b: number[]): number {
    let dot = 0, na = 0, nb = 0; const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) { const x=a[i], y=b[i]; dot += x*y; na += x*x; nb += y*y; }
    return (na>0 && nb>0) ? (dot / (Math.sqrt(na)*Math.sqrt(nb))) : 0;
  }

  private compute() {
    this.layout = [];
    if (!this.records.length || !this.focus) return;
    const idOf = (i:number)=> this.records[i].id;
    const labelOf = (i:number)=> (this.records[i].metadata?.label as string) || this.records[i].id;
    const idx = this.records.findIndex(r => labelOf(this.records.indexOf(r)) === this.focus || r.id === this.focus);
    const byId = new Map<string, number>(this.records.map((r,i)=>[r.id,i]));
    const findIndexByLabel = (lab:string)=> this.records.findIndex(r => labelOf(this.records.indexOf(r)) === lab);
    const focusIdx = idx >= 0 ? idx : findIndexByLabel(this.focus);
    if (focusIdx < 0) return;
    const focusEmb = this.records[focusIdx].embedding as number[];

    const seen = new Set<number>([focusIdx]);
    const ringSets: number[][] = [];
    // ring 1: direct neighbors (if provided)
    const neighIds = (this.records[focusIdx].neighbors || []) as string[];
    let ring = neighIds.map(id => byId.get(id)).filter((i): i is number => typeof i === 'number');
    ring = ring.filter(i=>!seen.has(i)); ring.forEach(i=>seen.add(i));
    ringSets.push(ring);
    // subsequent rings via neighbors-of-neighbors
    for (let r = 2; r <= this.rings; r++) {
      const prev = ringSets[ringSets.length-1] || [];
      const candSet = new Set<number>();
      for (const i of prev) {
        const ids = (this.records[i].neighbors || []) as string[];
        for (const nid of ids) {
          const j = byId.get(nid); if (j===undefined) continue; if (seen.has(j)) continue; candSet.add(j);
        }
      }
      const cands = Array.from(candSet);
      cands.sort((a,b)=> this.cosine(focusEmb, this.records[b].embedding as number[]) - this.cosine(focusEmb, this.records[a].embedding as number[]));
      const budget = Math.max(3, Math.round(this.base * Math.pow(this.phi, r-1)));
      const take = cands.slice(0, budget);
      take.forEach(i=>seen.add(i));
      ringSets.push(take);
    }
    // build layout entries with sims
    this.layout.push({ ring: 0, label: labelOf(focusIdx), sim: 1.0 });
    ringSets.forEach((arr, k) => {
      for (const i of arr) {
        const lab = labelOf(i);
        const sim = this.cosine(focusEmb, this.records[i].embedding as number[]);
        this.layout.push({ ring: k+1, label: lab, sim });
      }
    });
  }

  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }) {
    ctx.fillStyle = '#07090c'; ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    if (!this.layout.length) { ctx.fillStyle='#fff'; ctx.font='12px system-ui'; ctx.fillText('No focus selected.', rect.x+8, rect.y+18); return; }
    const cx = rect.x + rect.w/2, cy = rect.y + rect.h/2;
    const maxRing = Math.max(...this.layout.map(e=>e.ring));
    const r0 = Math.min(rect.w, rect.h) * 0.08;
    const step = Math.min(rect.w, rect.h) * 0.12; // spacing between rings
    // draw rings
    ctx.strokeStyle='#22344a';
    for (let k=0;k<=maxRing;k++){ ctx.beginPath(); ctx.arc(cx, cy, r0 + k*step, 0, Math.PI*2); ctx.stroke(); }
    // place nodes around each ring
    const byRing: Map<number, { label:string; sim:number }[]> = new Map();
    for (const e of this.layout) { if (!byRing.has(e.ring)) byRing.set(e.ring, []); byRing.get(e.ring)!.push({label:e.label, sim:e.sim}); }
    ctx.fillStyle='#ffffff'; ctx.font='10px system-ui';
    for (let k=0;k<=maxRing;k++){
      const items = byRing.get(k) || [];
      const radius = r0 + k*step;
      const n = Math.max(1, items.length);
      for (let i=0;i<items.length;i++){
        const a = (i / n) * Math.PI * 2;
        const x = cx + Math.cos(a) * radius;
        const y = cy + Math.sin(a) * radius;
        const s = Math.max(1, Math.min(3, Math.round((items[i].sim||0)*3)));
        ctx.fillStyle = k===0 ? '#ffcc00' : '#a8c0ff';
        ctx.fillRect(x-1, y-1, 2, 2);
        if (k<=1 && i<8) { ctx.fillStyle='#dddddd'; ctx.fillText(items[i].label.slice(0,22), x+4, y); }
      }
    }
    // focus label
    ctx.fillStyle='#ffcc00'; ctx.font='12px system-ui'; ctx.fillText('focus', cx+8, cy-8);
  }

  openOverlay(el: HTMLDivElement) {
    el.innerHTML='';
    const controls = document.createElement('div'); controls.style.display='flex'; controls.style.gap='8px'; controls.style.alignItems='center';
    const rings = document.createElement('input'); rings.type='number'; rings.min='1'; rings.max='8'; rings.value=String(this.rings);
    const base = document.createElement('input'); base.type='number'; base.min='4'; base.max='64'; base.value=String(this.base);
    const apply = document.createElement('button'); apply.textContent='Apply';
    const expand = document.createElement('button'); expand.textContent='Expand φ';
    const freeze = document.createElement('button'); freeze.textContent = this.frozen ? 'Unfreeze' : 'Freeze';
    apply.onclick = () => { this.rings = Math.max(1, Math.min(8, parseInt(rings.value||'4'))); this.base = Math.max(4, Math.min(64, parseInt(base.value||'12'))); this.compute(); };
    expand.onclick = () => { this.rings = Math.min(8, this.rings+1); this.base = Math.max(4, Math.round(this.base * this.phi)); this.compute(); rings.value=String(this.rings); base.value=String(this.base); };
    freeze.onclick = () => { this.frozen = !this.frozen; freeze.textContent = this.frozen ? 'Unfreeze' : 'Freeze'; if (!this.frozen) this.compute(); };
    controls.append('Rings:', rings, 'Base:', base, apply, expand, freeze);
    el.appendChild(controls);
    const hint = document.createElement('div'); hint.style.color='#ddd'; hint.style.marginTop='8px'; hint.textContent = 'Rings expand by phi; nodes per ring ≈ base × phi^(ring-1).'; el.appendChild(hint);
  }
}

export class StatsApp implements TabletApp {
  id = 'stats'; title = 'Live Stats';
  private totals = { goto: 0, resolved: 0, direct: 0, model: 0 };
  onEvent(ev: { type: string; payload?: any }) {
    if (ev.type === 'goto_resolution') {
      const p = ev.payload || {};
      this.totals.goto += 1;
      if (typeof p.sim === 'number' || (p.query && p.query !== p.target)) this.totals.resolved += 1; else this.totals.direct += 1;
      if (typeof p.model_confidence === 'number') this.totals.model += 1;
    }
  }
  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }) {
    ctx.fillStyle = '#0b0d0f'; ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.fillStyle = '#fff'; ctx.font = '12px system-ui';
    ctx.fillText(`goto total: ${this.totals.goto}`, rect.x+8, rect.y+18);
    ctx.fillText(`resolved: ${this.totals.resolved}`, rect.x+8, rect.y+36);
    ctx.fillText(`direct: ${this.totals.direct}`, rect.x+8, rect.y+54);
    ctx.fillText(`model assisted: ${this.totals.model}`, rect.x+8, rect.y+72);
  }
  openOverlay(el: HTMLDivElement) {
    el.innerHTML = '';
    const p = document.createElement('pre'); p.style.color='#ddd';
    p.textContent = JSON.stringify(this.totals, null, 2);
    el.appendChild(p);
  }
}

export class DoorsApp implements TabletApp {
  id = 'doors'; title = 'Doors';
  private items: { label: string; address?: string }[] = [];
  private publish: ((ev: { type: string; payload?: any }) => void) | null = null;
  setContext(ctx: { records: ReadonlyArray<K3DRecord>; publish?: (ev: { type: string; payload?: any }) => void }) { this.publish = ctx.publish || null; }
  onEvent(ev: { type: string; payload?: any }) {
    if (ev.type === 'doors_list' && Array.isArray(ev.payload?.items)) this.items = ev.payload.items as any[];
  }
  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }) {
    ctx.fillStyle = '#0b0d0f'; ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.fillStyle = '#fff'; ctx.font = '12px system-ui';
    ctx.fillText(`doors: ${this.items.length}`, rect.x+8, rect.y+18);
    let y = rect.y+36; const max = Math.min(8, this.items.length);
    for (let i=0;i<max;i++){ const it=this.items[i]; ctx.fillText(`${it.label}`, rect.x+8, y); y += 14; }
  }
  openOverlay(el: HTMLDivElement) {
    el.innerHTML = '';
    const list = document.createElement('div'); list.style.maxHeight='60vh'; list.style.overflow='auto'; list.style.marginTop='6px';
    for (const it of this.items) {
      const row = document.createElement('div'); row.style.display='flex'; row.style.justifyContent='space-between'; row.style.alignItems='center'; row.style.gap='8px'; row.style.marginTop='6px';
      const lab = document.createElement('div'); lab.textContent = it.label + (it.address?` — ${it.address}`:''); lab.style.color='#ddd'; lab.style.flex='1';
      const btn = document.createElement('button'); btn.textContent='Open'; btn.onclick=()=>{ this.publish?.({ type:'openDoor', payload:{ label: it.label, address: it.address } }); };
      row.appendChild(lab); row.appendChild(btn); list.appendChild(row);
    }
    el.appendChild(list);
  }
}

export class DiaryApp implements TabletApp {
  id = 'diary'; title = 'Diary';
  private store = openStore<any>('k3d-tablet','diary');
  private entries: { ts: number; text: string }[] = [];
  private publish: ((ev: { type: string; payload?: any }) => void) | null = null;
  async ensureLoaded(){ const arr=(await this.store.get('entries'))||[]; this.entries = Array.isArray(arr)?arr:[]; }
  async save(){ await this.store.put('entries', this.entries); }
  setContext(ctx: { records: ReadonlyArray<K3DRecord>; publish?: (ev: { type: string; payload?: any }) => void }) { this.publish = ctx.publish || null; }
  onEvent(ev: { type: string; payload?: any }) {}
  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }) {
    ctx.fillStyle = '#0b0d0f'; ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.fillStyle = '#fff'; ctx.font = '12px system-ui';
    const last = this.entries[this.entries.length-1];
    ctx.fillText(`entries: ${this.entries.length}`, rect.x+8, rect.y+18);
    if (last) ctx.fillText(`last: ${(new Date(last.ts)).toLocaleString()}`, rect.x+8, rect.y+36);
  }
  async openOverlay(el: HTMLDivElement) {
    await this.ensureLoaded(); el.innerHTML='';
    const ta = document.createElement('textarea'); ta.rows=4; ta.style.width='100%'; ta.placeholder='What did I learn?';
    const add = document.createElement('button'); add.textContent='Add'; add.onclick = async ()=>{ const t=ta.value.trim(); if(!t) return; this.entries.push({ ts: Date.now(), text: t }); await this.save(); this.publish?.({ type:'diaryAdd', payload:{ text: t } }); ta.value=''; renderList(); };
    const list = document.createElement('div'); list.style.marginTop='8px'; list.style.maxHeight='50vh'; list.style.overflow='auto';
    const renderList = ()=>{ list.innerHTML=''; for (const e of this.entries.slice().reverse()){ const p = document.createElement('div'); p.style.color='#ddd'; p.style.padding='6px'; p.textContent = new Date(e.ts).toLocaleString()+': '+e.text; list.appendChild(p);} };
    el.appendChild(ta); el.appendChild(add); el.appendChild(list); renderList();
  }
}

export class ControlApp implements TabletApp {
  id = 'control'; title = 'Control';
  private publish: ((ev: { type: string; payload?: any }) => void) | null = null;
  setContext(ctx: { records: ReadonlyArray<K3DRecord>; publish?: (ev: { type: string; payload?: any }) => void }) { this.publish = ctx.publish || null; }
  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }) {
    ctx.fillStyle = '#0b0d0f'; ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.fillStyle = '#fff'; ctx.font = '12px system-ui';
    ctx.fillText('Sleep/Wake orchestration', rect.x+8, rect.y+18);
    ctx.fillText('Use Focus → Control for buttons', rect.x+8, rect.y+36);
  }
  openOverlay(el: HTMLDivElement) {
    el.innerHTML = '';
    const row = document.createElement('div'); row.style.display='flex'; row.style.gap='8px';
    const sleep = document.createElement('button'); sleep.textContent='Sleep (pause)'; sleep.onclick=()=>{ this.publish?.({ type:'sleep', payload:{ mode:'pause' } }); };
    const consolidate = document.createElement('button'); consolidate.textContent='Sleep + Consolidate'; consolidate.onclick=()=>{ this.publish?.({ type:'sleep', payload:{ mode:'consolidate' } }); };
    const wake = document.createElement('button'); wake.textContent='Wake (resume)'; wake.onclick=()=>{ this.publish?.({ type:'wake' }); };
    row.appendChild(sleep); row.appendChild(consolidate); row.appendChild(wake);
    el.appendChild(row);
    const p = document.createElement('div'); p.style.color='#ddd'; p.style.marginTop='8px'; p.textContent = 'Sleep consolidates diary/reflections/training and re-exports memory_house.gltf (GMT-3 timestamps).'; el.appendChild(p);
  }
}

export class SummaryApp implements TabletApp {
  id = 'summary'; title = 'Summary';
  private dataset: { house?: string; nodes?: number; dims?: number; doors?: number; guided?: number } = {};
  private scoreboard: any = null;
  onEvent(ev: { type: string; payload?: any }) {
    if (ev.type === 'dataset_summary') this.dataset = { ...(ev.payload||{}) };
    if (ev.type === 'scoreboard_summary') this.scoreboard = ev.payload || null;
  }
  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }) {
    ctx.fillStyle = '#0b0d0f'; ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.fillStyle = '#fff'; ctx.font = '12px system-ui';
    ctx.fillText(`house: ${this.dataset.house||'—'}`, rect.x+8, rect.y+18);
    ctx.fillText(`nodes: ${this.dataset.nodes||0}  dims: ${this.dataset.dims||0}`, rect.x+8, rect.y+36);
    ctx.fillText(`doors: ${this.dataset.doors||0}  guided: ${this.dataset.guided||0}`, rect.x+8, rect.y+54);
    if (this.scoreboard) {
      const s = this.scoreboard;
      const y0 = rect.y+78;
      ctx.fillText(`Scoreboard: ${s.ts||''}`, rect.x+8, y0);
      ctx.fillText(`GOTO: ${s.goto.success}/${s.goto.count} med=${s.goto.median_hops}`, rect.x+8, y0+18);
      ctx.fillText(`DOOR: ${s.door.success}/${s.door.count} med=${s.door.median_hops}`, rect.x+8, y0+36);
    }
  }
  openOverlay(el: HTMLDivElement) {
    el.innerHTML='';
    const pre = document.createElement('pre'); pre.style.color='#ddd'; pre.style.whiteSpace='pre-wrap';
    const ds = this.dataset; const sb = this.scoreboard;
    pre.textContent = `Dataset\n- house: ${ds.house||'—'}\n- nodes: ${ds.nodes||0}\n- dims: ${ds.dims||0}\n- doors: ${ds.doors||0}\n- guided: ${ds.guided||0}\n\nScoreboard\n${sb?JSON.stringify(sb,null,2):'(none)'}`;
    el.appendChild(pre);
  }
}

export class LayersApp implements TabletApp {
  id = 'layers'; title = 'Layers';
  private layers: string[] = [];
  private enabled: Set<string> = new Set();
  private publish: ((ev: { type: string; payload?: any }) => void) | null = null;
  setContext(ctx: { records: ReadonlyArray<K3DRecord>; publish?: (ev: { type: string; payload?: any }) => void }) {
    this.publish = ctx.publish || null;
    const set = new Set<string>();
    for (const r of ctx.records) {
      const l = (r.metadata?.layer as string) || (Array.isArray(r.metadata?.tags) ? (r.metadata.tags[0] as string) : undefined);
      if (l) set.add(l);
    }
    this.layers = Array.from(set);
    this.enabled = new Set(this.layers);
  }
  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }) {
    ctx.fillStyle = '#0b0d0f'; ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
    ctx.fillStyle = '#fff'; ctx.font = '12px system-ui';
    ctx.fillText(`layers: ${this.layers.length}`, rect.x+8, rect.y+18);
    let y = rect.y+36;
    for (const name of this.layers.slice(0, 8)) {
      const on = this.enabled.has(name) ? '✓' : '✗';
      ctx.fillText(`${on} ${name}`, rect.x+8, y); y += 14;
    }
  }
  openOverlay(el: HTMLDivElement) {
    el.innerHTML = '';
    if (!this.layers.length) { const p = document.createElement('div'); p.textContent='No layers found.'; p.style.color='#ddd'; el.appendChild(p); return; }
    const list = document.createElement('div'); list.style.marginTop='6px';
    const apply = () => {
      if (this.publish) this.publish({ type: 'applyLayers', payload: { enabled: Array.from(this.enabled) } });
    };
    for (const name of this.layers) {
      const row = document.createElement('label'); row.style.display='block'; row.style.marginTop='4px';
      const cb = document.createElement('input'); cb.type='checkbox'; cb.checked = this.enabled.has(name);
      cb.onchange = () => { if (cb.checked) this.enabled.add(name); else this.enabled.delete(name); apply(); };
      row.appendChild(cb); row.appendChild(document.createTextNode(' '+name)); list.appendChild(row);
    }
    const all = document.createElement('button'); all.textContent='All'; all.onclick = () => { this.enabled = new Set(this.layers); apply(); };
    const none = document.createElement('button'); none.textContent='None'; none.style.marginLeft='6px'; none.onclick = () => { this.enabled.clear(); apply(); };
    el.appendChild(all); el.appendChild(none); el.appendChild(list);
  }
}
