import * as THREE from 'three';
import { openStore } from './cache';
import { RPN } from './rpn';

export interface TabletApp {
  id: string;
  title: string;
  renderCanvas(ctx: CanvasRenderingContext2D, rect: { x: number; y: number; w: number; h: number }): void;
  openOverlay(el: HTMLDivElement): void;
  onEvent?(ev: { type: string; payload?: any }): void;
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

