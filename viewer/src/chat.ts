export type ChatMessage = {
  type: 'chat';
  from: string;
  text: string;
  channel?: string;
  to?: string;
  action?: boolean;
};

export type CommandMessage = {
  type: 'command';
  command: string; // 'goto' | 'open' | others
  target: string;  // often a label or JSON string payload
  channel?: string;
};

export type LiveMessage = ChatMessage | CommandMessage;

type Handlers = {
  onChat?: (msg: ChatMessage) => void;
  onCommand?: (msg: CommandMessage) => void;
  onStatus?: (status: 'connected' | 'disconnected' | 'error') => void;
};

export class ChatClient {
  private ws: WebSocket | null = null;
  private url: string;
  private handlers: Handlers;
  private connected = false;
  private outbox = openStore<any>('k3d-tablet', 'outbox');

  constructor(url: string, handlers: Handlers = {}) {
    this.url = url;
    this.handlers = handlers;
  }

  connect() {
    try {
      this.ws = new WebSocket(this.url);
      this.ws.onopen = async () => {
        this.connected = true;
        this.handlers.onStatus?.('connected');
        await this.flushOutbox();
      };
      this.ws.onclose = () => { this.connected = false; this.handlers.onStatus?.('disconnected'); };
      this.ws.onerror = () => { this.connected = false; this.handlers.onStatus?.('error'); };
      this.ws.onmessage = (ev) => {
        try {
          const data = JSON.parse(String(ev.data)) as LiveMessage;
          if (data.type === 'chat') this.handlers.onChat?.(data);
          else if (data.type === 'command') this.handlers.onCommand?.(data);
        } catch {
          // ignore malformed
        }
      };
    } catch {
      this.handlers.onStatus?.('error');
    }
  }

  disconnect() {
    this.ws?.close();
    this.ws = null;
    this.connected = false;
  }

  isConnected(): boolean { return !!this.ws && this.connected && this.ws.readyState === WebSocket.OPEN; }

  private trySend(obj: any): boolean {
    if (this.isConnected()) { this.ws!.send(JSON.stringify(obj)); return true; }
    return false;
  }

  private async enqueue(obj: any) {
    const key = 'queue';
    const q = (await this.outbox.get(key)) || [];
    q.push({ ...obj, ts: Date.now() });
    await this.outbox.put(key, q);
  }

  async getQueueLength(): Promise<number> {
    const q = (await this.outbox.get('queue')) || [];
    return Array.isArray(q) ? q.length : 0;
  }

  async flushOutbox() {
    const key = 'queue';
    const q = (await this.outbox.get(key)) || [];
    if (!Array.isArray(q) || q.length === 0) return;
    const keep: any[] = [];
    for (const obj of q) {
      if (!this.trySend(obj)) keep.push(obj);
    }
    await this.outbox.put(key, keep);
  }

  sendChat(text: string) {
    const msg: ChatMessage = { type: 'chat', from: 'human', text };
    if (!this.trySend(msg)) this.enqueue(msg);
  }

  sendCommandGoto(target: string) {
    const msg: CommandMessage = { type: 'command', command: 'goto', target };
    if (!this.trySend(msg)) this.enqueue(msg);
  }

  sendEvent(event: Record<string, unknown>) {
    const payload = { type: 'event', event };
    if (!this.trySend(payload)) this.enqueue(payload);
  }
}
