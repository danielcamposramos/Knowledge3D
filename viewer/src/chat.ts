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
  command: 'goto';
  target: string;
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

  constructor(url: string, handlers: Handlers = {}) {
    this.url = url;
    this.handlers = handlers;
  }

  connect() {
    try {
      this.ws = new WebSocket(this.url);
      this.ws.onopen = () => this.handlers.onStatus?.('connected');
      this.ws.onclose = () => this.handlers.onStatus?.('disconnected');
      this.ws.onerror = () => this.handlers.onStatus?.('error');
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
  }

  sendChat(text: string) {
    const msg: ChatMessage = { type: 'chat', from: 'human', text };
    this.ws?.send(JSON.stringify(msg));
  }

  sendCommandGoto(target: string) {
    const msg: CommandMessage = { type: 'command', command: 'goto', target };
    this.ws?.send(JSON.stringify(msg));
  }
}
