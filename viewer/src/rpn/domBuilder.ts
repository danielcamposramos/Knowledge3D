export class DomBuilder {
  private elements: HTMLElement[] = [];
  private root: HTMLDivElement;

  constructor() {
    this.root = this.createRoot();
  }

  private createRoot(): HTMLDivElement {
    const root = document.createElement('div');
    root.className = 'k3d-dom-root';
    return root;
  }

  reset(): void {
    this.elements = [];
    this.root = this.createRoot();
  }

  pushElement(tag: string): void {
    this.elements.push(document.createElement(tag));
  }

  popElement(): HTMLElement | undefined {
    return this.elements.pop();
  }

  peekElement(): HTMLElement | undefined {
    return this.elements[this.elements.length - 1];
  }

  appendChild(): void {
    if (this.elements.length < 2) return;
    const child = this.elements.pop();
    const parent = this.peekElement();
    if (child && parent) parent.appendChild(child);
  }

  appendToRoot(): void {
    const element = this.popElement();
    if (element) this.root.appendChild(element);
  }

  setText(text: string): void {
    const element = this.peekElement();
    if (element) element.textContent = text;
  }

  setAttribute(name: string, value: string): void {
    const element = this.peekElement();
    if (element) element.setAttribute(name, value);
  }

  setClass(className: string): void {
    const element = this.peekElement();
    if (element) element.className = className;
  }

  toDOM(): HTMLDivElement {
    while (this.elements.length > 0) {
      const element = this.elements.shift();
      if (element) this.root.appendChild(element);
    }
    return this.root;
  }

  hasElements(): boolean {
    return this.elements.length > 0 || this.root.childElementCount > 0;
  }
}
