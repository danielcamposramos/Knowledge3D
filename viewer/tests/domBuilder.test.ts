import { DomBuilder } from '../src/rpn';

describe('DomBuilder', () => {
  it('creates and emits a paragraph', () => {
    const builder = new DomBuilder();
    builder.pushElement('p');
    builder.setText('Hello World');
    builder.appendToRoot();
    const root = builder.toDOM();
    expect(root.querySelector('p')?.textContent).toBe('Hello World');
  });

  it('nests elements via appendChild', () => {
    const builder = new DomBuilder();
    builder.pushElement('ul');
    builder.pushElement('li');
    builder.setText('Item 1');
    builder.appendChild();
    builder.appendToRoot();
    const root = builder.toDOM();
    const list = root.querySelector('ul');
    expect(list?.children.length).toBe(1);
    expect(list?.querySelector('li')?.textContent).toBe('Item 1');
  });

  it('sets attributes on elements', () => {
    const builder = new DomBuilder();
    builder.pushElement('span');
    builder.setAttribute('data-k3d-ref', 'char_u0041');
    builder.setText('A');
    builder.appendToRoot();
    const root = builder.toDOM();
    const span = root.querySelector('span');
    expect(span?.getAttribute('data-k3d-ref')).toBe('char_u0041');
    expect(span?.textContent).toBe('A');
  });

  it('resets cleanly', () => {
    const builder = new DomBuilder();
    builder.pushElement('p');
    builder.setText('First');
    builder.appendToRoot();
    builder.toDOM();
    builder.reset();
    const root = builder.toDOM();
    expect(root.children.length).toBe(0);
  });
});
