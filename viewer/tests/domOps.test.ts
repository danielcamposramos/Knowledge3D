import { rpnToDOM } from '../src/rpn';

describe('DOM RPN operations', () => {
  it('creates a paragraph with text', () => {
    const root = rpnToDOM('STR_Hello_World DOM_P DOM_TEXT DOM_EMIT');
    expect(root.querySelector('p')?.textContent).toBe('Hello World');
  });

  it('creates nested list', () => {
    const root = rpnToDOM(
      'DOM_UL ' +
      'STR_Item_one DOM_LI DOM_TEXT DOM_APPEND ' +
      'STR_Item_two DOM_LI DOM_TEXT DOM_APPEND ' +
      'DOM_EMIT'
    );
    const list = root.querySelector('ul');
    expect(list?.children.length).toBe(2);
    expect(list?.children[0]?.textContent).toBe('Item one');
  });

  it('creates heading with class', () => {
    const root = rpnToDOM('STR_k3d-title DOM_H1 DOM_CLASS STR_Welcome DOM_TEXT DOM_EMIT');
    const heading = root.querySelector('h1');
    expect(heading?.className).toBe('k3d-title');
    expect(heading?.textContent).toBe('Welcome');
  });

  it('renders character reference', () => {
    const root = rpnToDOM('STR_char_u0041 DOM_CHAR DOM_EMIT');
    const span = root.querySelector('span.k3d-char');
    expect(span?.textContent).toBe('A');
    expect(span?.getAttribute('data-k3d-ref')).toBe('char_u0041');
  });

  it('renders word as character spans', () => {
    const root = rpnToDOM('STR_word_cat DOM_WORD DOM_EMIT');
    const word = root.querySelector('span.k3d-word');
    expect(word?.getAttribute('data-k3d-ref')).toBe('word_cat');
    const chars = word?.querySelectorAll('span.k3d-char');
    expect(chars?.length).toBe(3);
    expect(chars?.[0]?.textContent).toBe('c');
    expect(chars?.[1]?.textContent).toBe('a');
    expect(chars?.[2]?.textContent).toBe('t');
  });

  it('string literals convert underscores to spaces', () => {
    const root = rpnToDOM('STR_Hello_beautiful_World DOM_P DOM_TEXT DOM_EMIT');
    expect(root.querySelector('p')?.textContent).toBe('Hello beautiful World');
  });
});
