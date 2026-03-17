import type { RpnOpHandler, RpnValue } from './engine';

export interface RpnString {
  str: string;
}

function isRpnString(value: unknown): value is RpnString {
  return typeof value === 'object' && value !== null && 'str' in value;
}

export function popString(stack: RpnValue[]): string {
  const value = stack.pop();
  if (value === undefined) {
    throw new Error('RPN underflow');
  }
  if (typeof value === 'number') {
    return String(value);
  }
  if (isRpnString(value)) {
    return String(value.str);
  }
  throw new Error(`Expected string, received ${typeof value}`);
}

function createElement(tag: string): RpnOpHandler {
  return (_stack, context) => {
    context.dom?.pushElement(tag);
  };
}

function normalizeRefToken(value: string): string {
  if (value.startsWith('char ') || value.startsWith('word ')) {
    return value.replace(/ /g, '_');
  }
  return value;
}

export const domOps: Record<string, RpnOpHandler> = {
  DOM_CREATE: (stack, context) => {
    context.dom?.pushElement(popString(stack));
  },
  DOM_P: createElement('p'),
  DOM_SPAN: createElement('span'),
  DOM_DIV: createElement('div'),
  DOM_H1: createElement('h1'),
  DOM_H2: createElement('h2'),
  DOM_H3: createElement('h3'),
  DOM_UL: createElement('ul'),
  DOM_LI: createElement('li'),
  DOM_SECTION: createElement('section'),
  DOM_ARTICLE: createElement('article'),
  DOM_TEXT: (stack, context) => {
    context.dom?.setText(popString(stack));
  },
  DOM_ATTR: (stack, context) => {
    const value = popString(stack);
    const name = popString(stack);
    context.dom?.setAttribute(name, value);
  },
  DOM_CLASS: (stack, context) => {
    context.dom?.setClass(popString(stack));
  },
  DOM_APPEND: (_stack, context) => {
    context.dom?.appendChild();
  },
  DOM_EMIT: (_stack, context) => {
    context.dom?.appendToRoot();
  },
  DOM_CHAR: (stack, context) => {
    const charRef = normalizeRefToken(popString(stack));
    context.dom?.pushElement('span');
    context.dom?.setClass('k3d-char');
    context.dom?.setAttribute('data-k3d-ref', charRef);
    const match = charRef.match(/^char_u([0-9a-fA-F]{4,6})$/);
    if (match) {
      context.dom?.setText(String.fromCodePoint(parseInt(match[1], 16)));
    }
  },
  DOM_WORD: (stack, context) => {
    const wordRef = normalizeRefToken(popString(stack));
    context.dom?.pushElement('span');
    context.dom?.setClass('k3d-word');
    context.dom?.setAttribute('data-k3d-ref', wordRef);
    const wordText = wordRef.replace(/^word_/, '').replace(/_/g, ' ');
    for (const character of wordText) {
      const codePoint = character.codePointAt(0);
      if (codePoint === undefined) continue;
      const hex = codePoint.toString(16).padStart(4, '0');
      context.dom?.pushElement('span');
      context.dom?.setClass('k3d-char');
      context.dom?.setAttribute('data-k3d-ref', `char_u${hex}`);
      context.dom?.setText(character);
      context.dom?.appendChild();
    }
  },
};
