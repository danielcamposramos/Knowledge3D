import type { ContentPage } from './contentRenderer';

function encodeLiteral(text: string): string {
  return `STR_${String(text ?? '').replace(/\s+/g, '_')}`;
}

export function buildNodeDomProgram(page: ContentPage | null): string | null {
  if (!page) return null;
  const tokens: string[] = [];
  tokens.push(encodeLiteral(page.title), 'DOM_H2', 'DOM_TEXT', 'DOM_EMIT');

  for (const section of page.sections) {
    tokens.push('DOM_SECTION');
    tokens.push(encodeLiteral(section.heading), 'DOM_H3', 'DOM_TEXT', 'DOM_APPEND');
    for (const line of section.lines) {
      tokens.push(encodeLiteral(line), 'DOM_P', 'DOM_TEXT', 'DOM_APPEND');
    }
    tokens.push('DOM_EMIT');
  }

  return tokens.join(' ');
}
