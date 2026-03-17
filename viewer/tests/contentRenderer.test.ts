import type { HouseNode } from '../src/loadHouseScene';
import { renderBookContent, renderNodeContent } from '../src/behavior';

const mockBookNode: HouseNode = {
  starId: 'book_mathematics_primer',
  meaningClass: 'book',
  domain: 'House/Library/Books',
  houseRoom: 'House/Library',
  housePosition: [0, 0, 0],
  surfaceForms: {
    en: { word_ref: 'Mathematics Primer', char_refs: [] },
    pt: { word_ref: 'Primer de Matematica', char_refs: [] },
  },
  behaviorRpn: 'BOOK OPEN LOAD_GALAXY READ_SEQUENCE',
  taxonomyRefs: ['concept_mathematics', 'num_0'],
  componentRefs: ['chapter_numbers', 'chapter_operations'],
  visualRpn: '1.0 GEN_CUBE 0.2 0.3 0.05 MAT4_SCALE MAT4_APPLY',
  galaxyRef: 'Book/MathematicsPrimer',
  object: {} as any,
};

describe('content renderer', () => {
  it('renders node content with all sections', () => {
    const page = renderNodeContent(mockBookNode);
    expect(page.title).toBeTruthy();
    expect(page.sections.find((section) => section.heading === 'Identity')).toBeTruthy();
    expect(page.sections.find((section) => section.heading === 'Content Galaxy')).toBeTruthy();
    expect(page.sections.find((section) => section.heading === 'References')).toBeTruthy();
  });

  it('renders book content hierarchy', () => {
    const page = renderBookContent(
      'Book/MathematicsPrimer',
      {
        entries: [
          {
            star_id: 'mathbook_ch1_numbers',
            meaning_class: 'chapter',
            domain: 'Book/MathematicsPrimer',
            meaning_rpn: 'CHAPTER NUMBERS',
            behavior_rpn: 'OPEN',
            surface_forms: { en: { word_ref: 'Chapter 1 Numbers', char_refs: [] } },
            taxonomy_refs: ['concept_mathematics'],
            grammar_refs: [],
            component_refs: ['mathbook_sec1_counting'],
          },
          {
            star_id: 'mathbook_sec1_counting',
            meaning_class: 'section',
            domain: 'Book/MathematicsPrimer',
            meaning_rpn: 'SECTION COUNTING',
            behavior_rpn: 'OPEN',
            surface_forms: { en: { word_ref: 'Counting and Order', char_refs: [] } },
            taxonomy_refs: ['num_0', 'num_1'],
            grammar_refs: ['grammar_numeric_literal'],
            component_refs: [],
          },
        ],
      },
      mockBookNode,
    );
    expect(page.title).toBe('Mathematics Primer');
    expect(page.sections[0]?.heading).toBe('Chapter 1 Numbers');
    expect(page.sections[0]?.lines[0]).toContain('§ Counting and Order');
    expect(page.sections[0]?.lines.some((line) => line.includes('rules: grammar_numeric_literal'))).toBe(true);
  });
});
