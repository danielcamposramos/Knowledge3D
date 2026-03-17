import type { HouseNode } from '../src/loadHouseScene';
import { renderNodeContent } from '../src/behavior';

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
});
