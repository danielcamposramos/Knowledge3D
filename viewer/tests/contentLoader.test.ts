import { clearHouseContent, getBookContent, loadHouseContent, resolveRef } from '../src/contentLoader';

const mockContent = {
  version: 1,
  books: {
    'Book/MathematicsPrimer': {
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
      ],
    },
  },
  concepts: {
    concept_mathematics: {
      star_id: 'concept_mathematics',
      meaning_class: 'concept',
      domain: 'Library/Mathematics',
      meaning_rpn: 'MATHEMATICS',
      behavior_rpn: 'DOMAIN_CENTER',
      surface_forms: { en: { word_ref: 'Mathematics', char_refs: [] } },
      taxonomy_refs: [],
      grammar_refs: [],
      component_refs: [],
    },
  },
};

describe('content loader', () => {
  beforeEach(() => {
    clearHouseContent();
    global.fetch = jest.fn().mockResolvedValue({
      ok: true,
      json: async () => mockContent,
    } as Response);
  });

  afterEach(() => {
    clearHouseContent();
    jest.resetAllMocks();
  });

  it('loads and resolves book content', async () => {
    await loadHouseContent('/house-content.json');
    expect(getBookContent('Book/MathematicsPrimer')?.entries.length).toBe(1);
    expect(resolveRef('concept_mathematics')?.star_id).toBe('concept_mathematics');
  });

  it('returns null for unknown ref', async () => {
    await loadHouseContent('/house-content.json');
    expect(resolveRef('nonexistent_star')).toBeNull();
  });
});
