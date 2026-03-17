import type { HouseNode } from '../src/loadHouseScene';
import { interpretBehavior } from '../src/behavior';

const mockNode: HouseNode = {
  starId: 'door_library_garden',
  meaningClass: 'door',
  domain: 'House/Connectivity',
  houseRoom: 'House/Library',
  housePosition: [0, 0, 0],
  surfaceForms: { en: { word_ref: 'Library Garden Door', char_refs: [] } },
  behaviorRpn: 'DOOR_TRAVERSE CONNECT House/Library House/Garden',
  taxonomyRefs: [],
  componentRefs: [],
  object: {} as any,
};

describe('behavior interpreter', () => {
  it('door behavior produces door_traverse action', () => {
    const action = interpretBehavior('DOOR_TRAVERSE CONNECT House/Library House/Garden', mockNode);
    expect(action.type).toBe('door_traverse');
    if (action.type !== 'door_traverse') return;
    expect(action.roomA).toBe('House/Library');
    expect(action.roomB).toBe('House/Garden');
  });

  it('book behavior produces load_galaxy action', () => {
    const node = { ...mockNode, galaxyRef: 'Book/MathematicsPrimer', meaningClass: 'book' } as HouseNode;
    const action = interpretBehavior('OPEN_BOOK LOAD_GALAXY Book/MathematicsPrimer', node);
    expect(action.type).toBe('load_galaxy');
    if (action.type !== 'load_galaxy') return;
    expect(action.galaxyRef).toBe('Book/MathematicsPrimer');
  });

  it('tablet behavior produces browse_galaxy action', () => {
    const action = interpretBehavior('TABLET ACTIVATE BROWSE_GALAXY', mockNode);
    expect(action.type).toBe('browse_galaxy');
  });

  it('holodesk behavior produces browse_galaxy action', () => {
    const action = interpretBehavior('HOLODESK ACTIVATE PROJECT_3D', mockNode);
    expect(action.type).toBe('browse_galaxy');
  });

  it('unknown behavior produces noop', () => {
    const action = interpretBehavior('UNKNOWN_COMMAND', mockNode);
    expect(action.type).toBe('noop');
  });
});
