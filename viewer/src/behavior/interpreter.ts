import type { HouseNode } from '../loadHouseScene';

export type BehaviorAction =
  | { type: 'room_enter'; room: string; domain: string }
  | { type: 'door_traverse'; roomA: string; roomB: string }
  | { type: 'load_galaxy'; galaxyRef: string }
  | { type: 'inspect_object'; starId: string }
  | { type: 'activate_display'; taxonomyRefs: string[] }
  | { type: 'browse_galaxy' }
  | { type: 'noop' };

export function interpretBehavior(
  behaviorRpn: string,
  node: HouseNode,
): BehaviorAction {
  const tokens = String(behaviorRpn || '').trim().split(/\s+/).filter(Boolean);
  const command = tokens[0] || '';
  switch (command) {
    case 'ROOM_ENTER':
      return { type: 'room_enter', room: node.houseRoom, domain: node.domain };
    case 'DOOR_TRAVERSE':
      return { type: 'door_traverse', roomA: tokens[2] || '', roomB: tokens[3] || '' };
    case 'OPEN_BOOK':
    case 'BOOK':
      return { type: 'load_galaxy', galaxyRef: node.galaxyRef || tokens[2] || '' };
    case 'TOOL_OBJECT':
    case 'OBSERVE':
    case 'PORTAL':
      return { type: 'inspect_object', starId: node.starId };
    case 'DISPLAY':
      return { type: 'activate_display', taxonomyRefs: node.taxonomyRefs };
    case 'TABLET':
    case 'HOLODESK':
      return { type: 'browse_galaxy' };
    default:
      return { type: 'noop' };
  }
}
