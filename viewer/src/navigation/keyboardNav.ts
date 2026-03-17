import type { HouseNode, LoadedHouseScene } from '../loadHouseScene';

export interface KeyboardNavCallbacks {
  onRoomChange: (room: HouseNode) => void;
}

function sortedRooms(scene: LoadedHouseScene): HouseNode[] {
  return [...scene.rooms].sort((a, b) => a.housePosition[0] - b.housePosition[0]);
}

export class KeyboardNav {
  private scene: LoadedHouseScene;
  private callbacks: KeyboardNavCallbacks;
  private handler: (event: KeyboardEvent) => void;

  constructor(scene: LoadedHouseScene, callbacks: KeyboardNavCallbacks) {
    this.scene = scene;
    this.callbacks = callbacks;
    this.handler = this.onKeyDown.bind(this);
  }

  attach(): void {
    window.addEventListener('keydown', this.handler);
  }

  detach(): void {
    window.removeEventListener('keydown', this.handler);
  }

  private onKeyDown(event: KeyboardEvent): void {
    const tag = ((event.target as HTMLElement | null)?.tagName || '').toUpperCase();
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

    const key = String(event.key || '');
    const normalized = key.toLowerCase();
    switch (normalized) {
      case 'arrowright':
      case 'd':
        this.moveToNeighbor(1);
        event.preventDefault();
        break;
      case 'arrowleft':
      case 'a':
        this.moveToNeighbor(-1);
        event.preventDefault();
        break;
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
        this.moveToRoomIndex(parseInt(normalized, 10) - 1);
        event.preventDefault();
        break;
      case 'h':
        this.moveToRoom('room_living');
        event.preventDefault();
        break;
      default:
        break;
    }
  }

  private moveToNeighbor(direction: number): void {
    const currentId = this.scene.currentRoom;
    const neighbors = this.scene.navGraph.edges
      .filter((edge) => edge.from === currentId)
      .map((edge) => this.scene.nodesByStarId.get(edge.to) || null)
      .filter((node): node is HouseNode => !!node && node.meaningClass === 'room')
      .sort((a, b) => a.housePosition[0] - b.housePosition[0]);
    if (!neighbors.length) return;

    const current = this.scene.nodesByStarId.get(currentId);
    if (!current) return;
    const currentX = current.housePosition[0];
    const target = direction > 0
      ? neighbors.find((node) => node.housePosition[0] > currentX)
      : [...neighbors].reverse().find((node) => node.housePosition[0] < currentX);
    if (target) this.callbacks.onRoomChange(target);
  }

  private moveToRoomIndex(index: number): void {
    const rooms = sortedRooms(this.scene);
    const target = rooms[index];
    if (target) this.callbacks.onRoomChange(target);
  }

  private moveToRoom(starId: string): void {
    const room = this.scene.nodesByStarId.get(starId);
    if (room && room.meaningClass === 'room') {
      this.callbacks.onRoomChange(room);
      return;
    }
    const fallback = sortedRooms(this.scene)[0];
    if (fallback) this.callbacks.onRoomChange(fallback);
  }
}
