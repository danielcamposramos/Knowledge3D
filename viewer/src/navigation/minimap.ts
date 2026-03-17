import type { HouseNode, LoadedHouseScene } from '../loadHouseScene';

function orderedRooms(scene: LoadedHouseScene): HouseNode[] {
  return [...scene.rooms].sort((a, b) => a.housePosition[0] - b.housePosition[0]);
}

export class Minimap {
  readonly element: HTMLDivElement;
  private dots = new Map<string, HTMLDivElement>();

  constructor(scene: LoadedHouseScene) {
    this.element = document.createElement('div');
    this.element.className = 'k3d-minimap';
    Object.assign(this.element.style, {
      position: 'fixed',
      bottom: '16px',
      left: '16px',
      width: '200px',
      minHeight: '40px',
      background: 'rgba(0,0,0,0.5)',
      borderRadius: '8px',
      padding: '6px 10px',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      gap: '8px',
      zIndex: '100',
      pointerEvents: 'auto',
    });

    for (const room of orderedRooms(scene)) {
      const dot = document.createElement('div');
      dot.title = room.surfaceForms.en?.word_ref || room.starId;
      dot.dataset.starId = room.starId;
      Object.assign(dot.style, {
        width: '12px',
        height: '12px',
        borderRadius: '50%',
        background: '#666',
        cursor: 'pointer',
        transition: 'all 0.3s',
        flex: '0 0 auto',
      });
      this.element.appendChild(dot);
      this.dots.set(room.starId, dot);
    }
    document.body.appendChild(this.element);
  }

  setCurrentRoom(starId: string): void {
    this.dots.forEach((dot, id) => {
      const active = id === starId;
      dot.style.background = active ? '#00ddff' : '#666';
      dot.style.transform = active ? 'scale(1.4)' : 'scale(1)';
      dot.style.boxShadow = active ? '0 0 8px #00ddff' : 'none';
    });
  }

  onClick(callback: (starId: string) => void): void {
    this.dots.forEach((dot, starId) => {
      dot.onclick = () => callback(starId);
    });
  }

  destroy(): void {
    this.element.remove();
    this.dots.clear();
  }
}
