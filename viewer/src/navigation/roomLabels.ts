import * as THREE from 'three';

import type { HouseNode, LoadedHouseScene } from '../loadHouseScene';

type LabelRecord = {
  node: HouseNode;
  element: HTMLDivElement;
};

export class RoomLabels {
  private labels: LabelRecord[] = [];
  private container: HTMLDivElement;

  constructor(scene: LoadedHouseScene) {
    this.container = document.createElement('div');
    this.container.className = 'k3d-room-labels';
    Object.assign(this.container.style, {
      position: 'fixed',
      top: '0',
      left: '0',
      width: '100%',
      height: '100%',
      pointerEvents: 'none',
      zIndex: '50',
    });

    for (const room of scene.rooms) {
      const label = document.createElement('div');
      label.textContent = room.surfaceForms.en?.word_ref || room.starId;
      Object.assign(label.style, {
        position: 'absolute',
        color: 'white',
        fontSize: '13px',
        fontFamily: 'monospace',
        textShadow: '0 1px 4px rgba(0,0,0,0.8)',
        opacity: '0.7',
        whiteSpace: 'nowrap',
        transform: 'translate(-50%, -100%)',
        transition: 'opacity 0.2s, font-size 0.2s, color 0.2s',
      });
      this.container.appendChild(label);
      this.labels.push({ node: room, element: label });
    }

    document.body.appendChild(this.container);
  }

  update(camera: THREE.Camera): void {
    const vec = new THREE.Vector3();
    for (const { node, element } of this.labels) {
      node.object.getWorldPosition(vec);
      vec.y += 6;
      vec.project(camera);
      const x = (vec.x * 0.5 + 0.5) * window.innerWidth;
      const y = (-vec.y * 0.5 + 0.5) * window.innerHeight;
      const behind = vec.z > 1 || vec.z < -1;
      element.style.left = `${x}px`;
      element.style.top = `${y}px`;
      element.style.display = behind ? 'none' : 'block';
    }
  }

  setCurrentRoom(starId: string): void {
    for (const { node, element } of this.labels) {
      const active = node.starId === starId;
      element.style.opacity = active ? '1' : '0.5';
      element.style.fontSize = active ? '15px' : '13px';
      element.style.color = active ? '#00ddff' : 'white';
    }
  }

  destroy(): void {
    this.container.remove();
    this.labels = [];
  }
}
