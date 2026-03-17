import * as THREE from 'three';

import type { HouseNode } from '../loadHouseScene';
import { ProjectionSurface } from './surface';

export class HolodeskProjector {
  readonly surface: ProjectionSurface;
  private rotationSpeed = 0.15;
  private currentNodeId: string | null = null;

  constructor(holodeskNode: HouseNode) {
    this.surface = new ProjectionSurface({
      anchor: new THREE.Vector3(0, 0.6, 0),
      bounds: new THREE.Vector3(1.4, 1.0, 0.8),
      mode: 'holographic',
    });
    holodeskNode.object.add(this.surface.group);
  }

  get visible(): boolean {
    return this.surface.visible;
  }

  projectMesh(mesh: THREE.Object3D, nodeId = 'custom'): void {
    const projected = mesh.clone(true);
    projected.traverse((child: THREE.Object3D) => {
      if (child instanceof THREE.Mesh) {
        child.material = new THREE.MeshBasicMaterial({
          color: 0x00ddff,
          wireframe: true,
          transparent: true,
          opacity: 0.7,
        });
      }
    });
    this.currentNodeId = nodeId;
    this.surface.setContent(projected);
    this.surface.show();
  }

  projectNodeVisual(node: HouseNode, rpnToMesh: (program: string) => THREE.Object3D): boolean {
    if (!node.visualRpn) {
      return false;
    }
    const mesh = rpnToMesh(node.visualRpn);
    this.projectMesh(mesh, node.starId);
    return true;
  }

  toggleNodeVisual(node: HouseNode, rpnToMesh: (program: string) => THREE.Object3D): boolean {
    if (this.surface.visible && this.currentNodeId === node.starId) {
      this.dismiss();
      return false;
    }
    if (!this.surface.visible && this.currentNodeId === node.starId) {
      this.surface.show();
      return true;
    }
    return this.projectNodeVisual(node, rpnToMesh);
  }

  dismiss(): void {
    this.surface.hide();
  }

  update(delta: number): void {
    this.surface.update(delta);
    if (this.surface.group.visible) {
      this.surface.group.rotation.y += this.rotationSpeed * delta;
    }
  }
}
