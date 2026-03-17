import * as THREE from 'three';

import type { HouseNode } from './loadHouseScene';

type CameraControls = {
  target: THREE.Vector3;
  update(): void;
};

export class RoomCamera {
  private camera: THREE.PerspectiveCamera;
  private controls: CameraControls;
  private rooms: Map<string, HouseNode>;
  private currentRoomId: string = '';
  private transitioning = false;
  private elapsed = 0;
  private duration = 1.0;
  private startPosition = new THREE.Vector3();
  private startTarget = new THREE.Vector3();
  private destinationPosition = new THREE.Vector3();
  private destinationTarget = new THREE.Vector3();
  private readonly offset = new THREE.Vector3(0, 8, 12);

  constructor(
    camera: THREE.PerspectiveCamera,
    controls: CameraControls,
    rooms: Iterable<HouseNode>,
    initialRoomId?: string,
  ) {
    this.camera = camera;
    this.controls = controls;
    this.rooms = new Map(Array.from(rooms).map((room) => [room.starId, room]));
    const fallback = initialRoomId && this.rooms.has(initialRoomId)
      ? initialRoomId
      : (Array.from(this.rooms.keys())[0] || '');
    if (fallback) {
      this.snapToRoom(fallback);
    }
  }

  get currentRoom(): string {
    return this.currentRoomId;
  }

  private roomCenter(room: HouseNode): THREE.Vector3 {
    const world = new THREE.Vector3();
    room.object.getWorldPosition(world);
    return world;
  }

  snapToRoom(roomStarId: string): void {
    const room = this.rooms.get(roomStarId);
    if (!room) {
      return;
    }
    const center = this.roomCenter(room);
    this.currentRoomId = roomStarId;
    this.transitioning = false;
    this.controls.target.copy(center);
    this.camera.position.copy(center.clone().add(this.offset));
    this.controls.update();
  }

  goToRoom(roomStarId: string): void {
    const room = this.rooms.get(roomStarId);
    if (!room) {
      return;
    }
    const center = this.roomCenter(room);
    this.currentRoomId = roomStarId;
    this.transitioning = true;
    this.elapsed = 0;
    this.startPosition.copy(this.camera.position);
    this.startTarget.copy(this.controls.target);
    this.destinationTarget.copy(center);
    this.destinationPosition.copy(center).add(this.offset);
  }

  update(delta: number): void {
    if (!this.transitioning) {
      return;
    }
    this.elapsed += Math.max(0, delta);
    const t = Math.min(1, this.elapsed / this.duration);
    this.camera.position.lerpVectors(this.startPosition, this.destinationPosition, t);
    this.controls.target.lerpVectors(this.startTarget, this.destinationTarget, t);
    this.controls.update();
    if (t >= 1) {
      this.transitioning = false;
    }
  }
}
