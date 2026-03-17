import type { HouseNode } from '../loadHouseScene';

export class RoomContext {
  private currentRoom: HouseNode | null = null;
  private onRoomChange: Array<(room: HouseNode) => void> = [];

  setRoom(room: HouseNode): void {
    if (this.currentRoom?.starId === room.starId) return;
    this.currentRoom = room;
    for (const callback of this.onRoomChange) callback(room);
  }

  onEnter(callback: (room: HouseNode) => void): void {
    this.onRoomChange.push(callback);
  }

  get current(): HouseNode | null {
    return this.currentRoom;
  }
}
