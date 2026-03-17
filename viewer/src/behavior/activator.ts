import type { HouseNode, LoadedHouseScene } from '../loadHouseScene';
import { RoomCamera } from '../roomCamera';
import { Tablet3D } from '../tablet';

import { renderNodeContent, renderContentPayload } from './contentRenderer';
import { interpretBehavior } from './interpreter';
import { RoomContext } from './roomContext';

export class HouseActivator {
  private scene: LoadedHouseScene;
  private roomCamera: RoomCamera;
  private tablet: Tablet3D;
  private roomContext: RoomContext;

  constructor(scene: LoadedHouseScene, roomCamera: RoomCamera, tablet: Tablet3D, roomContext: RoomContext) {
    this.scene = scene;
    this.roomCamera = roomCamera;
    this.tablet = tablet;
    this.roomContext = roomContext;
  }

  activate(node: HouseNode): void {
    const action = interpretBehavior(node.behaviorRpn, node);
    switch (action.type) {
      case 'door_traverse':
        this.handleDoorTraverse(action.roomA, action.roomB);
        break;
      case 'load_galaxy':
        this.handleLoadGalaxy(action.galaxyRef, node);
        break;
      case 'inspect_object':
        this.handleInspect(node);
        break;
      case 'activate_display':
        this.handleDisplay(action.taxonomyRefs, node);
        break;
      case 'browse_galaxy':
        this.handleBrowseGalaxy();
        break;
      case 'room_enter':
        this.handleRoomEnter(node, action.room, action.domain);
        break;
      case 'noop':
        break;
    }
  }

  private handleDoorTraverse(roomA: string, roomB: string): void {
    const currentRoomNode = this.scene.nodesByStarId.get(this.scene.currentRoom) || null;
    const currentHouseRoom = currentRoomNode?.houseRoom || '';
    const targetHouseRoom = currentHouseRoom === roomA ? roomB : roomA;
    const targetRoom = this.scene.rooms.find((room) => room.houseRoom === targetHouseRoom);
    if (!targetRoom) return;
    this.roomCamera.goToRoom(targetRoom.starId);
    this.scene.currentRoom = targetRoom.starId;
    this.roomContext.setRoom(targetRoom);
  }

  private handleLoadGalaxy(galaxyRef: string, node: HouseNode): void {
    this.tablet.showFocus();
    this.tablet.dispatch({ type: 'open_app', payload: { id: 'content' } });
    this.tablet.dispatch({
      type: 'showContent',
      payload: renderContentPayload({
        node,
        galaxyRef,
        title: node.surfaceForms.en?.word_ref || node.starId,
      }),
    });
  }

  private handleInspect(node: HouseNode): void {
    this.tablet.showFocus();
    this.tablet.dispatch({ type: 'open_app', payload: { id: 'content' } });
    this.tablet.dispatch({
      type: 'showContent',
      payload: renderNodeContent(node),
    });
  }

  private handleDisplay(taxonomyRefs: string[], node: HouseNode): void {
    this.tablet.showFocus();
    this.tablet.dispatch({ type: 'open_app', payload: { id: 'content' } });
    this.tablet.dispatch({
      type: 'showContent',
      payload: renderContentPayload({
        node,
        title: node.surfaceForms.en?.word_ref || node.starId,
        taxonomyRefs,
      }),
    });
  }

  private handleBrowseGalaxy(): void {
    this.tablet.showFocus();
    this.tablet.dispatch({ type: 'open_app', payload: { id: 'galaxy' } });
  }

  private handleRoomEnter(node: HouseNode, room: string, domain: string): void {
    if (node.meaningClass === 'room') {
      this.roomCamera.goToRoom(node.starId);
      this.scene.currentRoom = node.starId;
      this.roomContext.setRoom(node);
    }
    this.tablet.dispatch({
      type: 'roomContext',
      payload: { room, domain, title: node.surfaceForms.en?.word_ref || node.starId },
    });
  }
}
