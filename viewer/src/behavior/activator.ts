import type { HouseNode, LoadedHouseScene } from '../loadHouseScene';
import { getBookContent, getLoadedContent } from '../contentLoader';
import type { GalaxyPodProjector, HolodeskProjector } from '../projection';
import { rpnToMesh } from '../rpn';
import { RoomCamera } from '../roomCamera';
import { Tablet3D } from '../tablet';

import { renderBookContent, renderNodeContent, renderContentPayload } from './contentRenderer';
import { interpretBehavior } from './interpreter';
import { RoomContext } from './roomContext';

export class HouseActivator {
  private scene: LoadedHouseScene;
  private roomCamera: RoomCamera;
  private tablet: Tablet3D;
  private roomContext: RoomContext;
  private holodesk: HolodeskProjector | null;
  private galaxyPod: GalaxyPodProjector | null;
  private lastProjectedNode: HouseNode | null = null;

  constructor(
    scene: LoadedHouseScene,
    roomCamera: RoomCamera,
    tablet: Tablet3D,
    roomContext: RoomContext,
    options?: {
      holodesk?: HolodeskProjector | null;
      galaxyPod?: GalaxyPodProjector | null;
    },
  ) {
    this.scene = scene;
    this.roomCamera = roomCamera;
    this.tablet = tablet;
    this.roomContext = roomContext;
    this.holodesk = options?.holodesk || null;
    this.galaxyPod = options?.galaxyPod || null;
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
    this.rememberProjectedNode(node);
    this.tablet.showFocus();
    this.tablet.dispatch({ type: 'open_app', payload: { id: 'content' } });
    const bookContent = getBookContent(galaxyRef);
    this.tablet.dispatch({
      type: 'showContent',
      payload: bookContent
        ? renderBookContent(galaxyRef, bookContent, node)
        : renderContentPayload({
            node,
            galaxyRef,
            title: node.surfaceForms.en?.word_ref || node.starId,
          }),
    });
  }

  private handleInspect(node: HouseNode): void {
    if (node.starId === 'furniture_bathtub' && this.galaxyPod) {
      if (this.galaxyPod.visible) {
        this.galaxyPod.dismiss();
      } else {
        this.galaxyPod.projectFromContent(getLoadedContent());
      }
    }
    this.rememberProjectedNode(node);
    this.tablet.showFocus();
    this.tablet.dispatch({ type: 'open_app', payload: { id: 'content' } });
    this.tablet.dispatch({
      type: 'showContent',
      payload: renderNodeContent(node),
    });
  }

  private handleDisplay(taxonomyRefs: string[], node: HouseNode): void {
    this.rememberProjectedNode(node);
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
    if (this.holodesk) {
      if (this.holodesk.visible) {
        this.holodesk.dismiss();
      } else {
        const preferred = this.lastProjectedNode?.visualRpn
          ? this.lastProjectedNode
          : this.scene.nodesByStarId.get('furniture_knowledge_tree') || null;
        if (preferred?.visualRpn) {
          this.holodesk.projectNodeVisual(preferred, rpnToMesh);
        }
      }
    }
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

  private rememberProjectedNode(node: HouseNode): void {
    if (!node.visualRpn) return;
    if (node.starId === 'furniture_holodesk') return;
    this.lastProjectedNode = node;
  }
}
