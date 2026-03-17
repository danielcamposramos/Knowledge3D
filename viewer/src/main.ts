import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { HonestyMaterial } from './HonestyMaterial';
import { loadK3DFromGLTF, fetchCondoConfig, type K3DRecord, type CondoConfig, type LoadedK3D } from './loadK3D';
import { HouseActivator, RoomContext } from './behavior';
import { loadHouseScene, type HouseNode, type LoadedHouseScene } from './loadHouseScene';
import { RoomCamera } from './roomCamera';
import { K3DAgent } from './agent';
import { Tablet3D } from './tablet';
import { ChatClient, type ChatMessage, type CommandMessage } from './chat';
import { RPN } from './rpn';
import { openStore } from './cache';
import { kmeans, palette } from './cluster';
import { AISuggestionManager, DynamicLayerManager, LODRenderer, GridCulledPoints } from './extensions/smartGraph';
import { buildInstancedStars, buildInstancedBranches } from './shapes';

// --- DOM Elements ---
const canvas = document.getElementById('scene') as HTMLCanvasElement;
const tooltip = document.getElementById('tooltip') as HTMLDivElement;
const expertSelect = document.getElementById('expert-select') as HTMLSelectElement;
const hudLegend = document.getElementById('hud-legend') as HTMLDivElement | null;

// --- Scene Setup ---
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 0, 5);
const controls = new OrbitControls(camera, renderer.domElement);

// --- State ---
let k3dData: K3DRecord[] = [];
let recordMap: Map<string, K3DRecord> = new Map();
let currentPoints: THREE.Object3D | null = null;
let condoConfig: CondoConfig | null = null;
let agent: K3DAgent | null = null;
let cotOverlay: THREE.Group | null = null;
let chat: ChatClient | null = null;
const cache = openStore<LoadedK3D>();
const tabletStore = openStore<any>('k3d-tablet','tablet');
let tablet: Tablet3D | null = null;
let cacheEnabled = true;
let sugg: AISuggestionManager | null = null;
let layersMgr: DynamicLayerManager | null = null;
let lod: LODRenderer | null = null;
let lastHoverRecord: K3DRecord | null = null;
let layersOverlay: HTMLDivElement | null = null;
let edgesObject: THREE.LineSegments | null = null;
let gardenInstanced: THREE.Group | null = null;
let loadedHouseScene: LoadedHouseScene | null = null;
let roomCamera: RoomCamera | null = null;
let houseActivator: HouseActivator | null = null;
let roomContext: RoomContext | null = null;
// LOD HUD
const lodHud = document.createElement('div');
lodHud.id = 'lod-hud';
lodHud.style.position = 'absolute';
lodHud.style.top = '10px';
lodHud.style.right = '10px';
lodHud.style.background = 'rgba(0,0,0,0.6)';
lodHud.style.color = '#eee';
lodHud.style.fontSize = '12px';
lodHud.style.padding = '4px 6px';
lodHud.style.borderRadius = '4px';
lodHud.style.display = 'none';
lodHud.textContent = 'LOD: —';
document.body.appendChild(lodHud);
const lodToggle = document.getElementById('toggle-lod-hud') as HTMLInputElement | null;
if (lodToggle) lodToggle.onchange = () => { lodHud.style.display = lodToggle.checked ? 'block' : 'none'; };

// --- Boot Overlay (Flash-style preload shell) ---
const bootOverlay = document.createElement('div');
bootOverlay.id = 'boot-overlay';
bootOverlay.innerHTML = `
  <div id="boot-panel">
    <div id="boot-kicker">K3D Boot Sequence</div>
    <div id="boot-title">Preloading Procedural Space</div>
    <div id="boot-status">Starting scene bootstrap…</div>
    <div id="boot-progress-shell"><div id="boot-progress-bar"></div></div>
    <div id="boot-meta">
      <span id="boot-stage">stage: cold-start</span>
      <span id="boot-percent">0%</span>
    </div>
  </div>
`;
document.body.appendChild(bootOverlay);
const bootStatus = bootOverlay.querySelector('#boot-status') as HTMLDivElement;
const bootStage = bootOverlay.querySelector('#boot-stage') as HTMLSpanElement;
const bootPercent = bootOverlay.querySelector('#boot-percent') as HTMLSpanElement;
const bootProgressBar = bootOverlay.querySelector('#boot-progress-bar') as HTMLDivElement;
bootOverlay.style.transition = 'opacity 240ms ease';

let bootLocalProgress = 0.0;
let bootLocalStatus = 'Starting scene bootstrap…';
let bootLocalStage = 'stage: cold-start';
let bootRemoteProgress = 0.0;
let bootRemoteStatus = '';
let bootRemoteStage = '';
let bootPollHandle: number | null = null;
let bootFinished = false;

function refreshBootOverlay() {
    const progress = Math.max(bootLocalProgress, bootRemoteProgress);
    const statusText = bootRemoteStatus || bootLocalStatus;
    const stageText = bootRemoteStage || bootLocalStage;
    bootStatus.textContent = statusText;
    bootStage.textContent = stageText;
    bootPercent.textContent = `${Math.round(progress * 100)}%`;
    bootProgressBar.style.width = `${Math.round(progress * 100)}%`;
}

function setBootLocal(progress: number, statusText: string, stageText: string) {
    bootLocalProgress = Math.max(bootLocalProgress, Math.min(1.0, progress));
    bootLocalStatus = statusText;
    bootLocalStage = stageText;
    if (!bootFinished) refreshBootOverlay();
}

async function pollRuntimeBootStatus() {
    try {
        const res = await fetch(`/runtime_boot.json?ts=${Date.now()}`, { cache: 'no-store' });
        if (!res.ok) return;
        const payload = await res.json();
        const progress = typeof payload?.progress === 'number' ? payload.progress : 0.0;
        bootRemoteProgress = Math.max(bootRemoteProgress, Math.min(1.0, progress));
        const stage = String(payload?.stage || '').trim();
        const state = String(payload?.state || '').trim();
        bootRemoteStage = stage ? `stage: ${stage}` : bootRemoteStage;
        bootRemoteStatus = stage
            ? `${stage.replace(/_/g, ' ')}${state ? ` (${state})` : ''}`
            : bootRemoteStatus;
        if (!bootFinished) refreshBootOverlay();
    } catch {}
}

function startBootPolling() {
    if (bootPollHandle !== null) return;
    bootPollHandle = window.setInterval(() => { void pollRuntimeBootStatus(); }, 500);
    void pollRuntimeBootStatus();
}

function stopBootPolling() {
    if (bootPollHandle !== null) {
        window.clearInterval(bootPollHandle);
        bootPollHandle = null;
    }
}

function finishBootOverlay(statusText = 'K3D ready.') {
    if (bootFinished) return;
    bootFinished = true;
    bootRemoteProgress = Math.max(bootRemoteProgress, 1.0);
    bootRemoteStatus = statusText;
    bootRemoteStage = 'stage: ready';
    refreshBootOverlay();
    window.setTimeout(() => {
        bootOverlay.style.opacity = '0';
        window.setTimeout(() => {
            bootOverlay.style.display = 'none';
        }, 260);
    }, 180);
    stopBootPolling();
}

// --- Main Logic ---

/**
 * Clears the current 3D scene of any house-related objects.
 */
function clearScene() {
    if (loadedHouseScene) {
        scene.remove(loadedHouseScene.root);
        loadedHouseScene = null;
    }
    roomCamera = null;
    houseActivator = null;
    roomContext = null;
    if (currentPoints) {
        scene.remove(currentPoints);
        try { (currentPoints as any).geometry?.dispose?.(); } catch {}
        try { (currentPoints as any).material?.dispose?.(); } catch {}
        currentPoints = null;
    }
    if (edgesObject) {
        scene.remove(edgesObject);
        (edgesObject.geometry as THREE.BufferGeometry).dispose();
        (edgesObject.material as THREE.Material).dispose();
        edgesObject = null;
    }
    if (gardenInstanced) {
        scene.remove(gardenInstanced);
        gardenInstanced.children.forEach((ch:any) => {
            try { (ch as any).geometry?.dispose?.(); } catch {}
            try { (ch as any).material?.dispose?.(); } catch {}
        });
        gardenInstanced = null;
    }
    lod = null;
    layersMgr = null;
    k3dData = [];
    recordMap.clear();
    if (agent) {
        scene.remove(agent.object);
        agent = null;
    }
    if (cotOverlay) {
        try {
            cotOverlay.traverse((obj: any) => {
                try { obj.geometry?.dispose?.(); } catch {}
                try { obj.material?.dispose?.(); } catch {}
            });
        } catch {}
        scene.remove(cotOverlay);
        cotOverlay = null;
    }
    if (chat) {
        chat.disconnect();
        chat = null;
    }
}

function isHouseGlbUrl(url: string): boolean {
    const value = String(url || '').trim().toLowerCase();
    return value === 'house' || value.endsWith('/house.glb') || value.endsWith('house.glb');
}

function houseNodeLabel(node: HouseNode): string {
    return node.surfaceForms.en?.word_ref || node.surfaceForms.pt?.word_ref || node.starId;
}

function cloneMaterialForHouse(mesh: THREE.Mesh) {
    const anyMesh = mesh as any;
    if (anyMesh.userData?.k3dMaterialCloned) return;
    if (Array.isArray(mesh.material)) {
        mesh.material = mesh.material.map((material) => material.clone());
    } else if (mesh.material) {
        mesh.material = mesh.material.clone();
    }
    anyMesh.userData = { ...(anyMesh.userData || {}), k3dMaterialCloned: true };
}

function setNodeVisualOpacity(node: HouseNode, opacity: number, transparent: boolean) {
    node.object.traverse((child: any) => {
        if (!child?.isMesh || !child.material) return;
        const mesh = child as THREE.Mesh;
        cloneMaterialForHouse(mesh);
        const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
        for (const material of materials) {
            if (!material) continue;
            material.transparent = transparent;
            material.opacity = opacity;
            material.needsUpdate = true;
        }
    });
}

function buildDoorItemsForRoom(targetRoom: HouseNode | null): Array<{ label: string; address?: string; starId?: string; roomA?: string; roomB?: string }> {
    if (!loadedHouseScene) return [];
    const currentRoomStarId = targetRoom?.starId || loadedHouseScene.currentRoom;
    const currentHouseRoom = targetRoom?.houseRoom || loadedHouseScene.nodesByStarId.get(currentRoomStarId)?.houseRoom || '';
    const visibleDoors = loadedHouseScene.doors.filter((door) => {
        const edgeMatch = loadedHouseScene!.navGraph.edges.some((edge) =>
            edge.door === door.starId && (edge.from === currentRoomStarId || edge.to === currentRoomStarId)
        );
        if (edgeMatch) return true;
        const tokens = String(door.behaviorRpn || '').split(/\s+/).filter(Boolean);
        return tokens[2] === currentHouseRoom || tokens[3] === currentHouseRoom;
    });
    return visibleDoors.map((door) => {
        const tokens = String(door.behaviorRpn || '').split(/\s+/).filter(Boolean);
        const roomA = tokens[2] || '';
        const roomB = tokens[3] || '';
        return {
            label: houseNodeLabel(door),
            address: roomA && roomB ? `${roomA} <-> ${roomB}` : door.starId,
            starId: door.starId,
            roomA,
            roomB,
        };
    });
}

function applyRoomDimming(currentRoom: HouseNode | null) {
    if (!loadedHouseScene || !currentRoom) return;
    loadedHouseScene.nodesByStarId.forEach((node) => {
        if (node.meaningClass === 'room') {
            setNodeVisualOpacity(node, 1.0, false);
            return;
        }
        const alwaysVisible = node.houseRoom === 'House';
        const inCurrentRoom = alwaysVisible || node.houseRoom === currentRoom.houseRoom;
        setNodeVisualOpacity(node, inCurrentRoom ? 1.0 : 0.3, !inCurrentRoom);
    });
}

function applyRoomContext(currentRoom: HouseNode | null) {
    if (!tablet || !loadedHouseScene || !currentRoom) return;
    applyRoomDimming(currentRoom);
    loadedHouseScene.currentRoom = currentRoom.starId;
    tablet.dispatch({
        type: 'roomContext',
        payload: {
            room: currentRoom.houseRoom,
            domain: currentRoom.domain,
            title: houseNodeLabel(currentRoom),
        },
    });
    tablet.dispatch({ type: 'doors_list', payload: { items: buildDoorItemsForRoom(currentRoom) } });
    tablet.setStatus({
        house: currentHouseUrl,
        nodes: loadedHouseScene.nodesByStarId.size,
        info: `room=${currentRoom.starId}`,
    });
}

function findHouseNodeForObject(object: THREE.Object3D | null): HouseNode | null {
    if (!loadedHouseScene || !object) return null;
    let cursor: THREE.Object3D | null = object;
    while (cursor) {
        const starId = String((cursor.userData as any)?.k3d?.star_id || '').trim();
        if (starId) {
            return loadedHouseScene.nodesByStarId.get(starId) || null;
        }
        cursor = cursor.parent;
    }
    return null;
}

function showHouseTooltip(node: HouseNode) {
    const label = houseNodeLabel(node);
    const visual = String(node.visualRpn || '').trim();
    const preview = visual.length > 72 ? `${visual.slice(0, 69)}...` : visual;
    const behavior = String(node.behaviorRpn || '').trim();
    const behaviorPreview = behavior.length > 56 ? `${behavior.slice(0, 53)}...` : behavior;
    tooltip.style.display = 'block';
    tooltip.style.left = `${mouse.x * window.innerWidth / 2 + window.innerWidth / 2 + 5}px`;
    tooltip.style.top = `${-mouse.y * window.innerHeight / 2 + window.innerHeight / 2 + 5}px`;
    tooltip.innerHTML = [
        `<div><strong>${label}</strong></div>`,
        `<div style="margin-top:4px; font-size:12px; color:#333;">${node.meaningClass}</div>`,
        preview ? `<div style="margin-top:4px; font-size:11px; color:#555;">${preview}</div>` : '',
        behaviorPreview ? `<div style="margin-top:2px; font-size:11px; color:#666;">${behaviorPreview}</div>` : '',
    ].join('');
    if (tablet) tablet.setFocusLabel(label);
}

function onSceneClick(event: MouseEvent) {
    if (!loadedHouseScene || event.target !== canvas) return;
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const houseHits = raycaster.intersectObjects([loadedHouseScene.root], true);
    const houseNode = houseHits
        .map((hit) => findHouseNodeForObject(hit.object))
        .find((node): node is HouseNode => node !== null);
    if (!houseNode) return;
    houseActivator?.activate(houseNode);
}

async function loadHouseSceneAsset(url: string) {
    clearScene();
    currentHouseUrl = url;
    try {
        loadedHouseScene = await loadHouseScene(url);
        scene.add(loadedHouseScene.root);
        roomCamera = new RoomCamera(camera, controls, loadedHouseScene.rooms, loadedHouseScene.currentRoom || 'room_library');
        roomCamera.snapToRoom(loadedHouseScene.currentRoom || 'room_library');
        loadedHouseScene.currentRoom = roomCamera.currentRoom;

        if (!tablet) {
            tablet = new Tablet3D();
            scene.add(tablet.object);
        }
        tablet.setStatus({
            house: url,
            nodes: loadedHouseScene.nodesByStarId.size,
            info: `rooms=${loadedHouseScene.rooms.length} doors=${loadedHouseScene.doors.length}`,
        });
        tablet.setDataset([]);

        roomContext = new RoomContext();
        houseActivator = new HouseActivator(loadedHouseScene, roomCamera, tablet, roomContext);
        roomContext.onEnter((room) => {
            applyRoomContext(room);
        });

        tablet.setLocalHandler?.((ev: any) => {
            if (ev?.type !== 'openDoor' || !loadedHouseScene || !houseActivator) return;
            const payload = ev.payload || {};
            const starId = String(payload.starId || '').trim();
            let door = starId ? loadedHouseScene.nodesByStarId.get(starId) || null : null;
            if (!door) {
                const label = String(payload.label || '').trim();
                door = loadedHouseScene.doors.find((candidate) => houseNodeLabel(candidate) === label) || null;
            }
            if (!door) return;
            houseActivator.activate(door);
        });

        const initialRoom = loadedHouseScene.nodesByStarId.get(loadedHouseScene.currentRoom) || loadedHouseScene.rooms[0] || null;
        if (initialRoom && roomContext) {
            roomContext.setRoom(initialRoom);
        }
    } catch (e) {
        console.error(`Failed to load house scene from ${url}:`, e);
    }
}

/**
 * Loads and displays a house from the given K3D data URL.
 * @param k3dUrl The URL of the .k3d file to load.
 */
async function loadHouse(k3dUrl: string) {
    clearScene();
    currentHouseUrl = k3dUrl;

    try {
        let loaded: LoadedK3D | null = null;
        if (cacheEnabled) loaded = await cache.get(k3dUrl);
        if (!loaded) {
            loaded = await loadK3DFromGLTF(k3dUrl);
            if (cacheEnabled) await cache.put(k3dUrl, loaded);
        }
        k3dData = loaded.data;
        if (k3dData.length === 0) {
            console.warn(`No data found in ${k3dUrl}`);
            return;
        }
        recordMap = new Map(k3dData.map((r) => [r.id, r]));

        const positions = new Float32Array(k3dData.length * 3);
        const colors = new Float32Array(k3dData.length * 3);
        const color = new THREE.Color();

        const vectors = k3dData.map((d) => d.vector);
        const min = new THREE.Vector3(Infinity, Infinity, Infinity);
        const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
        vectors.forEach(vec => {
            min.min(new THREE.Vector3().fromArray(vec));
            max.max(new THREE.Vector3().fromArray(vec));
        });
        const size = new THREE.Vector3().subVectors(max, min);

        const mask = loaded.info?.ai?.mask?.has_new_information;
        const tmask = loaded.info?.temporal?.alphaMask;
        k3dData.forEach((record, i) => {
            positions.set(record.vector, i * 3);

            // Uniform coloring by position; optional AI per-node mask tints via multiplier below
            if (mask && mask[i] === true) {
                // AI per-node cue: green for nodes with new info
                color.setRGB(0.0, 1.0, 0.0);
            } else {
                const r = size.x > 0 ? (record.vector[0] - min.x) / size.x : 0.5;
                const g = size.y > 0 ? (record.vector[1] - min.y) / size.y : 0.5;
                const b = size.z > 0 ? (record.vector[2] - min.z) / size.z : 0.5;
                color.setRGB(r, g, b);
            }
            // Temporal per-node dimming via alphaMask
            if (tmask && typeof tmask[i] === 'number') {
                const a = Math.max(0, Math.min(1, tmask[i]));
                color.multiplyScalar(a);
            }
            colors.set([color.r, color.g, color.b], i * 3);
        });

        const baseGeom = new THREE.BufferGeometry();
        baseGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        baseGeom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        baseGeom.computeBoundingSphere();

        const material = new THREE.PointsMaterial({ size: 0.1, vertexColors: true });
        const talpha = loaded.info?.temporal?.alpha;
        if (typeof talpha === 'number') {
            material.transparent = talpha < 1.0;
            material.opacity = Math.max(0, Math.min(1, talpha));
        }
        // For larger datasets, use grid culling; otherwise use LOD
        if (k3dData.length > 8000) {
            const grid = new GridCulledPoints(baseGeom, material, 4);
            currentPoints = grid.attach(scene);
            // piggyback update call via lod reference (optional)
            lod = null;
        } else {
            lod = new LODRenderer(baseGeom, material);
            currentPoints = lod.attach(scene);
        }

        // AI-native visual hint (global): if has_new_information on primitive and no mask provided
        if (loaded.info?.ai?.flags?.has_new_information && !loaded.info?.ai?.mask?.has_new_information && currentPoints) {
            try {
                if ((currentPoints as any).isPoints) {
                    const colorsAttr = ((currentPoints as THREE.Points).geometry as THREE.BufferGeometry).getAttribute('color') as THREE.BufferAttribute;
                    for (let i = 0; i < colorsAttr.count; i++) colorsAttr.setXYZ(i, 0.0, 1.0, 0.0);
                    colorsAttr.needsUpdate = true;
                }
                // Skip group retint for grid-culling to keep perf simple
            } catch {}
        }

        // Near-field shapes + rays for better semantics (meaning-driven; modality shown as rays)
        try {
          const posArr = (currentPoints as any).geometry.getAttribute('position').array as Float32Array;
          // Build detail layer for a capped set to avoid perf issues
          buildInstancedStars(k3dData, posArr, scene, Math.min(2000, k3dData.length));
        } catch {}

        // dataset info
        const infoEl = document.getElementById('dataset-info') as HTMLDivElement;
        const inf = loaded.info;
        const fmt = (b?: number) => b !== undefined ? `${(b/1e6).toFixed(2)} MB` : 'n/a';
        infoEl.textContent = `precision=${inf.precision} dims=${inf.dims} count=${inf.count} vectors=${fmt(inf.byteLengthVectors)} embeddings=${fmt(inf.byteLengthEmbeddings)}`;
        // quick scoreboard lines (reuse earlier mask)
        const guided = Array.isArray(mask) ? mask.filter(Boolean).length : 0;
        const doorsCount = k3dData.filter(r => (r.metadata?.type as string) === 'door').length;
        if (tablet) {
            tablet.pushExplain(`House loaded: count=${k3dData.length} dims=${inf.dims}`);
            tablet.pushExplain(`Doors=${doorsCount} Guided=${guided}`);
            tablet.dispatch({ type: 'dataset_summary', payload: { house: k3dUrl, nodes: k3dData.length, dims: inf.dims, doors: doorsCount, guided } });
        }
        // try fetch viewer/public training scoreboard
        try {
            const res = await fetch('/training/latest.json', { cache: 'no-store' });
            if (res.ok) {
                const s = await res.json();
                tablet?.pushExplain(`Scoreboard ts=${s.ts} GOTO: ${s.goto.success}/${s.goto.count} (med=${s.goto.median_hops}) DOOR: ${s.door.success}/${s.door.count} (med=${s.door.median_hops})`);
                tablet?.dispatch({ type: 'scoreboard_summary', payload: s });
            }
        } catch {}

        // Initialize or reset the agent
        const explainLog = document.getElementById('explain-log') as HTMLDivElement;
        const pushExplain = (text: string) => {
            const el = document.createElement('div');
            el.textContent = text;
            explainLog.appendChild(el);
            explainLog.scrollTop = explainLog.scrollHeight;
            // also emit to live server for logging/analysis
            if (chat) {
                chat.sendEvent({ kind: 'explain', text });
            }
            if (tablet) tablet.pushExplain(text);
        };
        agent = new K3DAgent(scene, camera, pushExplain);
        agent.setRecords(k3dData);

        // Optional edges rendering (e.g., Knowledge Gardens ontology)
        if (loaded.edges && loaded.edges.length > 0) {
            const idToIndex = new Map<string, number>();
            k3dData.forEach((r, i) => idToIndex.set(r.id, i));
            const E = loaded.edges.filter(([a,b]) => idToIndex.has(a) && idToIndex.has(b));
            const edgePos = new Float32Array(E.length * 2 * 3);
            let ptr = 0;
            for (const [a, b] of E) {
                const ia = idToIndex.get(a)!;
                const ib = idToIndex.get(b)!;
                const ax = positions[ia*3+0], ay = positions[ia*3+1], az = positions[ia*3+2];
                const bx = positions[ib*3+0], by = positions[ib*3+1], bz = positions[ib*3+2];
                edgePos[ptr++] = ax; edgePos[ptr++] = ay; edgePos[ptr++] = az;
                edgePos[ptr++] = bx; edgePos[ptr++] = by; edgePos[ptr++] = bz;
            }
            const eg = new THREE.BufferGeometry();
            eg.setAttribute('position', new THREE.BufferAttribute(edgePos, 3));
            const edgeColor = loaded.info?.edgeColor;
            const edgeCol = edgeColor ? new THREE.Color(edgeColor[0], edgeColor[1], edgeColor[2]) : new THREE.Color(0x66cc66);
            const emat = new THREE.LineBasicMaterial({ color: edgeCol, transparent: true, opacity: 0.25 });
            const lines = new THREE.LineSegments(eg, emat);
            // draw slightly behind points to reduce overdraw
            (lines.material as THREE.LineBasicMaterial).depthWrite = false;
            scene.add(lines);
            edgesObject = lines;
            // Near-view upgrade: instanced branches + leaves (cap to keep light)
            try {
                const eIdx: Array<[number, number]> = E.slice(0, Math.min(E.length, 5000)).map(([a,b]) => [idToIndex.get(a)!, idToIndex.get(b)!]);
                gardenInstanced = buildInstancedBranches(positions, eIdx, scene, 5000);
            } catch {}
        }

        if (loaded.teacherEdges && loaded.teacherEdges.length > 0) {
            const idToIndex = new Map<string, number>();
            k3dData.forEach((r, i) => idToIndex.set(r.id, i));
            const E = loaded.teacherEdges.filter(([a,b]) => idToIndex.has(a) && idToIndex.has(b));
            if (E.length > 0) {
                const edgePos = new Float32Array(E.length * 2 * 3);
                let ptr = 0;
                for (const [a, b] of E) {
                    const ia = idToIndex.get(a)!;
                    const ib = idToIndex.get(b)!;
                    const ax = positions[ia*3+0], ay = positions[ia*3+1], az = positions[ia*3+2];
                    const bx = positions[ib*3+0], by = positions[ib*3+1], bz = positions[ib*3+2];
                    edgePos[ptr++] = ax; edgePos[ptr++] = ay; edgePos[ptr++] = az;
                    edgePos[ptr++] = bx; edgePos[ptr++] = by; edgePos[ptr++] = bz;
                }
                const eg = new THREE.BufferGeometry();
                eg.setAttribute('position', new THREE.BufferAttribute(edgePos, 3));
                const edgeColor = loaded.info?.teacherEdgeColor;
                const edgeCol = edgeColor ? new THREE.Color(edgeColor[0], edgeColor[1], edgeColor[2]) : new THREE.Color(0x00ffff);
                const emat = new THREE.LineBasicMaterial({ color: edgeCol, transparent: true, opacity: 0.35 });
                const lines = new THREE.LineSegments(eg, emat);
                (lines.material as THREE.LineBasicMaterial).depthWrite = false;
                scene.add(lines);
            }
        }

        // Setup tablet (3D object) if not present
        if (!tablet) {
            tablet = new Tablet3D();
            scene.add(tablet.object);
        }
        // Forward tablet app events to live server for session logging
        tablet.setEmitter?.((ev: any) => { if (chat) chat.sendEvent(ev); });
        // Local handler for in-world actions from Tablet apps (e.g., Layers)
        tablet.setLocalHandler?.((ev: any) => {
            if (ev?.type === 'applyLayers' && layersMgr) {
                try {
                    const names: string[] = Array.isArray(ev.payload?.enabled) ? ev.payload.enabled : [];
                    // rebuild geometry according to enabled names
                    if (layersMgr.setEnabled) layersMgr.setEnabled(names);
                    const newGeom = layersMgr.buildGeometry(); newGeom.computeBoundingSphere();
                    if (lod) lod.setBase(newGeom);
                } catch {}
            } else if (ev?.type === 'openDoor') {
                try { const label = String(ev.payload?.label || ''); if (chat && label) chat.sendChat(`/open ${label}`); } catch {}
            } else if (ev?.type === 'diaryAdd') {
                try { const text = String(ev.payload?.text || ''); if (chat && text) chat.sendEvent({ kind: 'diary_entry', text, tz: '-03:00' }); } catch {}
            } else if (ev?.type === 'sleep') {
                try {
                    const mode = String(ev.payload?.mode || 'pause');
                    if (chat) chat.sendChat(mode === 'consolidate' ? '/sleep consolidate' : '/sleep');
                } catch {}
            } else if (ev?.type === 'wake') {
                try { if (chat) chat.sendChat('/resume'); } catch {}
            } else if (ev?.type === 'sendChat') {
                try { const text = String(ev.payload?.text || ''); if (chat && text) chat.sendChat(text); } catch {}
            }
        });
        // Update tablet with house info and dataset
        tablet.setStatus({ house: k3dUrl, nodes: k3dData.length, info: `dims=${loaded.info.dims} precision=${loaded.info.precision}` });
        tablet.setDataset(k3dData);

        // Try to load consolidated memory assets (House interior and Garden) into the same space
        try {
            const loader = new GLTFLoader();
            const tryAsset = async (url: string, pos: THREE.Vector3, scl: number, visible = true) => {
                try {
                    const r = await fetch(url, { method: 'HEAD', cache: 'no-store' });
                    if (!r.ok) return null;
                    const gltf = await new Promise<any>((resolve, reject) => loader.load(url, resolve, undefined, reject));
                    const obj = gltf.scene || gltf.scenes?.[0] || null;
                    if (obj) {
                        obj.position.copy(pos); obj.scale.setScalar(scl); obj.visible = visible;
                        scene.add(obj);
                    }
                    return obj;
                } catch { return null; }
            };
            // Place house near origin; garden to the side
            await tryAsset('/memory_house.glb', new THREE.Vector3(-6, -2, 0), 1.0, true);
            await tryAsset('/knowledge_garden.glb', new THREE.Vector3(6, -2, 0), 1.0, true);
        } catch {}

        // Start chat connection (prefer HUD elements if present)
        const chatLog = (document.getElementById('hud-chat-log') as HTMLDivElement) || (document.getElementById('chat-log') as HTMLDivElement);
        // Load persisted chat history from memory_house.glb (if present)
        try {
            const res = await fetch('/memory_house.glb', { cache: 'no-store' });
            if (res.ok) {
                const gltfJson = await res.json();
                const prim = gltfJson?.meshes?.[0]?.primitives?.[0];
                const k3d = prim?.extras?.k3d;
                const ids: string[] = Array.isArray(k3d?.ids) ? k3d.ids : [];
                const meta: any[] = Array.isArray(k3d?.metadata) ? k3d.metadata : [];
                // Collect chat_message items with timestamps
                type CM = { id: string; nick: string; text: string; ts: string };
                const items: CM[] = [];
                for (let i = 0; i < ids.length; i++) {
                    const m = meta[i] || {};
                    if (String(m.type||'') === 'chat_message') {
                        const id = String(ids[i]);
                        const nick = String(m.nick || '');
                        const text = String(m.text || '');
                        const ts = String(m.ts || '');
                        items.push({ id, nick, text, ts });
                    }
                }
                // Sort by ts (ISO)
                items.sort((a,b) => (a.ts < b.ts ? -1 : (a.ts > b.ts ? 1 : 0)));
                // Append last N messages to HUD log
                const maxHist = 200;
                const hist = items.slice(-maxHist);
                for (const m of hist) {
                    const row = document.createElement('div');
                    const when = m.ts ? new Date(m.ts.replace('Z','')).toLocaleString() : '';
                    row.textContent = when ? `[${when}] ${m.nick}: ${m.text}` : `${m.nick}: ${m.text}`;
                    row.style.opacity = '0.85';
                    chatLog.appendChild(row);
                }
                chatLog.scrollTop = chatLog.scrollHeight;
            }
        } catch {}
        const chatStatus = (document.getElementById('chat-status') as HTMLDivElement) || document.createElement('div');
        const topicBanner = document.getElementById('topic-banner') as HTMLDivElement | null;
        const hudTopicEl = document.getElementById('hud-topic') as HTMLDivElement | null;
        const append = (from: string, text: string) => {
            const el = document.createElement('div');
            el.textContent = `${from}: ${text}`;
            chatLog.appendChild(el);
            chatLog.scrollTop = chatLog.scrollHeight;
        };
        // Resolve WS endpoint: URL ?ws= takes precedence, then Vite env; otherwise try a candidate list
        const params = new URLSearchParams(window.location.search);
        const wsParam = params.get('ws');
        const envWs = (import.meta as any).env?.VITE_K3D_WS_URL as string | undefined;
        let wsCandidates: string[] = [];
        if (wsParam && wsParam.length > 0) {
            wsCandidates = [wsParam];
        } else if (envWs && envWs.length > 0) {
            wsCandidates = [envWs];
        } else {
            const base = 'ws://localhost:';
            wsCandidates = [base+'8765', base+'8787', base+'8788', base+'8789'];
        }
        let wsIndex = 0;
        const handlers: any = {
            onStatus: async (s: 'connected' | 'disconnected' | 'error') => {
                chatStatus.textContent = `WS: ${s}`;
                const q = await chat!.getQueueLength();
                const tinfo = document.getElementById('tablet-info') as HTMLDivElement;
                if (tinfo) tinfo.textContent = `Tablet: ${s === 'connected' ? 'online' : 'offline'}, queue=${q}`;
                if (tablet) tablet.setStatus({ ws: s, queue: q });
                // Auto-fallback to alternate default ports when no explicit ws param/env
                if (s === 'error') {
                    if (!wsParam && !envWs && chat && !chat.isConnected() && wsIndex + 1 < wsCandidates.length) {
                        wsIndex += 1;
                        const next = wsCandidates[wsIndex];
                        chat = new ChatClient(next, handlers);
                        chat.setContext({ house: k3dUrl, mode: 'ai' });
                        chat.connect();
                        const tip = document.getElementById('hud-tip') as HTMLDivElement | null;
                        if (tip) tip.textContent = `WS fallback → ${next}`;
                    }
                }
                // On connect, request topic and short history preload
                try { if (s === 'connected' && chat) { chat.sendChat('/topic show'); chat.sendChat('/history 50'); } } catch {}
            },
            onChat: (m: ChatMessage) => {
                if (m.action) append('* ' + m.from, m.text);
                else if (m.to) append(`${m.from}→${m.to}`, m.text);
                else if (m.channel) append(`[${m.channel}] ${m.from}`, m.text);
                else append(m.from, m.text);
                // Topic banner from system lines
                if ((topicBanner || hudTopicEl) && m.from === 'system' && m.text) {
                    const setMatch = /^Topic set for\s+(#[^:]+):\s+(.+)$/.exec(m.text);
                    const showMatch = /^Topic for\s+(#[^:]+):\s+(.+)$/.exec(m.text);
                    const mm = setMatch || showMatch;
                    if (mm) {
                        const txt = `${mm[1]} — ${mm[2]}`;
                        if (topicBanner) { topicBanner.textContent = txt; topicBanner.style.display='block'; }
                        if (hudTopicEl) { hudTopicEl.textContent = txt; }
                    }
                }
                // Mirror to tablet chat app
                try { if (tablet) tablet.dispatch({ type: 'chat_msg', payload: m }); } catch {}
                // mirror into tablet info box
                if (tablet) tablet.setStatus({ info: `last: ${m.from}: ${m.text.slice(0, 60)}` });
                // Parse goto resolution notes from agent messages to stats app
                try {
                    if (m.from === 'agent' && tablet && m.text) {
                        // Formats: "Navigating to <target>" or "[model x.xx] Navigating to <target> (from '<q>' sim=0.xxx)"
                        const txt = m.text;
                        const modelMatch = /^\[model\s+([0-9.]+)\]/.exec(txt);
                        const after = txt.replace(/^\[model[^\]]+\]\s*/, '');
                        const navMatch = /^Navigating to\s+([^()]+)(?:\s*\(from\s*'([^']+)'\s*(?:sim=([0-9.]+))?\))?/.exec(after);
                        if (navMatch) {
                            const target = navMatch[1].trim();
                            const query = (navMatch[2]||'').trim() || undefined;
                            const sim = navMatch[3] ? parseFloat(navMatch[3]) : undefined;
                            const model_confidence = modelMatch ? parseFloat(modelMatch[1]) : undefined;
                            tablet.dispatch({ type: 'goto_resolution', payload: { target, query, sim, model_confidence } });
                        }
                        // Parse consolidation summary: "Sleep: consolidated memory (reflections+X, training+Y, diary+Z)."
                        const cons = /^Sleep: consolidated memory \(reflections\+(\d+),\s*training\+(\d+),\s*diary\+(\d+)\)/.exec(txt);
                        if (cons) {
                            const reflections = parseInt(cons[1]||'0',10);
                            const training = parseInt(cons[2]||'0',10);
                            const diary = parseInt(cons[3]||'0',10);
                            tablet.dispatch({ type: 'consolidation_summary', payload: { reflections, training, diary } });
                        }
                    }
                } catch {}
            },
            onCommand: (m: CommandMessage) => {
                if (m.command === 'goto' && agent) {
                    agent.goToLabel(m.target);
                    append('system', `Agent navigating to ${m.target}`);
                } else if (m.command === 'open') {
                    try {
                        const info = JSON.parse(m.target);
                        const label = info.label ?? 'unknown';
                        const addr = info.address ?? 'k3d://';
                        const hops = Array.isArray(info.path) ? info.path.length - 1 : 0;
                        append('system', `Door opened to ${label} via ${hops} hops (${addr})`);
                        if (agent && label) agent.goToLabel(label);
                    } catch {
                        append('system', `Door opened: ${m.target}`);
                    }
                } else if (m.command === 'highlight') {
                    try {
                        const info = JSON.parse(m.target);
                        const labels: string[] = Array.isArray(info?.labels) ? info.labels : [];
                        if (labels.length && tablet) {
                            tablet.dispatch({ type: 'highlightNeighbors', payload: { labels } });
                            append('system', `Highlight: ${labels.join(', ')}`);
                        }
                    } catch {
                        if (tablet && m.target) {
                            const s = String(m.target);
                            const labels = s.split(',').map(x=>x.trim()).filter(Boolean);
                            tablet.dispatch({ type: 'highlightNeighbors', payload: { labels } });
                            append('system', `Highlight: ${labels.join(', ')}`);
                        }
                    }
                } else if (m.command === 'reasoning_path') {
                    try {
                        const payload = JSON.parse(m.target);
                        drawReasoningOverlay(payload);
                        append('system', `Spatial CoT overlay: ${payload?.mode || 'compose'}`);
                    } catch {}
                }
            }
        };
        chat = new ChatClient(wsCandidates[0], handlers);
        // Provide context for logging and continuity
        chat.setContext({ house: k3dUrl, mode: 'ai' });
        // Wire HUD History
        try { const hb = document.getElementById('chat-history') as HTMLButtonElement | null; if (hb) hb.onclick = ()=> chat?.sendChat('/history 200'); } catch {}
        chat.connect();

        // Share dataset graph with live server for routing (ids, neighbors, labels, positions)
        const ids = k3dData.map(r => r.id);
        const neighbors = k3dData.map(r => r.neighbors || []);
        const labelsArr = k3dData.map(r => (r.metadata?.label as string) || r.id);
        const positionsList = k3dData.map(r => r.vector as [number, number, number]);
        try { chat.sendEvent({ kind: 'dataset_graph', ids, neighbors, labels: labelsArr, positions: positionsList }); } catch {}
        // Send small snippet set for RAG (label + text), truncated to reduce payload
        try {
            const maxSnips = 1024;
            const pairs: [string, string][] = [];
            for (let i = 0; i < Math.min(k3dData.length, maxSnips); i++) {
                const lab = (k3dData[i].metadata?.label as string) || k3dData[i].id;
                const txt = (k3dData[i].metadata?.text as string) || '';
                if (txt) pairs.push([String(lab), String(txt).slice(0, 240)]);
            }
            if (pairs.length) chat.sendEvent({ kind: 'dataset_snippets', pairs });
        } catch {}
        await tabletStore.put(`graph:${k3dUrl}`, { ids, neighbors, labels: labelsArr });

        // Update HUD legend counts by metadata.type
        if (hudLegend) {
            const counts: Record<string, number> = {};
            for (const r of k3dData) {
                const t = String((r.metadata?.type as any) || 'unknown');
                counts[t] = (counts[t] || 0) + 1;
            }
            const total = k3dData.length;
            const lines = Object.entries(counts).sort((a,b)=>b[1]-a[1]).slice(0,8)
              .map(([k,v])=>`${k}: ${v} (${Math.round((v/Math.max(1,total))*100)}%)`);
            hudLegend.innerHTML = `<div style="font-weight:600;margin-bottom:4px;">Legend</div>${lines.map(s=>`<div>${s}</div>`).join('')}`;
            hudLegend.style.display = 'block';
        }

        // Door broadcasting is disabled for the House-as-rooms model (no inter-house linking here)

        // Try to load alias map and send to live server to enrich gazetteer
        try {
            const resAliases = await fetch('/aliases.json', { cache: 'no-store' });
            if (resAliases.ok) {
                const data = await resAliases.json();
                if (Array.isArray(data?.items)) {
                    chat.sendEvent({ kind: 'aliases', items: data.items });
                }
            }
        } catch {}
        // Fallback: fetch a small set of Wikipedia redirects live (limit to 24 labels)
        try {
            const labels = k3dData.slice(0, 24).map(r => (r.metadata?.label as string) || r.id);
            const items: { alias: string; label: string }[] = [];
            const fetchOne = async (lab: string) => {
                const api = `https://en.wikipedia.org/w/api.php?action=query&format=json&prop=redirects&rdlimit=25&titles=${encodeURIComponent(lab)}&origin=*`;
                const r = await fetch(api);
                if (!r.ok) return;
                const j = await r.json();
                const pages = j?.query?.pages || {};
                for (const k in pages) {
                    const pg = pages[k];
                    const reds = pg?.redirects || [];
                    for (const it of reds) {
                        const a = String(it?.title || '').trim();
                        if (a && a.toLowerCase() !== lab.toLowerCase()) items.push({ alias: a, label: lab });
                    }
                }
            };
            // limit concurrency to 6
            for (let i = 0; i < labels.length; i += 6) {
                await Promise.all(labels.slice(i, i + 6).map(l => fetchOne(l)));
            }
            if (items.length) chat.sendEvent({ kind: 'aliases', items });
        } catch {}

        // Prepare smart suggestions and layers
        sugg = new AISuggestionManager(scene, camera, renderer.domElement);
        sugg.setRecords(k3dData);
        layersMgr = new DynamicLayerManager();
        layersMgr.setRecords(k3dData as any);

    } catch (e) {
        console.error(`Failed to load house from ${k3dUrl}:`, e);
    }
}

async function loadAssetGLB(url: string) {
    clearScene();
    const loader = new GLTFLoader();
    loader.load(url, (gltf) => {
        // Check memory_realm for first primitive to decide material strategy
        const json: any = (gltf as any).parser?.json;
        let realm: string | null = null;
        try { realm = String(json?.meshes?.[0]?.primitives?.[0]?.extras?.k3d?.memory_realm || ''); } catch {}
        if (realm === 'house') {
            const honestyByMesh: number[] = [];
            if (json && Array.isArray(json.meshes)) {
                for (let i = 0; i < json.meshes.length; i++) {
                    let h = 1.0;
                    try {
                        const prim = json.meshes[i]?.primitives?.[0];
                        const hx = prim?.extras?.k3d?.rays?.[0]?.honesty;
                        if (typeof hx === 'number') h = hx;
                    } catch {}
                    honestyByMesh[i] = h;
                }
            }
            let meshIdx = 0;
            gltf.scene.traverse((obj: any) => {
                if (obj.isMesh) {
                    const h = honestyByMesh[meshIdx] ?? 1.0;
                    obj.material = new HonestyMaterial(h);
                    meshIdx++;
                }
            });
        } else {
            // Garden: keep default materials (lines/points)
        }
        scene.add(gltf.scene);
    });
}

async function loadGLB(url: string): Promise<any> {
    return new Promise((resolve, reject) => {
        const loader = new GLTFLoader();
        loader.load(url, (gltf) => resolve(gltf), undefined, reject);
    });
}

async function fetchJSON<T=any>(url: string): Promise<T | null> {
    try {
        const r = await fetch(url, { cache: 'no-store' });
        if (!r.ok) return null;
        return await r.json();
    } catch {
        return null;
    }
}

function findFloorNode(sceneObj: THREE.Object3D): THREE.Mesh | null {
    let found: THREE.Mesh | null = null;
    sceneObj.traverse((obj: any) => {
        if (found) return;
        const name = (obj.name || '').toLowerCase();
        if (obj.isMesh && (name.includes('greenhousefloor') || name.includes('greenhouse_floor') || name.includes('floor'))) {
            found = obj as THREE.Mesh;
        }
    });
    return found;
}

function getDomainColor(name: string): THREE.Color {
    const map: Record<string, number> = {
        physics: 0x3366ff,
        biology: 0x33cc66,
        mathematics: 0xdddddd,
        philosophy: 0xe6c229,
        art: 0xff66b3,
        engineering: 0x99ccff,
    };
    const key = (name || '').toLowerCase();
    const c = map[key] ?? 0x444444;
    return new THREE.Color(c);
}

function findSectorByAngle(angleDeg: number, sectors: Record<string, [number, number]>): string {
    const a = ((angleDeg % 360) + 360) % 360;
    for (const [name, rng] of Object.entries(sectors || {})) {
        const [s, e] = rng.map(Number) as [number, number];
        if (s <= e) {
            if (a >= s && a < e) return name;
        } else {
            // wrap-around sector
            if (a >= s || a < e) return name;
        }
    }
    return 'Unknown';
}

function applySectorColorsToFloor(sceneObj: THREE.Object3D, sectors: Record<string, [number, number]>) {
    const floor = findFloorNode(sceneObj);
    if (!floor) return;
    const geom = floor.geometry as THREE.BufferGeometry;
    const pos = geom.getAttribute('position') as THREE.BufferAttribute;
    if (!pos) return;
    const colors = new Float32Array(pos.count * 3);
    for (let i = 0; i < pos.count; i++) {
        const x = pos.getX(i);
        const z = pos.getZ(i);
        let ang = Math.atan2(z, x) * 180 / Math.PI;
        ang = (ang + 360) % 360;
        const sector = findSectorByAngle(ang, sectors);
        const col = getDomainColor(sector);
        colors[i*3+0] = col.r;
        colors[i*3+1] = col.g;
        colors[i*3+2] = col.b;
    }
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    (floor.material as any) = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95 });
}

async function loadGardenAndGreenhouse() {
    // Load greenhouse
    const gh = await loadGLB('/greenhouse_base.glb');
    scene.add(gh.scene);
    // Load garden (assembled)
    const garden = await loadGLB('/knowledge_garden/knowledge_garden.glb');
    scene.add(garden.scene);
    // Apply sector colors
    const sectors = await fetchJSON<Record<string, [number, number]>>('/knowledge_garden/knowledge_sectors.json');
    if (sectors) applySectorColorsToFloor(gh.scene, sectors);
}

async function loadMaterializedShapes(honestyMin: number = 0.7) {
    try {
        const r = await fetch('/house/materialized_objects/manifest.json', { cache: 'no-store' });
        if (!r.ok) return;
        const data = await r.json();
        const arr: Array<{path: string, honesty_score?: number}> = data?.shapes || [];
        if (!Array.isArray(arr) || arr.length === 0) return;
        const loader = new GLTFLoader();
        for (const it of arr) {
            const path = String(it.path || '');
            if (!path) continue;
            if (typeof it.honesty_score === 'number' && it.honesty_score < honestyMin) continue;
            try {
                const gltf = await loader.loadAsync(path);
                const root = gltf.scene;
                // Try to read extras.k3d via userData
                let k3d: any = null;
                root.traverse((node: any) => {
                    if (!k3d && node?.userData?.k3d) k3d = node.userData.k3d;
                });
                // scatter near origin of garden
                root.position.x += (Math.random() * 2 - 1) * 2.0;
                root.position.z += (Math.random() * 2 - 1) * 2.0;
                scene.add(root);
                console.log(`🌿 Garden shape: ${path} honesty=${k3d?.honesty_score ?? it.honesty_score ?? '?'}`);
            } catch (e) {
                console.warn('Failed to load shape', path, e);
            }
        }
    } catch (e) {
        console.warn('No materialized shapes manifest found');
    }
}

async function loadRayBundles() {
    try {
        const r = await fetch('/house/materialized_objects/manifest.json', { cache: 'no-store' });
        if (!r.ok) return;
        const data = await r.json();
        const rays: Array<{path: string, modality?: string, ray_count?: number, honesty_score?: number}> = data?.rays || [];
        if (!Array.isArray(rays) || rays.length === 0) return;
        for (const it of rays) {
            const path = String(it.path || '');
            if (!path) continue;
            try {
                if (path.endsWith('.json')) {
                    const resp = await fetch(path, { cache: 'no-store' });
                    if (!resp.ok) continue;
                    const rayData = await resp.json();
                    if (typeof rayData.honesty_score === 'number' && rayData.honesty_score < rayHonestyMin) continue;
                    let count = 0;
                    (rayData.rays || []).forEach((ray: any) => {
                        const o = new THREE.Vector3(...(ray.origin || [0,0,0]));
                        const d = new THREE.Vector3(...(ray.direction || [0,0,1])).normalize();
                        const end = o.clone().add(d.multiplyScalar(2.0));
                        const curve = new THREE.LineCurve3(o, end);
                        const radius = typeof ray.thickness === 'number' ? Math.max(0.002, Math.min(0.2, ray.thickness)) : 0.02;
                        const tube = new THREE.TubeGeometry(curve, 8, radius, 8, false);
                        const col = new THREE.Color(...(ray.color || [1,1,1]));
                        const mat = new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.85 });
                        const mesh = new THREE.Mesh(tube, mat);
                        mesh.visible = raysVisible;
                        mesh.userData = { type: 'ray', honesty_score: rayData.honesty_score ?? 1.0, modality: ray.modality || 'text', thickness: radius, source_shape: rayData.source_shape || '' };
                        scene.add(mesh);
                        rayObjects.push(mesh);
                        count++;
                    });
                    console.log(`✨ Rendered ${count} volumetric rays from ${path}`);
                } else {
                    const loader = new GLTFLoader();
                    const gltf = await loader.loadAsync(path);
                    // get honesty from k3d extras
                    let honesty = 1.0;
                    gltf.scene.traverse((node: any) => {
                        const ex = node?.userData?.k3d;
                        if (ex && typeof ex.honesty_score === 'number') honesty = ex.honesty_score;
                    });
                    if (honesty < rayHonestyMin) continue;
                    // Convert lines to tubes using positions and optional thickness array
                    const tubes: THREE.Object3D[] = [];
                    gltf.scene.traverse((obj: any) => {
                        if (obj.isLineSegments) {
                            const pos = (obj.geometry as THREE.BufferGeometry).getAttribute('position') as THREE.BufferAttribute;
                            const colors = (obj.geometry as THREE.BufferGeometry).getAttribute('color') as THREE.BufferAttribute | undefined;
                            const ex = obj?.userData?.k3d || {};
                            const rayThickness: number[] = (ex.ray_thickness || []) as number[];
                            for (let i = 0; i + 1 < pos.count; i += 2) {
                                const o = new THREE.Vector3(pos.getX(i), pos.getY(i), pos.getZ(i));
                                const e = new THREE.Vector3(pos.getX(i+1), pos.getY(i+1), pos.getZ(i+1));
                                const curve = new THREE.LineCurve3(o, e);
                                const radius = Math.max(0.002, Math.min(0.2, rayThickness[Math.floor(i/2)] || 0.02));
                                const tube = new THREE.TubeGeometry(curve, 8, radius, 8, false);
                                let col = new THREE.Color(1,1,1);
                                if (colors) col = new THREE.Color(colors.getX(i), colors.getY(i), colors.getZ(i));
                                const mat = new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.85 });
                                const mesh = new THREE.Mesh(tube, mat);
                                mesh.visible = raysVisible;
                                mesh.userData = { type: 'ray', honesty_score: honesty, modality: ex.modality || 'text', thickness: radius, source_shape: ex.source_shape || '' };
                                scene.add(mesh);
                                rayObjects.push(mesh);
                                tubes.push(mesh);
                            }
                        }
                    });
                    console.log(`✨ Rendered ray bundle ${path} as ${tubes.length} tubes (GLB)`);
                }
            } catch (e) { console.warn('Failed to load rays', path, e); }
        }
    } catch (e) {
        /* ignore */
    }
}

/**
 * Initializes the expert selector dropdown by fetching the condo configuration.
 */
async function initCondoSelector() {
    try {
        // Fetch condo.json from the public directory, served at the root
        const condoUrl = '/condo.json';
        condoConfig = await fetchCondoConfig(condoUrl);

        expertSelect.innerHTML = ''; // Clear "Loading..."
        condoConfig.houses.forEach(houseInfo => {
            const option = document.createElement('option');
            option.value = houseInfo.expert;
            option.textContent = houseInfo.expert;
            expertSelect.appendChild(option);
        });

        expertSelect.addEventListener('change', () => {
            const selectedExpert = expertSelect.value;
            const houseInfo = condoConfig?.houses.find(h => h.expert === selectedExpert);
            if (houseInfo) {
                // The URI in condo.json is now a direct path, e.g., "/math_house.k3d"
                const houseUrl = houseInfo.uri;
                loadHouse(houseUrl);
            }
        });

        // Load the first house by default
        if (condoConfig.houses.length > 0) {
            expertSelect.value = condoConfig.houses[0].expert;
            expertSelect.dispatchEvent(new Event('change'));
        }

    } catch (e) {
        console.error('Failed to initialize condo selector:', e);
        expertSelect.innerHTML = '<option value="">Failed to load</option>';
    }
}

// --- Interactivity ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
let rayHonestyMin = 0.7;
const rayObjects: THREE.Object3D[] = [];
let raysVisible = true;

function onMouseMove(event: MouseEvent) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
}

function checkIntersects() {
    raycaster.setFromCamera(mouse, camera);
    if (loadedHouseScene) {
        const houseHits = raycaster.intersectObjects([loadedHouseScene.root], true);
        const houseNode = houseHits
            .map((hit) => findHouseNodeForObject(hit.object))
            .find((node): node is HouseNode => node !== null);
        if (houseNode) {
            showHouseTooltip(houseNode);
            return;
        }
        tooltip.style.display = 'none';
    }

    if (!currentPoints) {
        // Also check ray meshes
        const rayHits = rayObjects.length ? raycaster.intersectObjects(rayObjects, true) : [];
        if (rayHits.length > 0) {
            const obj: any = rayHits[0].object;
            const ud = obj?.userData || {};
            let rtip = document.getElementById('ray-tooltip') as HTMLDivElement | null;
            if (!rtip) {
                rtip = document.createElement('div');
                rtip.id = 'ray-tooltip';
                rtip.style.position = 'absolute';
                rtip.style.background = 'rgba(0,0,0,0.8)';
                rtip.style.color = '#fff';
                rtip.style.padding = '4px 6px';
                rtip.style.borderRadius = '4px';
                rtip.style.fontSize = '12px';
                rtip.style.pointerEvents = 'none';
                document.body.appendChild(rtip);
            }
            rtip.style.display = 'block';
            rtip.style.left = `${mouse.x * window.innerWidth / 2 + window.innerWidth / 2 + 5}px`;
            rtip.style.top = `${-mouse.y * window.innerHeight / 2 + window.innerHeight / 2 + 5}px`;
            rtip.innerHTML = `<strong>Ray</strong><br/>modality=${ud.modality || 'n/a'} honesty=${(ud.honesty_score ?? 1.0).toFixed(2)} thickness=${(ud.thickness ?? 0.02).toFixed(3)}`;
            return;
        } else {
            const rtip = document.getElementById('ray-tooltip') as HTMLDivElement | null;
            if (rtip) rtip.style.display = 'none';
        }
        return;
    }

    raycaster.setFromCamera(mouse, camera);
    const intersects = currentPoints ? raycaster.intersectObject(currentPoints as any, true) : [];

    if (intersects.length > 0 && intersects[0].index !== undefined) {
        const idx = intersects[0].index;
        const record = k3dData[idx];

        if (record) {
            tooltip.style.display = 'block';
            tooltip.style.left = `${mouse.x * window.innerWidth / 2 + window.innerWidth / 2 + 5}px`;
            tooltip.style.top = `${-mouse.y * window.innerHeight / 2 + window.innerHeight / 2 + 5}px`;
            const label = (record.metadata?.label as string) || record.id;
            lastHoverRecord = record;
            const text = (record.metadata?.text as string) || '';
            const img = (record.metadata?.image as string) || '';
            let html = `<div><strong>${label}</strong></div>`;
            if (img) {
                const safeSrc = String(img).replace(/"/g, '');
                html += `<div style="margin-top:4px;"><img src="${safeSrc}" alt="thumb" style="max-width:240px; max-height:140px; object-fit:cover; border:1px solid #ccc;"/></div>`;
            }
            if (text) {
                const t = text.length > 200 ? (text.slice(0, 197) + '...') : text;
                html += `<div style=\"margin-top:4px; font-size:12px; color:#333;\">${t}</div>`;
            }
            tooltip.innerHTML = html;
            if (tablet) tablet.setFocusLabel(label);
            try { const t = document.getElementById('hud-topic') as HTMLDivElement | null; if (t) t.textContent = label; } catch {}
        } else {
            tooltip.style.display = 'none';
        }
    } else {
        tooltip.style.display = 'none';
        // Also test rays
        const rayHits = rayObjects.length ? raycaster.intersectObjects(rayObjects, true) : [];
        if (rayHits.length > 0) {
            const obj: any = rayHits[0].object;
            const ud = obj?.userData || {};
            let rtip = document.getElementById('ray-tooltip') as HTMLDivElement | null;
            if (!rtip) {
                rtip = document.createElement('div');
                rtip.id = 'ray-tooltip';
                rtip.style.position = 'absolute';
                rtip.style.background = 'rgba(0,0,0,0.8)';
                rtip.style.color = '#fff';
                rtip.style.padding = '4px 6px';
                rtip.style.borderRadius = '4px';
                rtip.style.fontSize = '12px';
                rtip.style.pointerEvents = 'none';
                document.body.appendChild(rtip);
            }
            rtip.style.display = 'block';
            rtip.style.left = `${mouse.x * window.innerWidth / 2 + window.innerWidth / 2 + 5}px`;
            rtip.style.top = `${-mouse.y * window.innerHeight / 2 + window.innerHeight / 2 + 5}px`;
            rtip.innerHTML = `<strong>Ray</strong><br/>modality=${ud.modality || 'n/a'} honesty=${(ud.honesty_score ?? 1.0).toFixed(2)} thickness=${(ud.thickness ?? 0.02).toFixed(3)}`;
        } else {
            const rtip = document.getElementById('ray-tooltip') as HTMLDivElement | null;
            if (rtip) rtip.style.display = 'none';
        }
    }
}

window.addEventListener('mousemove', onMouseMove);
canvas.addEventListener('click', onSceneClick);
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// Update LOD HUD periodically based on camera proximity and trigger automatic LOD switching
setInterval(() => {
    if (!k3dData || k3dData.length === 0) return;
    // Estimate nearest record to camera from a capped sample
    const sampleN = Math.min(k3dData.length, 4096);
    let best = Infinity;
    for (let i = 0; i < sampleN; i++) {
        const v = k3dData[i].vector;
        const dx = v[0] - camera.position.x;
        const dy = v[1] - camera.position.y;
        const dz = v[2] - camera.position.z;
        const d = Math.sqrt(dx*dx + dy*dy + dz*dz);
        if (d < best) best = d;
    }
    // Handle automatic LOD switching
    switchLodIfNeeded(best);
    if (lodHud.style.display !== 'none') {
        const near = 3.0, far = 30.0, minCap = 3, maxCap = 12;
        const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
        const smooth01 = (x: number) => { x = clamp01(x); return x*x*(3-2*x); };
        let t = 1.0 - ((best - near) / Math.max(1e-6, (far - near)));
        t = smooth01(t);
        const cap = Math.round(minCap + (maxCap - minCap) * t);
        const lodMethod = getLodMethod(best);
        lodHud.textContent = `LOD ${lodMethod.toUpperCase()} dist≈${best.toFixed(1)} cap≈${cap}`;
    }
}, 500);
window.addEventListener('keydown', (ev: KeyboardEvent) => {
    if (ev.key.toLowerCase() === 'f' && tablet) {
        tablet.toggleFocus();
    }
    if (ev.key.toLowerCase() === 's') {
        if (sugg && lastHoverRecord && agent) {
            sugg.showSuggestions(lastHoverRecord, (targetId) => { agent!.goToLabel(targetId); });
        }
    }
    if (ev.key.toLowerCase() === 'l') {
        toggleLayersOverlay();
    }
    if (ev.key.toLowerCase() === 'r') {
        // reload rays with current filter: remove existing and load again
        while (rayObjects.length) {
            const obj = rayObjects.pop()!;
            scene.remove(obj);
        }
        loadRayBundles();
    }
});

// Agent UI controls
const agentTarget = document.getElementById('agent-target') as HTMLInputElement;
const agentGo = document.getElementById('agent-go') as HTMLButtonElement;
const agentFollow = document.getElementById('agent-follow') as HTMLInputElement;
const chatInput = document.getElementById('chat-input') as HTMLInputElement;
const chatSend = document.getElementById('chat-send') as HTMLButtonElement;
const chatPause = document.getElementById('chat-pause') as HTMLButtonElement;
const chatResume = document.getElementById('chat-resume') as HTMLButtonElement;
const chatStatusBtn = document.getElementById('chat-status-btn') as HTMLButtonElement;
const chatAskThoughts = document.getElementById('chat-ask-thoughts') as HTMLButtonElement;
const chatWhoAmI = document.getElementById('chat-whoami') as HTMLButtonElement;
const cacheToggle = document.getElementById('cache-enable') as HTMLInputElement;
const cacheClear = document.getElementById('cache-clear') as HTMLButtonElement;
const colorMode = document.getElementById('color-mode') as HTMLSelectElement;
const clusterK = document.getElementById('cluster-k') as HTMLInputElement;
const applyColor = document.getElementById('apply-color') as HTMLButtonElement;
const legend = document.getElementById('legend') as HTMLDivElement;
const toggleTrails = document.getElementById('toggle-trails') as HTMLInputElement;
const tabletFocusBtn = document.getElementById('tablet-focus') as HTMLButtonElement;
const tabletModeSel = document.getElementById('tablet-mode') as HTMLSelectElement;
const tabletVisible = document.getElementById('tablet-visible') as HTMLInputElement;
const lodReductionSel = document.getElementById('lod-reduction') as HTMLSelectElement;
const updateLodBtn = document.getElementById('update-lod') as HTMLButtonElement;
let currentHouseUrl: string = '';
if (agentGo) {
    agentGo.addEventListener('click', () => {
        if (agent && agentTarget?.value) {
            agent.goToLabel(agentTarget.value);
            if (chat) chat.sendCommandGoto(agentTarget.value);
        }
    });
}
if (chatSend) {
    chatSend.addEventListener('click', () => {
        if (chat && chatInput?.value) {
            chat.sendChat(chatInput.value);
            chatInput.value = '';
        }
    });
}
if (chatPause) {
    chatPause.addEventListener('click', () => {
        if (chat) chat.sendChat('/pause analyze');
    });
}
if (chatResume) {
    chatResume.addEventListener('click', () => {
        if (chat) chat.sendChat('/resume');
    });
}
if (chatStatusBtn) {
    chatStatusBtn.addEventListener('click', () => {
        if (chat) chat.sendChat('/status');
    });
}

// Simple honesty slider for rays
try {
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '0';
    slider.max = '1';
    slider.step = '0.05';
    slider.value = String(rayHonestyMin);
    slider.style.position = 'fixed';
    slider.style.right = '16px';
    slider.style.bottom = '16px';
    slider.style.zIndex = '1000';
    slider.title = 'Ray Honesty Filter';
    slider.addEventListener('input', () => {
        rayHonestyMin = parseFloat(slider.value);
    });
    slider.addEventListener('change', () => {
        // reload rays on change
        while (rayObjects.length) {
            const obj = rayObjects.pop()!;
            scene.remove(obj);
        }
        loadRayBundles();
    });
    document.body.appendChild(slider);
} catch {}

// Rays toggle button
try {
    const btn = document.createElement('button');
    btn.textContent = '👁️ Hide Rays';
    btn.style.position = 'fixed';
    btn.style.right = '16px';
    btn.style.bottom = '48px';
    btn.style.zIndex = '1000';
    btn.addEventListener('click', () => {
        raysVisible = !raysVisible;
        btn.textContent = raysVisible ? '👁️ Hide Rays' : '👁️ Show Rays';
        rayObjects.forEach(o => { o.visible = raysVisible; });
    });
    document.body.appendChild(btn);
} catch {}
if (chatAskThoughts) {
    chatAskThoughts.addEventListener('click', () => {
        if (chat) chat.sendChat('/ask-thoughts');
    });
}
if (chatWhoAmI) {
    chatWhoAmI.addEventListener('click', () => {
        if (chat) chat.sendChat('/whoami');
    });
}
if (cacheToggle) {
    cacheToggle.addEventListener('change', () => {
        cacheEnabled = !!cacheToggle.checked;
    });
}
if (cacheClear) {
    cacheClear.addEventListener('click', async () => {
        await cache.clear();
        const infoEl = document.getElementById('dataset-info') as HTMLDivElement;
        if (infoEl) infoEl.textContent = 'cache cleared';
    });
}

function applyPositionColors() {
    if (!currentPoints || k3dData.length === 0) return;
    const positions = k3dData.map(d => d.vector);
    const min = new THREE.Vector3(Infinity, Infinity, Infinity);
    const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
    positions.forEach(v => { min.min(new THREE.Vector3().fromArray(v)); max.max(new THREE.Vector3().fromArray(v)); });
    const size = new THREE.Vector3().subVectors(max, min);
    const updateGeomColors = (geom: THREE.BufferGeometry) => {
        const colors = geom.getAttribute('color') as THREE.BufferAttribute;
        if (!colors) return;
        const pos = geom.getAttribute('position') as THREE.BufferAttribute;
        for (let i = 0; i < colors.count; i++) {
            const x = pos.getX(i);
            const y = pos.getY(i);
            const z = pos.getZ(i);
            const r = size.x > 0 ? (x - min.x) / size.x : 0.5;
            const g = size.y > 0 ? (y - min.y) / size.y : 0.5;
            const b = size.z > 0 ? (z - min.z) / size.z : 0.5;
            colors.setXYZ(i, r, g, b);
        }
        colors.needsUpdate = true;
    };
    if ((currentPoints as any).isPoints) updateGeomColors((currentPoints as THREE.Points).geometry as THREE.BufferGeometry);
    else if ((currentPoints as any).isGroup) {
        (currentPoints as THREE.Group).children.forEach(ch => { if ((ch as any).isPoints) updateGeomColors((ch as THREE.Points).geometry as THREE.BufferGeometry); });
    }
    legend.textContent = 'position-based coloring';
}

function applyClusterColors(k: number) {
    if (!currentPoints || k3dData.length === 0) return;
    const emb = k3dData.map(d => d.embedding);
    const { labels } = kmeans(emb, Math.max(2, Math.min(20, Math.floor(k))));
    const pal = palette(Math.max(...labels) + 1);
    const counts: Record<number, number> = {};
    const applyColors = (geom: THREE.BufferGeometry) => {
        const colors = geom.getAttribute('color') as THREE.BufferAttribute;
        if (!colors) return;
        for (let i = 0; i < colors.count; i++) {
            const cidx = i % labels.length;
            const c = labels[cidx] ?? 0; counts[c] = (counts[c] ?? 0) + 1;
            const val = new THREE.Color(pal[c % pal.length]);
            colors.setXYZ(i, val.r, val.g, val.b);
        }
        colors.needsUpdate = true;
    };
    if ((currentPoints as any).isPoints) applyColors((currentPoints as THREE.Points).geometry as THREE.BufferGeometry);
    else if ((currentPoints as any).isGroup) (currentPoints as THREE.Group).children.forEach(ch => { if ((ch as any).isPoints) applyColors((ch as THREE.Points).geometry as THREE.BufferGeometry); });
    const items = Object.keys(counts).map(k => ({ c: Number(k), n: counts[Number(k)] })).sort((a,b)=>a.c-b.c);
    legend.innerHTML = items.map(it => `<span style="display:inline-block;width:12px;height:12px;background:${pal[it.c%pal.length]};margin-right:4px;"></span> C${it.c} (${it.n})`).join(' &nbsp; ');
}

if (applyColor) {
    applyColor.addEventListener('click', () => {
        if (colorMode.value === 'cluster') {
            applyClusterColors(Number(clusterK.value || '5'));
        } else {
            applyPositionColors();
        }
    });
}
if (toggleTrails) {
    toggleTrails.addEventListener('change', () => {
        if (agent) agent.trailsEnabled = !!toggleTrails.checked;
    });
}
const cotToggle = document.getElementById('toggle-cot') as HTMLInputElement | null;
if (cotToggle) {
    cotToggle.addEventListener('change', () => {
        if (cotOverlay) cotOverlay.visible = !!cotToggle.checked;
    });
}
if (tabletFocusBtn) {
    tabletFocusBtn.addEventListener('click', () => {
        if (tablet) tablet.toggleFocus();
    });
}
if (tabletModeSel) {
    tabletModeSel.addEventListener('change', () => {
        if (tablet) tablet.setStatus({ mode: (tabletModeSel.value as any) });
    });
}
if (tabletVisible) {
    tabletVisible.addEventListener('change', () => {
        if (tablet) tablet.object.visible = !!tabletVisible.checked;
    });
}
if (agentFollow) {
    agentFollow.addEventListener('change', () => {
        if (agent) agent.followCamera = !!agentFollow.checked;
    });
}
if (updateLodBtn) {
    updateLodBtn.addEventListener('click', () => {
        if (!lodReductionSel || !currentHouseUrl) return;
        const reduction = lodReductionSel.value;
        if (!['umap', 'pca', 'tsne'].includes(reduction)) return;
        let newUrl = currentHouseUrl.replace(/\.(umap|pca|tsne)?\.glb$/, `.${reduction}.glb`);
        loadHouse(newUrl);
    });
}

// HUD: History and Compose
(() => {
  const h = document.getElementById('hud-history') as HTMLButtonElement | null;
  if (h) h.addEventListener('click', () => { try { chat?.sendChat('/history 200'); } catch {} });
  const c = document.getElementById('hud-compose') as HTMLButtonElement | null;
  if (c) c.addEventListener('click', () => { try { if (tablet) { tablet.showFocus(); tablet.dispatch({ type: 'open_app', payload: { id: 'chat', tab: 'compose' } }); } } catch {} });
})();

let last = performance.now();
let currentLodLevel: string | null = null;

// Determine appropriate LOD method based on camera distance
function getLodMethod(distance: number): string {
    if (currentPoints && (currentPoints.children?.length || k3dData.length) > 5000 && distance > 20) {
        // For large datasets, use fast method at distance
        return distance > 40 ? 'pca' : 'umap';
    }
    // Standard LOD levels based on distance
    if (distance > 30) return 'pca';  // Distant: fastest
    if (distance > 10) return 'umap'; // Medium: balanced
    return 'tsne'; // Close: most accurate
}

// Handle automatic LOD switching
let lodSwitchTimeout: any = null;
function switchLodIfNeeded(distance: number) {
    if (!currentHouseUrl) return;
    const newLod = getLodMethod(distance);
    if (newLod !== currentLodLevel) {
        currentLodLevel = newLod;
        if (lodSwitchTimeout) clearTimeout(lodSwitchTimeout);
        lodSwitchTimeout = setTimeout(() => {
            let newUrl = currentHouseUrl.replace(/\.(umap|pca|tsne)?\.glb$/, `.${newLod}.glb`);
            if (newUrl !== currentHouseUrl) {
                loadHouse(newUrl);
            }
        }, 500); // Debounce switching
    }
}

function animate() {
    const now = performance.now();
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    requestAnimationFrame(animate);
    if (roomCamera) roomCamera.update(dt);
    controls.update();
    checkIntersects();
    if (agent) agent.update(dt);
    if (lod) lod.update(camera);
    renderer.render(scene, camera);
}

// --- Layers Overlay ---
function toggleLayersOverlay() {
    if (layersOverlay) { document.body.removeChild(layersOverlay); layersOverlay = null; return; }
    if (!layersMgr) return;
    const div = document.createElement('div');
    div.style.position = 'fixed'; div.style.right = '12px'; div.style.top = '12px';
    div.style.background = 'rgba(0,0,0,0.8)'; div.style.color = '#fff';
    div.style.padding = '10px'; div.style.zIndex = '1500'; div.style.fontFamily = 'system-ui, sans-serif';
    const title = document.createElement('div'); title.textContent = 'Layers'; title.style.fontWeight = 'bold'; div.appendChild(title);
    const layers = layersMgr.getLayers();
    if (!layers.length) {
        const p = document.createElement('div'); p.textContent = '(no layers found)'; div.appendChild(p);
    } else {
        for (const name of layers) {
            const row = document.createElement('label'); row.style.display = 'block'; row.style.marginTop = '4px';
            const cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = layersMgr.isEnabled(name);
            cb.onchange = () => {
                layersMgr!.toggle(name, cb.checked);
                const newGeom = layersMgr!.buildGeometry(); newGeom.computeBoundingSphere();
                if (lod) lod.setBase(newGeom);
            };
            row.appendChild(cb); row.appendChild(document.createTextNode(' ' + name));
            div.appendChild(row);
        }
    }
    const close = document.createElement('button'); close.textContent = 'Close'; close.style.marginTop = '8px'; close.onclick = () => { toggleLayersOverlay(); };
    div.appendChild(close);
    document.body.appendChild(div); layersOverlay = div;
}

// --- Mic Toggle ---
let micStream: MediaStream | null = null;
let micAnalyzer: AnalyserNode | null = null;
let micData: Uint8Array | null = null;
const rpn = new RPN();
const micToggleBtn = document.getElementById('mic-toggle') as HTMLButtonElement;
const micLevelBar = document.getElementById('mic-level') as HTMLDivElement;
const micStatus = document.getElementById('mic-status') as HTMLSpanElement;

async function startMic() {
    try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true }, video: false });
        const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
        const src = ctx.createMediaStreamSource(micStream);
        micAnalyzer = ctx.createAnalyser();
        micAnalyzer.fftSize = 256;
        src.connect(micAnalyzer);
        micData = new Uint8Array(micAnalyzer.frequencyBinCount);
        micStatus.textContent = 'mic: on';
        micToggleBtn.textContent = 'Stop Mic';
    } catch (e) {
        micStatus.textContent = 'mic: error';
    }
}

function stopMic() {
    if (micStream) {
        micStream.getTracks().forEach(t => t.stop());
        micStream = null;
    }
    micAnalyzer = null;
    micData = null;
    micStatus.textContent = 'mic: off';
    micToggleBtn.textContent = 'Start Mic';
    if (micLevelBar) micLevelBar.style.width = '0%';
}

if (micToggleBtn) {
    micToggleBtn.addEventListener('click', () => {
        if (micStream) stopMic(); else startMic();
    });
}

// update mic level in the render loop
const _origRender = renderer.render.bind(renderer);
renderer.render = (s, c) => {
    if (micAnalyzer && micData && micLevelBar) {
        micAnalyzer.getByteTimeDomainData(micData as any);
        let peak = 0;
        for (let i = 0; i < micData.length; i++) {
            // v = |x - 128| / 128 using RPN
            const v = rpn.eval([micData[i], 128, '-', 'abs', 128, '/',]);
            if (v > peak) peak = v;
        }
        const pct = Math.min(100, Math.max(0, Math.round(peak * 100)));
        micLevelBar.style.width = pct + '%';
    }
    _origRender(s, c);
};

// --- Stars backdrop to suggest depth even before loading a House ---
function createStars(count = 600) {
    const geom = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        const r = 80 + Math.random() * 120;
        const phi = Math.random() * Math.PI * 2;
        const costh = Math.random() * 2 - 1; const sinth = Math.sqrt(1 - costh * costh);
        positions[i*3+0] = r * Math.cos(phi) * sinth;
        positions[i*3+1] = r * Math.sin(phi) * sinth;
        positions[i*3+2] = r * costh;
    }
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({ color: 0x88aaff, size: 0.5, sizeAttenuation: true });
    const stars = new THREE.Points(geom, mat);
    scene.add(stars);
}

// --- Spatial CoT Overlay ---
function drawReasoningOverlay(payload: any) {
    // Remove existing overlay
    if (cotOverlay) {
        try {
            cotOverlay.traverse((obj: any) => {
                try { obj.geometry?.dispose?.(); } catch {}
                try { obj.material?.dispose?.(); } catch {}
            });
        } catch {}
        scene.remove(cotOverlay);
        cotOverlay = null;
    }
    const group = new THREE.Group();
    const steps: any[] = Array.isArray(payload?.steps) ? payload.steps : [];
    const wp: Array<[number, number, number]> = Array.isArray(payload?.waypoints) ? payload.waypoints : [];
    let points: THREE.Vector3[] = [];
    if (wp.length > 0) {
        points = wp.map(([x,y,z]) => new THREE.Vector3(x,y,z));
    } else {
        // Map labels -> vectors using recordMap
        const labels: string[] = [];
        for (const s of steps) {
            const op = String(s?.op || '');
            let lab = String(s?.label || '');
            if (!lab || op === 'compare') continue;
            if (op === 'synthesize' && lab.includes(',')) lab = lab.split(',')[0].trim();
            if (!labels.includes(lab)) labels.push(lab);
        }
        for (const lab of labels) {
            const rec = findRecordByLabel(lab);
            if (rec) points.push(new THREE.Vector3(rec.vector[0], rec.vector[1], rec.vector[2]));
        }
    }
    if (points.length >= 2) {
        // Draw polyline for path
        const geom = new THREE.BufferGeometry().setFromPoints(points);
        const mat = new THREE.LineBasicMaterial({ color: 0x66ccff, transparent: true, opacity: 0.8 });
        const line = new THREE.Line(geom, mat);
        (line.material as THREE.LineBasicMaterial).depthWrite = false;
        group.add(line);
    }
    // Draw step markers
    const sphere = new THREE.SphereGeometry(0.12, 12, 12);
    const colorFor = (op: string, verified?: boolean): number => {
        if (op === 'retrieve') return 0x00e0ff;  // cyan
        if (op === 'compare') return 0xffd166;   // yellow
        if (op === 'synthesize') return 0xff66cc; // pink
        if (op === 'verify') return verified === false ? 0xff6b6b : 0x8bdc7f; // red/green
        return 0xffffff;
    };
    for (const s of steps) {
        const op = String(s?.op || '');
        let lab = String(s?.label || '');
        if (!lab || op === 'compare') continue;
        if (op === 'synthesize' && lab.includes(',')) lab = lab.split(',')[0].trim();
        const rec = findRecordByLabel(lab);
        if (!rec) continue;
        const pos = new THREE.Vector3(rec.vector[0], rec.vector[1], rec.vector[2]);
        const mat = new THREE.MeshBasicMaterial({ color: colorFor(op, !!s?.verified) });
        const m = new THREE.Mesh(sphere, mat);
        m.position.copy(pos);
        // Scale by confidence if present (visualize step certainty)
        const conf = typeof s?.confidence === 'number' ? Math.max(0, Math.min(1, s.confidence)) : 0.6;
        const scale = 0.8 + 0.6 * conf;
        m.scale.setScalar(scale);
        group.add(m);
    }
    cotOverlay = group;
    scene.add(group);
}

function findRecordByLabel(label: string): K3DRecord | null {
    // Prefer exact match on metadata.label; then contains; then id
    let rec = k3dData.find(r => (r.metadata?.label as string) === label);
    if (rec) return rec;
    rec = k3dData.find(r => ((r.metadata?.label as string) || '').toLowerCase() === label.toLowerCase());
    if (rec) return rec;
    rec = k3dData.find(r => ((r.metadata?.label as string) || '').toLowerCase().includes(label.toLowerCase()));
    if (rec) return rec;
    rec = k3dData.find(r => r.id === label);
    return rec || null;
}

// --- Game HUD chat behavior ---
function initHudChat() {
    const input = document.getElementById('hud-chat-input') as HTMLInputElement | null;
    if (!input) return;
    const tip = document.getElementById('hud-tip') as HTMLDivElement | null;
    const openInput = () => { input.style.display = 'block'; input.focus(); if (tip) tip.style.display = 'none'; };
    const closeInput = () => { input.style.display = 'none'; (document.activeElement as HTMLElement)?.blur?.(); };
    // Idle sleep consolidation: trigger '/sleep consolidate' after prolonged inactivity
    let lastAction = performance.now();
    const idleThresholdMs = 90000; // 90s idle
    const markActivity = () => { lastAction = performance.now(); };
    ['mousemove','keydown','pointerdown','wheel','touchstart'].forEach(evt => window.addEventListener(evt, markActivity, { passive: true }));
    setInterval(() => {
        const idle = performance.now() - lastAction;
        if (idle > idleThresholdMs && chat && chat.isConnected()) {
            chat.sendChat('/sleep consolidate');
            lastAction = performance.now();
        }
    }, 5000);
    window.addEventListener('keydown', (ev: KeyboardEvent) => {
        if (ev.key === 'Enter' && input.style.display === 'none') { ev.preventDefault(); openInput(); return; }
        if (ev.key === '/' && input.style.display === 'none') { ev.preventDefault(); openInput(); input.value = '/'; return; }
        if (ev.key === 'Escape' && input.style.display !== 'none') { ev.preventDefault(); closeInput(); return; }
        if (ev.key === 'Enter' && input.style.display !== 'none') {
            ev.preventDefault();
            const txt = input.value.trim();
            if (txt && chat) { chat.sendChat(txt); }
            input.value = '';
        }
    });
}

// --- Auto mode vs Dev mode ---
function startApp() {
    const params = new URLSearchParams(window.location.search);
    const dev = params.get('dev') === '1';
    const panel = document.getElementById('ui-container') as HTMLDivElement | null;
    if (panel) panel.style.display = dev ? 'block' : 'none';
    startBootPolling();
    setBootLocal(0.08, 'Igniting stars backdrop…', 'stage: stars');
    createStars();
    setBootLocal(0.14, 'Preparing HUD and tablet surfaces…', 'stage: interface');
    initHudChat();
    if (dev) {
        setBootLocal(0.32, 'Developer controls ready.', 'stage: developer-panel');
        initCondoSelector();
        setBootLocal(1.0, 'K3D ready.', 'stage: ready');
        finishBootOverlay('K3D ready.');
    } else {
        const assetCombo = params.get('asset');
        if (assetCombo && assetCombo.toLowerCase() === 'garden+greenhouse') {
            setBootLocal(0.28, 'Loading greenhouse and garden assets…', 'stage: asset-preload');
            (async () => {
                try {
                    await loadGardenAndGreenhouse();
                    setBootLocal(0.62, 'Materializing procedural shapes…', 'stage: scene-compose');
                    await loadMaterializedShapes(0.7);
                    setBootLocal(0.82, 'Loading ray bundles…', 'stage: scene-rays');
                    await loadRayBundles();
                } catch (e) {
                    console.error(e);
                } finally {
                    setBootLocal(1.0, 'Entering K3D world…', 'stage: ready');
                    animate();
                    finishBootOverlay('K3D ready.');
                }
            })();
            return;
        }
        const asset = params.get('asset');
        if (asset && asset.length > 0) {
            setBootLocal(0.28, `Loading asset ${asset}…`, 'stage: asset-preload');
            (async () => {
                try {
                    const assetUrl = asset.toLowerCase() === 'house' ? '/house.glb' : asset;
                    if (isHouseGlbUrl(assetUrl)) await loadHouseSceneAsset(assetUrl);
                    else await loadAssetGLB(assetUrl);
                } catch {}
                finally {
                    setBootLocal(1.0, 'Entering K3D world…', 'stage: ready');
                    animate();
                    finishBootOverlay('K3D ready.');
                }
            })();
            return;
        }
        // Optional override via URL: ?gltf=/galaxy.cross.glb
        const pick = params.get('gltf') || params.get('house');
        if (pick && pick.length > 0) {
            setBootLocal(0.28, `Checking ${pick}…`, 'stage: house-probe');
            (async () => {
                try {
                    const r = await fetch(pick, { method: 'HEAD', cache: 'no-store' });
                    if (r.ok) {
                        setBootLocal(0.52, `Loading ${pick}…`, 'stage: house-load');
                        if (isHouseGlbUrl(pick)) await loadHouseSceneAsset(pick);
                        else await loadHouse(pick);
                        setBootLocal(1.0, 'Entering K3D world…', 'stage: ready');
                        animate();
                        finishBootOverlay('K3D ready.');
                        return;
                    }
                } catch {}
                const tip = document.getElementById('hud-tip') as HTMLDivElement | null;
                if (tip) tip.textContent = `Not found: ${pick}`;
                setBootLocal(1.0, `Not found: ${pick}`, 'stage: ready');
                animate();
                finishBootOverlay(`Not found: ${pick}`);
            })();
            return;
        }
        // Pick a default galaxy from known names
        const candidates = ['/galaxy.glb', '/coco_50k.glb', '/clotho.glb', '/vatex_2k.glb'];
        setBootLocal(0.22, 'Scanning default houses…', 'stage: house-probe');
        (async () => {
            for (const u of candidates) {
                try {
                    const r = await fetch(u, { method: 'HEAD', cache: 'no-store' });
                    if (r.ok) {
                        setBootLocal(0.55, `Loading ${u}…`, 'stage: house-load');
                        await loadHouse(u);
                        setBootLocal(1.0, 'Entering K3D world…', 'stage: ready');
                        animate();
                        finishBootOverlay('K3D ready.');
                        return;
                    }
                } catch {}
            }
            try {
                const houseUrl = '/house.glb';
                const r = await fetch(houseUrl, { method: 'HEAD', cache: 'no-store' });
                if (r.ok) {
                    setBootLocal(0.55, `Loading ${houseUrl}…`, 'stage: house-load');
                    await loadHouseSceneAsset(houseUrl);
                    setBootLocal(1.0, 'Entering K3D world…', 'stage: ready');
                    animate();
                    finishBootOverlay('K3D ready.');
                    return;
                }
            } catch {}
            const tip = document.getElementById('hud-tip') as HTMLDivElement | null;
            if (tip) tip.textContent = 'No house found. Build one (see docs/RUNBOOK_MULTIMODAL_50K.md).';
            setBootLocal(1.0, 'No house found. Build one to enter the world.', 'stage: ready');
            animate();
            finishBootOverlay('No house found.');
        })();
        return;
    }
    animate();
}

// Start
startApp();
