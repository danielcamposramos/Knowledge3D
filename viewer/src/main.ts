import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { loadK3DFromGLTF, fetchCondoConfig, type K3DRecord, type CondoConfig, type HouseInfo, type LoadedK3D } from './loadK3D';
import { K3DAgent } from './agent';
import { Tablet3D } from './tablet';
import { ChatClient, type ChatMessage, type CommandMessage } from './chat';
import { RPN } from './rpn';
import { openStore } from './cache';
import { kmeans, palette } from './cluster';
import { AISuggestionManager, DynamicLayerManager, LODRenderer, GridCulledPoints } from './extensions/smartGraph';

// --- DOM Elements ---
const canvas = document.getElementById('scene') as HTMLCanvasElement;
const tooltip = document.getElementById('tooltip') as HTMLDivElement;
const expertSelect = document.getElementById('expert-select') as HTMLSelectElement;

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

// --- Main Logic ---

/**
 * Clears the current 3D scene of any house-related objects.
 */
function clearScene() {
    if (currentPoints) {
        scene.remove(currentPoints);
        currentPoints.geometry.dispose();
        (currentPoints.material as THREE.Material).dispose();
        currentPoints = null;
    }
    if (edgesObject) {
        scene.remove(edgesObject);
        (edgesObject.geometry as THREE.BufferGeometry).dispose();
        (edgesObject.material as THREE.Material).dispose();
        edgesObject = null;
    }
    lod = null;
    layersMgr = null;
    k3dData = [];
    recordMap.clear();
    if (agent) {
        scene.remove(agent.object);
        agent = null;
    }
    if (chat) {
        chat.disconnect();
        chat = null;
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

            const isDoor = (record.metadata?.type as string) === 'door';
            if (mask && mask[i] === true) {
                // AI per-node cue: green for nodes with new info
                color.setRGB(0.0, 1.0, 0.0);
            } else if (isDoor) {
                color.setRGB(0.2, 0.5, 1.0); // bluish for doors
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
            const colorsAttr = currentPoints.geometry.getAttribute('color') as THREE.BufferAttribute;
            for (let i = 0; i < k3dData.length; i++) colorsAttr.setXYZ(i, 0.0, 1.0, 0.0);
            colorsAttr.needsUpdate = true;
        }

        // dataset info
        const infoEl = document.getElementById('dataset-info') as HTMLDivElement;
        const inf = loaded.info;
        const fmt = (b?: number) => b !== undefined ? `${(b/1e6).toFixed(2)} MB` : 'n/a';
        infoEl.textContent = `precision=${inf.precision} dims=${inf.dims} count=${inf.count} vectors=${fmt(inf.byteLengthVectors)} embeddings=${fmt(inf.byteLengthEmbeddings)}`;
        // quick scoreboard lines
        const mask = loaded.info?.ai?.mask?.has_new_information;
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
            const emat = new THREE.LineBasicMaterial({ color: 0x66cc66, transparent: true, opacity: 0.25 });
            const lines = new THREE.LineSegments(eg, emat);
            // draw slightly behind points to reduce overdraw
            (lines.material as THREE.LineBasicMaterial).depthWrite = false;
            scene.add(lines);
            edgesObject = lines;
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
            }
        });
        // Update tablet with house info and dataset
        tablet.setStatus({ house: k3dUrl, nodes: k3dData.length, info: `dims=${loaded.info.dims} precision=${loaded.info.precision}` });
        tablet.setDataset(k3dData);

        // Start chat connection
        const chatLog = document.getElementById('chat-log') as HTMLDivElement;
        const chatStatus = document.getElementById('chat-status') as HTMLDivElement;
        const append = (from: string, text: string) => {
            const el = document.createElement('div');
            el.textContent = `${from}: ${text}`;
            chatLog.appendChild(el);
            chatLog.scrollTop = chatLog.scrollHeight;
        };
        chat = new ChatClient('ws://localhost:8765', {
            onStatus: async (s) => {
                chatStatus.textContent = `WS: ${s}`;
                const q = await chat!.getQueueLength();
                const tinfo = document.getElementById('tablet-info') as HTMLDivElement;
                if (tinfo) tinfo.textContent = `Tablet: ${s === 'connected' ? 'online' : 'offline'}, queue=${q}`;
                if (tablet) tablet.setStatus({ ws: s, queue: q });
            },
            onChat: (m: ChatMessage) => {
                if (m.action) append('* ' + m.from, m.text);
                else if (m.to) append(`${m.from}→${m.to}`, m.text);
                else if (m.channel) append(`[${m.channel}] ${m.from}`, m.text);
                else append(m.from, m.text);
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
                }
            }
        });
        // Provide context for logging and continuity
        chat.setContext({ house: k3dUrl, mode: 'ai' });
        chat.connect();

        // Share dataset graph with live server for routing (ids, neighbors, labels, positions)
        const ids = k3dData.map(r => r.id);
        const neighbors = k3dData.map(r => r.neighbors || []);
        const labelsArr = k3dData.map(r => (r.metadata?.label as string) || r.id);
        const positions = k3dData.map(r => r.vector as [number, number, number]);
        chat.sendEvent({ kind: 'dataset_graph', ids, neighbors, labels: labelsArr, positions });
        await tabletStore.put(`graph:${k3dUrl}`, { ids, neighbors, labels: labelsArr });

        // Share registered doors (type === 'door') and their spatial addresses
        const doorItems = k3dData
            .map((r) => {
                const isDoor = (r.metadata?.type as string) === 'door';
                if (!isDoor) return null;
                const label = (r.metadata?.label as string) || r.id;
                let address: string | undefined = undefined;
                try {
                    // Prefer explicit metadata address for inter-house links
                    const metaAddr = (r.metadata?.address as string) || undefined;
                    if (metaAddr && typeof metaAddr === 'string') address = metaAddr;
                    if (!address && (window as any).k3dSpatialAddress) {
                        address = (window as any).k3dSpatialAddress(r.vector as [number, number, number], 1.0, 0, label);
                    }
                } catch {}
                return { label, address };
            })
            .filter(Boolean);
        if (doorItems.length > 0) {
            chat.sendEvent({ kind: 'doors', items: doorItems });
            await tabletStore.put(`doors:${k3dUrl}`, doorItems);
        }

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

function onMouseMove(event: MouseEvent) {
    mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
}

function checkIntersects() {
    if (!currentPoints) {
        tooltip.style.display = 'none';
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
            const isDoor = (record.metadata?.type as string) === 'door';
            let extra = '';
            try {
                const addr = (window as any).k3dSpatialAddress
                    ? (window as any).k3dSpatialAddress(record.vector as [number, number, number], 1.0, 0, label)
                    : undefined;
                if (addr) extra = ` [${addr}]`;
            } catch {}
            const head = isDoor ? `🚪 ${label}` : label;
            tooltip.textContent = text ? `${head}: ${text.slice(0, 120)}${extra}` : (head + extra);
            if (tablet) tablet.setFocusLabel(label);
        } else {
            tooltip.style.display = 'none';
        }
    } else {
        tooltip.style.display = 'none';
    }
}

window.addEventListener('mousemove', onMouseMove);
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

// --- Start Application ---
initCondoSelector();
animate();
