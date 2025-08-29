import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { loadK3DFromGLTF, fetchCondoConfig, type K3DRecord, type CondoConfig, type HouseInfo, type LoadedK3D } from './loadK3D';
import { K3DAgent } from './agent';
import { ChatClient, type ChatMessage, type CommandMessage } from './chat';
import { RPN } from './rpn';
import { openStore } from './cache';
import { kmeans, palette } from './cluster';

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
let currentPoints: THREE.Points | null = null;
let condoConfig: CondoConfig | null = null;
let agent: K3DAgent | null = null;
let chat: ChatClient | null = null;
const cache = openStore<LoadedK3D>();
let cacheEnabled = true;

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
            colors.set([color.r, color.g, color.b], i * 3);
        });

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({ size: 0.1, vertexColors: true });
        currentPoints = new THREE.Points(geometry, material);
        scene.add(currentPoints);

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
        };
        agent = new K3DAgent(scene, camera, pushExplain);
        agent.setRecords(k3dData);

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
            onStatus: s => chatStatus.textContent = `WS: ${s}`,
            onChat: (m: ChatMessage) => {
                if (m.action) append('* ' + m.from, m.text);
                else if (m.to) append(`${m.from}→${m.to}`, m.text);
                else if (m.channel) append(`[${m.channel}] ${m.from}`, m.text);
                else append(m.from, m.text);
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
                }
            }
        });
        chat.connect();

        // Share dataset graph with live server for routing (ids, neighbors, labels)
        const ids = k3dData.map(r => r.id);
        const neighbors = k3dData.map(r => r.neighbors || []);
        const labelsArr = k3dData.map(r => (r.metadata?.label as string) || r.id);
        chat.sendEvent({ kind: 'dataset_graph', ids, neighbors, labels: labelsArr });

        // Share registered doors (type === 'door') and their spatial addresses
        const doorItems = k3dData
            .map((r) => {
                const isDoor = (r.metadata?.type as string) === 'door';
                if (!isDoor) return null;
                const label = (r.metadata?.label as string) || r.id;
                const address = (window as any).k3dSpatialAddress
                    ? (window as any).k3dSpatialAddress(r.vector as [number, number, number], 1.0, 0, label)
                    : undefined;
                return { label, address };
            })
            .filter(Boolean);
        if (doorItems.length > 0) {
            chat.sendEvent({ kind: 'doors', items: doorItems });
        }

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
    const intersects = raycaster.intersectObject(currentPoints);

    if (intersects.length > 0 && intersects[0].index !== undefined) {
        const idx = intersects[0].index;
        const record = k3dData[idx];

        if (record) {
            tooltip.style.display = 'block';
            tooltip.style.left = `${mouse.x * window.innerWidth / 2 + window.innerWidth / 2 + 5}px`;
            tooltip.style.top = `${-mouse.y * window.innerHeight / 2 + window.innerHeight / 2 + 5}px`;
            const label = (record.metadata?.label as string) || record.id;
            const text = (record.metadata?.text as string) || '';
            tooltip.textContent = text ? `${label}: ${text.slice(0, 120)}` : label;
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

// Agent UI controls
const agentTarget = document.getElementById('agent-target') as HTMLInputElement;
const agentGo = document.getElementById('agent-go') as HTMLButtonElement;
const agentFollow = document.getElementById('agent-follow') as HTMLInputElement;
const chatInput = document.getElementById('chat-input') as HTMLInputElement;
const chatSend = document.getElementById('chat-send') as HTMLButtonElement;
const chatPause = document.getElementById('chat-pause') as HTMLButtonElement;
const chatResume = document.getElementById('chat-resume') as HTMLButtonElement;
const chatStatusBtn = document.getElementById('chat-status-btn') as HTMLButtonElement;
const cacheToggle = document.getElementById('cache-enable') as HTMLInputElement;
const cacheClear = document.getElementById('cache-clear') as HTMLButtonElement;
const colorMode = document.getElementById('color-mode') as HTMLSelectElement;
const clusterK = document.getElementById('cluster-k') as HTMLInputElement;
const applyColor = document.getElementById('apply-color') as HTMLButtonElement;
const legend = document.getElementById('legend') as HTMLDivElement;
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
    const colors = currentPoints.geometry.getAttribute('color') as THREE.BufferAttribute;
    for (let i = 0; i < k3dData.length; i++) {
        const v = k3dData[i].vector;
        const r = size.x > 0 ? (v[0] - min.x) / size.x : 0.5;
        const g = size.y > 0 ? (v[1] - min.y) / size.y : 0.5;
        const b = size.z > 0 ? (v[2] - min.z) / size.z : 0.5;
        colors.setXYZ(i, r, g, b);
    }
    colors.needsUpdate = true;
    legend.textContent = 'position-based coloring';
}

function applyClusterColors(k: number) {
    if (!currentPoints || k3dData.length === 0) return;
    const emb = k3dData.map(d => d.embedding);
    const { labels } = kmeans(emb, Math.max(2, Math.min(20, Math.floor(k))));
    const colors = currentPoints.geometry.getAttribute('color') as THREE.BufferAttribute;
    const pal = palette(Math.max(...labels) + 1);
    const counts: Record<number, number> = {};
    for (let i = 0; i < k3dData.length; i++) {
        const c = labels[i] ?? 0; counts[c] = (counts[c] ?? 0) + 1;
        const tmp = new THREE.Color(pal[c % pal.length]);
        colors.setXYZ(i, tmp.r, tmp.g, tmp.b);
    }
    colors.needsUpdate = true;
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
if (agentFollow) {
    agentFollow.addEventListener('change', () => {
        if (agent) agent.followCamera = !!agentFollow.checked;
    });
}

let last = performance.now();
function animate() {
    const now = performance.now();
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    requestAnimationFrame(animate);
    controls.update();
    checkIntersects();
    if (agent) agent.update(dt);
    renderer.render(scene, camera);
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
        micAnalyzer.getByteTimeDomainData(micData);
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
