import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { fetchK3D, fetchCondoConfig, type K3DRecord, type CondoConfig, type HouseInfo } from './loadK3D';
import { K3DAgent } from './agent';

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
}

/**
 * Loads and displays a house from the given K3D data URL.
 * @param k3dUrl The URL of the .k3d file to load.
 */
async function loadHouse(k3dUrl: string) {
    clearScene();

    try {
        k3dData = await fetchK3D(k3dUrl);
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

        k3dData.forEach((record, i) => {
            positions.set(record.vector, i * 3);

            const r = size.x > 0 ? (record.vector[0] - min.x) / size.x : 0.5;
            const g = size.y > 0 ? (record.vector[1] - min.y) / size.y : 0.5;
            const b = size.z > 0 ? (record.vector[2] - min.z) / size.z : 0.5;
            color.setRGB(r, g, b);
            colors.set([color.r, color.g, color.b], i * 3);
        });

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({ size: 0.1, vertexColors: true });
        currentPoints = new THREE.Points(geometry, material);
        scene.add(currentPoints);

        // Initialize or reset the agent
        agent = new K3DAgent(scene, camera);
        agent.setRecords(k3dData);

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
if (agentGo) {
    agentGo.addEventListener('click', () => {
        if (agent && agentTarget?.value) {
            agent.goToLabel(agentTarget.value);
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

// --- Start Application ---
initCondoSelector();
animate();
