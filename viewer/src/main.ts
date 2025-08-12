import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { fetchK3D, type K3DRecord } from './loadK3D';

const canvas = document.getElementById('scene') as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({ canvas });
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 2;

const controls = new OrbitControls(camera, renderer.domElement);

let k3dData: K3DRecord[] = [];
let k3dIds: string[] = [];
let recordMap: Map<string, K3DRecord> = new Map();
let pointPositions: THREE.BufferAttribute;

const gltfUrl = '../examples/solar_system.gltf';
const loader = new GLTFLoader();

// Lines for neighbor visualization
const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.5 });
const lineGeometry = new THREE.BufferGeometry();
const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
scene.add(lines);

loader.load(gltfUrl, async (gltf) => {
  const k3dExtension = gltf.parser.json.extensions?.['K3D_nodes'];
  if (k3dExtension) {
    const k3dUrl = new URL(k3dExtension.uri, gltfUrl).href;
    try {
      k3dData = await fetchK3D(k3dUrl);
      recordMap = new Map(k3dData.map((r) => [r.id, r]));
    } catch (e) {
      console.error(e);
    }
  }

  const mesh = gltf.scene.children[0] as THREE.Mesh;
  const primitive = mesh.geometry as THREE.BufferGeometry;
  pointPositions = primitive.getAttribute('position') as THREE.BufferAttribute;

  const primitiveIdsProperty = k3dExtension?.primitiveIdsProperty;
  if (primitiveIdsProperty === 'extras.k3dIds') {
    const ids = gltf.parser.json.meshes?.[0]?.primitives?.[0]?.extras?.k3dIds as string[];
    if (ids) {
      k3dIds = ids;
    }
  }

  const count = pointPositions.count;
  const colors = new Float32Array(count * 3);
  const color = new THREE.Color();

  if (k3dData.length > 0) {
    const vectors = k3dData.map((d) => d.vector);
    const min = new THREE.Vector3(Infinity, Infinity, Infinity);
    const max = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
    for (const vec of vectors) {
      min.x = Math.min(min.x, vec[0]);
      min.y = Math.min(min.y, vec[1]);
      min.z = Math.min(min.z, vec[2]);
      max.x = Math.max(max.x, vec[0]);
      max.y = Math.max(max.y, vec[1]);
      max.z = Math.max(max.z, vec[2]);
    }
    const size = new THREE.Vector3().subVectors(max, min);

    for (let i = 0; i < count; i++) {
      const recordId = k3dIds[i];
      const record = recordMap.get(recordId);

      if (record) {
        const vec = record.vector;
        const r = size.x > 0 ? (vec[0] - min.x) / size.x : 0.5;
        const g = size.y > 0 ? (vec[1] - min.y) / size.y : 0.5;
        const b = size.z > 0 ? (vec[2] - min.z) / size.z : 0.5;
        color.setRGB(r, g, b);
      } else {
        color.setHSL(i / count, 1.0, 0.5);
      }
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
  } else {
    for (let i = 0; i < count; i++) {
      color.setHSL(i / count, 1.0, 0.5);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    }
  }

  primitive.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const points = new THREE.Points(
    primitive,
    new THREE.PointsMaterial({ size: 0.05, vertexColors: true })
  );
  scene.add(points);
});

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tooltip = document.getElementById('tooltip') as HTMLDivElement;

function onMouseMove(event: MouseEvent) {
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(scene.children.filter(c => c.type === 'Points'), false);

  if (intersects.length > 0 && k3dIds.length > 0 && k3dData.length > 0) {
    const idx = intersects[0].index;
    if (idx === undefined) return;

    const recordId = k3dIds[idx];
    const record = recordMap.get(recordId);

    if (record) {
      tooltip.style.display = 'block';
      tooltip.style.left = event.clientX + 5 + 'px';
      tooltip.style.top = event.clientY + 5 + 'px';
      tooltip.textContent = record.id;

      if (record.neighbors && record.neighbors.length > 0) {
        const positions = [];
        const fromVec = new THREE.Vector3().fromBufferAttribute(pointPositions, idx);

        for (const neighborId of record.neighbors) {
          const neighborIdx = k3dIds.indexOf(neighborId);
          if (neighborIdx !== -1) {
            positions.push(fromVec.x, fromVec.y, fromVec.z);
            const toVec = new THREE.Vector3().fromBufferAttribute(pointPositions, neighborIdx);
            positions.push(toVec.x, toVec.y, toVec.z);
          }
        }
        lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        lineGeometry.computeBoundingSphere();
      } else {
        lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute([], 3));
      }
    } else {
      tooltip.style.display = 'none';
      lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute([], 3));
    }
  } else {
    tooltip.style.display = 'none';
    lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute([], 3));
  }
}

window.addEventListener('mousemove', onMouseMove);
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
