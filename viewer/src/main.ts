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

fetchK3D('../examples/sample_output.k3d').then((data) => {
  k3dData = data;
});

const loader = new GLTFLoader();
loader.load('../examples/sample_output.gltf', (gltf) => {
  const mesh = gltf.scene.children[0] as THREE.Mesh;
  const geometry = mesh.geometry as THREE.BufferGeometry;
  const count = geometry.getAttribute('position').count;
  const colors = new Float32Array(count * 3);
  const color = new THREE.Color();
  for (let i = 0; i < count; i++) {
    color.setHSL(i / count, 1, 0.5);
    colors[i * 3] = color.r;
    colors[i * 3 + 1] = color.g;
    colors[i * 3 + 2] = color.b;
  }
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const points = new THREE.Points(
    geometry,
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
  const intersects = raycaster.intersectObjects(scene.children, false);
  if (intersects.length > 0 && k3dData.length > 0) {
    const idx = intersects[0].index ?? 0;
    const record = k3dData[idx];
    tooltip.style.display = 'block';
    tooltip.style.left = event.clientX + 5 + 'px';
    tooltip.style.top = event.clientY + 5 + 'px';
    tooltip.textContent = record.id;
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

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
