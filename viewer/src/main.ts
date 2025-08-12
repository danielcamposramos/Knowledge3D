import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { fetchK3D, type K3DNode } from './loadK3D';

// Basic scene setup
const canvas = document.getElementById('scene') as HTMLCanvasElement;
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
});
renderer.setSize(window.innerWidth, window.innerHeight);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.z = 10;
scene.add(camera);

const controls = new OrbitControls(camera, renderer.domElement);
controls.autoRotate = true;

// Data stores
let k3dNodeMap = new Map<string, K3DNode>();
const renderableObjects: THREE.Object3D[] = [];

// Tooltip setup
const tooltip = document.getElementById('tooltip') as HTMLDivElement;

// Load the glTF scene, which acts as the entry point
const loader = new GLTFLoader();
loader.load(
  '../examples/sample_output.gltf',
  async (gltf) => {
    console.log('glTF loaded:', gltf);

    // 1. Find the K3D extension to locate the .k3d file
    const k3dExtension = gltf.parser.json.extensions?.['K3D_nodes'];
    if (!k3dExtension) {
      console.error('K3D_nodes extension not found in glTF file.');
      return;
    }

    // 2. Fetch and process the .k3d file
    const k3dUrl = new URL(k3dExtension.uri, gltf.parser.options.path || window.location.href);
    try {
      k3dNodeMap = await fetchK3D(k3dUrl.href);
      console.log('K3D data loaded:', k3dNodeMap);
    } catch (error) {
      console.error('Failed to load K3D data:', error);
      return;
    }

    // 3. Process the glTF scene nodes
    const pointMaterial = new THREE.PointsMaterial({
      size: 0.1,
      vertexColors: true,
    });

    gltf.scene.children.forEach((node) => {
      if (node instanceof THREE.Points) {
        // The generator creates Points objects directly now.
        // In a future version, it might be Meshes.
        const points = node as THREE.Points;
        const k3dId = points.userData.k3dId;
        const k3dNode = k3dNodeMap.get(k3dId);

        if (k3dNode) {
          // Assign a color based on the node's ID hash for variety
          const color = new THREE.Color().setHSL(
            (parseInt(k3dId.replace('node-', ''), 10) / k3dNodeMap.size) % 1,
            1,
            0.5
          );
          (points.geometry as THREE.BufferGeometry).setAttribute(
            'color',
            new THREE.Float32BufferAttribute([color.r, color.g, color.b], 3)
          );
          points.material = pointMaterial;
          renderableObjects.push(points);
        }
      }
    });

    scene.add(gltf.scene);
  },
  undefined, // onProgress callback not needed
  (error) => {
    console.error('An error happened during glTF loading:', error);
  }
);

// Raycasting for interactivity
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function onMouseMove(event: MouseEvent) {
  // Update mouse position
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;

  // Update raycaster
  raycaster.setFromCamera(mouse, camera);
  raycaster.params.Points.threshold = 0.1; // Adjust threshold for easier hovering

  // Check for intersections
  const intersects = raycaster.intersectObjects(renderableObjects, false);

  if (intersects.length > 0) {
    const intersectedObject = intersects[0].object as THREE.Object3D;
    const k3dId = intersectedObject.userData.k3dId;
    const k3dNode = k3dNodeMap.get(k3dId);

    if (k3dNode) {
      tooltip.style.display = 'block';
      tooltip.style.left = `${event.clientX + 10}px`;
      tooltip.style.top = `${event.clientY + 10}px`;
      tooltip.textContent = k3dNode.metadata.label;
    }
  } else {
    tooltip.style.display = 'none';
  }
}

// Window event listeners
window.addEventListener('mousemove', onMouseMove);
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Animation loop
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
