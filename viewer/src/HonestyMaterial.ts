import * as THREE from 'three';

export class HonestyMaterial extends THREE.ShaderMaterial {
  constructor(honesty = 1.0) {
    super({
      uniforms: {
        honesty_score: { value: honesty },
        light_direction: { value: new THREE.Vector3(0, -1, 0) },
        camera_position: { value: new THREE.Vector3(0, 0, 5) },
      },
      vertexShader: `
        varying vec3 vWorldPosition;
        varying vec3 vNormal;
        varying float vHonesty;
        uniform float honesty_score;
        void main() {
          vWorldPosition = (modelMatrix * vec4(position, 1.0)).xyz;
          vNormal = normalize(normalMatrix * normal);
          vHonesty = honesty_score;
          gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        }
      `,
      fragmentShader: `
        uniform vec3 light_direction;
        uniform vec3 camera_position;
        varying vec3 vWorldPosition;
        varying vec3 vNormal;
        varying float vHonesty;
        void main() {
          vec3 normal = normalize(vNormal);
          vec3 lightDir = normalize(-light_direction);
          float ndotl = max(dot(normal, lightDir), 0.0);
          vec3 result = vec3(0.7, 0.7, 0.7) * ndotl;
          if (vHonesty < 0.0) {
            result += vec3(1.0, 0.0, 0.0) * (1.0 + vHonesty);
          } else if (vHonesty < 0.5) {
            result += vec3(1.0, 0.5, 0.0) * (0.5 - vHonesty) * 2.0;
          } else {
            result += vec3(0.0, 1.0, 0.0) * (vHonesty - 0.5) * 2.0;
          }
          gl_FragColor = vec4(result, 1.0);
        }
      `,
    });
  }
}

