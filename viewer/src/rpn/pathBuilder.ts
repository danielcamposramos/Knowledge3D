import * as THREE from 'three';

import { RpnEngine } from './engine';
import { mat4Ops } from './mat4Ops';
import { pathOps } from './pathOps';

export interface PathPoint {
  type: 'move' | 'line' | 'quad' | 'cubic' | 'arc' | 'close';
  coords: number[];
}

type Bounds = { minX: number; minY: number; maxX: number; maxY: number };

function updateBounds(bounds: Bounds, x: number, y: number) {
  bounds.minX = Math.min(bounds.minX, x);
  bounds.minY = Math.min(bounds.minY, y);
  bounds.maxX = Math.max(bounds.maxX, x);
  bounds.maxY = Math.max(bounds.maxY, y);
}

export class PathBuilder {
  private segments: PathPoint[] = [];

  reset(): void {
    this.segments = [];
  }

  move(x: number, y: number): void {
    this.segments.push({ type: 'move', coords: [x, y] });
  }

  line(x: number, y: number): void {
    this.segments.push({ type: 'line', coords: [x, y] });
  }

  quad(cx: number, cy: number, x: number, y: number): void {
    this.segments.push({ type: 'quad', coords: [cx, cy, x, y] });
  }

  cubic(c1x: number, c1y: number, c2x: number, c2y: number, x: number, y: number): void {
    this.segments.push({ type: 'cubic', coords: [c1x, c1y, c2x, c2y, x, y] });
  }

  arc(cx: number, cy: number, radius: number, startAngle: number, endAngle: number): void {
    this.segments.push({ type: 'arc', coords: [cx, cy, radius, startAngle, endAngle] });
  }

  close(): void {
    this.segments.push({ type: 'close', coords: [] });
  }

  hasSegments(): boolean {
    return this.segments.length > 0;
  }

  toCanvas2D(ctx: CanvasRenderingContext2D): void {
    ctx.beginPath();
    for (const segment of this.segments) {
      const [a, b, c, d, e, f] = segment.coords;
      switch (segment.type) {
        case 'move':
          ctx.moveTo(a, b);
          break;
        case 'line':
          ctx.lineTo(a, b);
          break;
        case 'quad':
          ctx.quadraticCurveTo(a, b, c, d);
          break;
        case 'cubic':
          ctx.bezierCurveTo(a, b, c, d, e, f);
          break;
        case 'arc':
          ctx.arc(a, b, c, d, e);
          break;
        case 'close':
          ctx.closePath();
          break;
      }
    }
  }

  toSVGPath(): string {
    const parts: string[] = [];
    for (const segment of this.segments) {
      const coords = segment.coords.map((value) => Number(value.toFixed(4)));
      switch (segment.type) {
        case 'move':
          parts.push(`M ${coords[0]} ${coords[1]}`);
          break;
        case 'line':
          parts.push(`L ${coords[0]} ${coords[1]}`);
          break;
        case 'quad':
          parts.push(`Q ${coords[0]} ${coords[1]} ${coords[2]} ${coords[3]}`);
          break;
        case 'cubic':
          parts.push(`C ${coords[0]} ${coords[1]} ${coords[2]} ${coords[3]} ${coords[4]} ${coords[5]}`);
          break;
        case 'arc':
          parts.push(`A ${coords[2]} ${coords[2]} 0 0 1 ${coords[0]} ${coords[1]}`);
          break;
        case 'close':
          parts.push('Z');
          break;
      }
    }
    return parts.join(' ');
  }

  bounds(): Bounds {
    const points = this.sampledPoints();
    if (!points.length) {
      return { minX: 0, minY: 0, maxX: 0, maxY: 0 };
    }
    const bounds: Bounds = {
      minX: Number.POSITIVE_INFINITY,
      minY: Number.POSITIVE_INFINITY,
      maxX: Number.NEGATIVE_INFINITY,
      maxY: Number.NEGATIVE_INFINITY,
    };
    for (const point of points) updateBounds(bounds, point.x, point.y);
    return bounds;
  }

  applyMatrix(matrix: THREE.Matrix4): void {
    const transformPair = (coords: number[], offset: number) => {
      const point = new THREE.Vector3(coords[offset] || 0, coords[offset + 1] || 0, 0).applyMatrix4(matrix);
      coords[offset] = point.x;
      coords[offset + 1] = point.y;
    };
    for (const segment of this.segments) {
      switch (segment.type) {
        case 'move':
        case 'line':
          transformPair(segment.coords, 0);
          break;
        case 'quad':
          transformPair(segment.coords, 0);
          transformPair(segment.coords, 2);
          break;
        case 'cubic':
          transformPair(segment.coords, 0);
          transformPair(segment.coords, 2);
          transformPair(segment.coords, 4);
          break;
        case 'arc':
          transformPair(segment.coords, 0);
          break;
        case 'close':
          break;
      }
    }
  }

  toShape(): THREE.Shape {
    const shape = new THREE.Shape();
    for (const segment of this.segments) {
      const [a, b, c, d, e, f] = segment.coords;
      switch (segment.type) {
        case 'move':
          shape.moveTo(a, b);
          break;
        case 'line':
          shape.lineTo(a, b);
          break;
        case 'quad':
          shape.quadraticCurveTo(a, b, c, d);
          break;
        case 'cubic':
          shape.bezierCurveTo(a, b, c, d, e, f);
          break;
        case 'arc':
          shape.absarc(a, b, c, d, e, false);
          break;
        case 'close':
          shape.closePath();
          break;
      }
    }
    return shape;
  }

  toLathePoints(): THREE.Vector2[] {
    return this.sampledPoints().map((point) => new THREE.Vector2(point.x, point.y));
  }

  sampledPoints(curveSamples = 12): THREE.Vector2[] {
    const points: THREE.Vector2[] = [];
    let cursor = new THREE.Vector2();
    let subpathStart: THREE.Vector2 | null = null;
    for (const segment of this.segments) {
      const [a, b, c, d, e, f] = segment.coords;
      if (segment.type === 'move') {
        cursor = new THREE.Vector2(a, b);
        subpathStart = cursor.clone();
        points.push(cursor.clone());
        continue;
      }
      if (segment.type === 'line') {
        cursor = new THREE.Vector2(a, b);
        points.push(cursor.clone());
        continue;
      }
      if (segment.type === 'quad') {
        const start = cursor.clone();
        for (let i = 1; i <= curveSamples; i++) {
          const t = i / curveSamples;
          const mt = 1 - t;
          points.push(new THREE.Vector2(
            mt * mt * start.x + 2 * mt * t * a + t * t * c,
            mt * mt * start.y + 2 * mt * t * b + t * t * d,
          ));
        }
        cursor = new THREE.Vector2(c, d);
        continue;
      }
      if (segment.type === 'cubic') {
        const start = cursor.clone();
        for (let i = 1; i <= curveSamples; i++) {
          const t = i / curveSamples;
          const mt = 1 - t;
          points.push(new THREE.Vector2(
            mt * mt * mt * start.x + 3 * mt * mt * t * a + 3 * mt * t * t * c + t * t * t * e,
            mt * mt * mt * start.y + 3 * mt * mt * t * b + 3 * mt * t * t * d + t * t * t * f,
          ));
        }
        cursor = new THREE.Vector2(e, f);
        continue;
      }
      if (segment.type === 'arc') {
        const centerX = a;
        const centerY = b;
        const radius = c;
        const startAngle = d;
        const endAngle = e;
        for (let i = 0; i <= curveSamples; i++) {
          const t = i / curveSamples;
          const angle = startAngle + (endAngle - startAngle) * t;
          points.push(new THREE.Vector2(
            centerX + Math.cos(angle) * radius,
            centerY + Math.sin(angle) * radius,
          ));
        }
        cursor = points[points.length - 1].clone();
        continue;
      }
      if (segment.type === 'close' && subpathStart) {
        cursor = subpathStart.clone();
        points.push(cursor.clone());
      }
    }
    return points;
  }
}

export function rpnToSVG(program: string): SVGSVGElement {
  const engine = new RpnEngine();
  engine.context.path = new PathBuilder();
  engine.registerModule(mat4Ops);
  engine.registerModule(pathOps);
  engine.execute(program);
  const path = engine.context.path!;
  const bounds = path.bounds();
  const width = Math.max(1, bounds.maxX - bounds.minX);
  const height = Math.max(1, bounds.maxY - bounds.minY);
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', `${bounds.minX - 8} ${bounds.minY - 8} ${width + 16} ${height + 16}`);
  svg.setAttribute('width', '100%');
  svg.setAttribute('height', '100%');
  const pathEl = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  pathEl.setAttribute('d', path.toSVGPath());
  pathEl.setAttribute('fill', 'rgba(148, 210, 189, 0.22)');
  pathEl.setAttribute('stroke', '#94d2bd');
  pathEl.setAttribute('stroke-width', '0.04');
  svg.appendChild(pathEl);
  return svg;
}
