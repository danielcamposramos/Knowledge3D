"""
3D Model WINE Adapter - Convert external 3D models to sovereign PTX execution

Purpose:
- Bridge TRELLIS and HunyuanWorld 3D models with K3D's sovereign GPU execution
- Convert external model outputs to procedural RPN programs for GPU execution
- Maintain architectural sovereignty by isolating external dependencies to ingestion phase

Features:
- TRELLIS adapter: Convert GLB meshes to sovereign RPN construction programs
- HunyuanWorld adapter: Convert scene graphs to procedural RPN compositions
- Multi-modal fusion: Combine text, image, video inputs for 3D generation
- Tablet envelope system: Route through existing tablet infrastructure
"""

from __future__ import annotations

import json
import hashlib
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from knowledge3d.bridge.headless_tablet import TabletEnvelope, TabletIngest


# Define supported 3D model types
MODEL_TYPE_TRELLIS = "trellis"
MODEL_TYPE_HUNYUAN = "hunyuan"
MODEL_TYPE_PROCEDURAL = "procedural"

# Default galaxies for 3D operations
DEFAULT_3D_GALAXIES: Tuple[str, ...] = ("Drawing", "Reality", "Visual", "Spatial")


class TRELLISWineAdapter:
    """WINE-like adapter for TRELLIS 3D model integration."""
    
    def __init__(self):
        self.model_type = MODEL_TYPE_TRELLIS
        self._last_parse_errors: List[str] = []
    
    def ingest_trellis_output(self, trellis_glb_path: str, *, task_id: str | None = None) -> TabletEnvelope:
        """Convert TRELLIS GLB output to tablet-compatible procedural RPN."""
        # Parse TRELLIS GLB file
        mesh_data = self._parse_trellis_glb(trellis_glb_path)
        
        # Extract semantic embeddings and metadata
        text_embedding = mesh_data.get("text_embedding", [])
        shape_params = dict(mesh_data.get("shape_parameters", {}))
        shape_params["mesh_metadata"] = dict(mesh_data.get("metadata", {}))
        
        # Convert to procedural RPN programs
        rpn_program = self._mesh_to_rpn(mesh_data)
        
        # Create tablet envelope for sovereign execution
        return TabletIngest.procedural_3d_task(
            task_id=task_id or _stable_task_id("trellis", trellis_glb_path),
            source="trellis_external",
            rpn_program=rpn_program,
            embeddings=text_embedding,
            metadata=shape_params
        )
    
    def _parse_trellis_glb(self, glb_path: str) -> Dict[str, Any]:
        """Parse TRELLIS-generated GLB file to extract mesh data."""
        try:
            from pygltflib import GLTF2
            gltf = GLTF2().load(glb_path)
            self._last_parse_errors = []
            
            mesh_data = {
                "vertices": [],
                "faces": [],
                "materials": [],
                "embeddings": [],
                "metadata": {
                    "source_path": str(glb_path),
                    "parse_errors": [],
                }
            }
            
            # Extract mesh data from first primitive
            if gltf.meshes and gltf.meshes[0].primitives:
                primitive = gltf.meshes[0].primitives[0]
                
                # Extract vertices
                if "POSITION" in primitive.attributes:
                    pos_accessor = gltf.accessors[primitive.attributes["POSITION"]]
                    if pos_accessor.bufferView is not None:
                        buffer_view = gltf.bufferViews[pos_accessor.bufferView]
                        buffer = gltf.buffers[buffer_view.buffer]
                        
                        try:
                            vertices = self._extract_vertex_data(buffer, buffer_view, pos_accessor)
                            mesh_data["vertices"] = vertices
                        except ValueError as exc:
                            self._last_parse_errors.append(f"vertex_extract_failed:{exc}")
                
                # Extract indices (faces)
                if primitive.indices is not None:
                    indices_accessor = gltf.accessors[primitive.indices]
                    if indices_accessor.bufferView is not None:
                        buffer_view = gltf.bufferViews[indices_accessor.bufferView]
                        buffer = gltf.buffers[buffer_view.buffer]
                        
                        try:
                            indices = self._extract_index_data(buffer, buffer_view, indices_accessor)
                            mesh_data["faces"] = indices
                        except ValueError as exc:
                            self._last_parse_errors.append(f"index_extract_failed:{exc}")
                
                # Extract extras (embeddings/metadata)
                if primitive.extras:
                    mesh_data["metadata"] = primitive.extras
                    if "text_embedding" in primitive.extras:
                        mesh_data["text_embedding"] = primitive.extras["text_embedding"]
                    if "shape_parameters" in primitive.extras:
                        mesh_data["shape_parameters"] = primitive.extras["shape_parameters"]

            mesh_data["metadata"]["vertex_count"] = len(mesh_data["vertices"])
            mesh_data["metadata"]["face_count"] = len(mesh_data["faces"])
            if self._last_parse_errors:
                mesh_data["metadata"]["parse_errors"] = list(self._last_parse_errors)
            
            return mesh_data
            
        except Exception as e:
            # Fallback to empty mesh data
            return {
                "vertices": [],
                "faces": [],
                "materials": [],
                "embeddings": [],
                "metadata": {"error": str(e), "source_path": str(glb_path), "parse_errors": [str(e)]}
            }
    
    def _extract_vertex_data(self, buffer, buffer_view, accessor) -> List[List[float]]:
        """Extract vertex position data from GLB buffer."""
        vertices = []
        
        buffer_data = buffer.data
        if not buffer_data:
            return vertices

        byte_offset = (accessor.byteOffset or 0) + (buffer_view.byteOffset or 0)
        component_type = accessor.componentType
        type_info = accessor.type
        if component_type != 5126 or type_info != "VEC3":
            raise ValueError(
                f"unsupported vertex accessor component_type={component_type} type={type_info}"
            )

        for i in range(accessor.count):
            offset = byte_offset + i * 12
            if offset + 12 > len(buffer_data):
                raise ValueError("vertex buffer truncated before declared accessor count")
            x = struct.unpack('f', buffer_data[offset:offset+4])[0]
            y = struct.unpack('f', buffer_data[offset+4:offset+8])[0]
            z = struct.unpack('f', buffer_data[offset+8:offset+12])[0]
            vertices.append([x, y, z])

        return vertices
    
    def _extract_index_data(self, buffer, buffer_view, accessor) -> List[List[int]]:
        """Extract triangle index data from GLB buffer."""
        indices = []
        
        buffer_data = buffer.data
        if not buffer_data:
            return indices

        byte_offset = (accessor.byteOffset or 0) + (buffer_view.byteOffset or 0)
        component_type = accessor.componentType
        if component_type == 5123:
            stride = 2
            unpack_token = 'H'
        elif component_type == 5125:
            stride = 4
            unpack_token = 'I'
        else:
            raise ValueError(f"unsupported index component_type={component_type}")

        if accessor.count % 3 != 0:
            raise ValueError("index accessor does not describe triangles")

        for i in range(0, accessor.count, 3):
            offset = byte_offset + i * stride
            required = 3 * stride
            if offset + required > len(buffer_data):
                raise ValueError("index buffer truncated before declared accessor count")
            a = struct.unpack(unpack_token, buffer_data[offset:offset + stride])[0]
            b = struct.unpack(unpack_token, buffer_data[offset + stride:offset + 2 * stride])[0]
            c = struct.unpack(unpack_token, buffer_data[offset + 2 * stride:offset + 3 * stride])[0]
            indices.append([a, b, c])

        return indices
    
    def _mesh_to_rpn(self, mesh_data: Dict[str, Any]) -> str:
        """Convert mesh vertices/faces to sovereign RPN construction program."""
        vertices = mesh_data.get("vertices", [])
        faces = mesh_data.get("faces", [])
        
        # Generate RPN for mesh construction
        rpn_tokens = []
        
        # Define mesh construction sequence
        rpn_tokens.append("MESH_BEGIN")
        rpn_tokens.append(str(len(vertices)))  # Vertex count
        rpn_tokens.append(str(len(faces)))     # Face count
        
        # Add vertices
        for vertex in vertices:
            if len(vertex) >= 3:
                rpn_tokens.extend([
                    str(vertex[0]), str(vertex[1]), str(vertex[2]), "VERTEX3"
                ])
        
        # Add faces
        for face in faces:
            if len(face) >= 3:
                rpn_tokens.extend([
                    str(face[0]), str(face[1]), str(face[2]), "TRI_FACE"
                ])
        
        # Add metadata if available
        if mesh_data.get("metadata"):
            rpn_tokens.append("METADATA_BEGIN")
            for key, value in mesh_data["metadata"].items():
                rpn_tokens.extend([str(key), str(value), "META_PAIR"])
            rpn_tokens.append("METADATA_END")
        
        rpn_tokens.append("MESH_END")
        
        return " ".join(rpn_tokens)


class HunyuanWineAdapter:
    """WINE-like adapter for HunyuanWorld 3D scene integration."""
    
    def __init__(self):
        self.model_type = MODEL_TYPE_HUNYUAN
    
    def ingest_hunyuan_output(self, hunyuan_scene_path: str, *, task_id: str | None = None) -> TabletEnvelope:
        """Convert HunyuanWorld scene output to tablet-compatible procedural RPN."""
        # Parse HunyuanWorld scene data
        scene_data = self._parse_hunyuan_scene(hunyuan_scene_path)
        
        # Extract scene graph and semantic information
        scene_graph = scene_data.get("scene_graph", {})
        text_embedding = scene_data.get("text_embedding", [])
        route_metadata = {
            "scene_graph": scene_graph,
            "scene_metadata": dict(scene_data.get("metadata", {})),
        }
        
        # Convert to procedural RPN programs
        rpn_program = self._scene_to_rpn(scene_data)
        
        # Create tablet envelope for sovereign execution
        return TabletIngest.procedural_3d_task(
            task_id=task_id or _stable_task_id("hunyuan", hunyuan_scene_path),
            source="hunyuan_external",
            rpn_program=rpn_program,
            embeddings=text_embedding,
            metadata=route_metadata
        )
    
    def _parse_hunyuan_scene(self, scene_path: str) -> Dict[str, Any]:
        """Parse HunyuanWorld-generated scene file."""
        try:
            # Assume JSON format for scene description
            scene_file = Path(scene_path)
            if scene_file.suffix.lower() == '.json':
                with open(scene_file, 'r', encoding='utf-8') as f:
                    scene_data = json.load(f)
            else:
                # Fallback: create basic scene structure
                scene_data = {
                    "scene_graph": {
                        "root": {
                            "type": "scene",
                            "objects": []
                        }
                    },
                    "text_embedding": [],
                    "metadata": {"source": "hunyuan_fallback"}
                }
            
            return scene_data
            
        except Exception as e:
            # Fallback to empty scene data
            return {
                "scene_graph": {
                    "root": {
                        "type": "scene",
                        "objects": []
                    }
                },
                "text_embedding": [],
                "metadata": {"error": str(e)}
            }
    
    def _scene_to_rpn(self, scene_data: Dict[str, Any]) -> str:
        """Convert scene graph to sovereign RPN composition program."""
        scene_graph = scene_data.get("scene_graph", {})
        
        rpn_tokens = []
        
        # Begin scene composition
        rpn_tokens.append("SCENE_BEGIN")
        
        # Process scene hierarchy
        if "root" in scene_graph:
            self._process_scene_node(scene_graph["root"], rpn_tokens)
        
        # Add scene metadata
        if scene_data.get("text_embedding"):
            rpn_tokens.append("TEXT_EMBED_BEGIN")
            for embed_val in scene_data["text_embedding"]:
                rpn_tokens.extend([str(embed_val), "EMBED_VAL"])
            rpn_tokens.append("TEXT_EMBED_END")
        
        rpn_tokens.append("SCENE_END")
        
        return " ".join(rpn_tokens)
    
    def _process_scene_node(self, node: Dict[str, Any], rpn_tokens: List[str]) -> None:
        """Recursively process scene graph node."""
        node_type = node.get("type", "unknown")
        
        rpn_tokens.extend(["NODE_BEGIN", node_type])
        
        # Process node properties
        if "position" in node:
            pos = node["position"]
            if len(pos) >= 3:
                rpn_tokens.extend([
                    str(pos[0]), str(pos[1]), str(pos[2]), "NODE_POSITION"
                ])
        
        if "rotation" in node:
            rot = node["rotation"]
            if len(rot) >= 4:  # Quaternion
                rpn_tokens.extend([
                    str(rot[0]), str(rot[1]), str(rot[2]), str(rot[3]), "NODE_ROTATION"
                ])
        
        if "scale" in node:
            scale = node["scale"]
            if len(scale) >= 3:
                rpn_tokens.extend([
                    str(scale[0]), str(scale[1]), str(scale[2]), "NODE_SCALE"
                ])
        
        # Process child objects
        if "objects" in node:
            for obj in node["objects"]:
                self._process_scene_object(obj, rpn_tokens)
        
        rpn_tokens.append("NODE_END")
    
    def _process_scene_object(self, obj: Dict[str, Any], rpn_tokens: List[str]) -> None:
        """Process individual scene object."""
        obj_type = obj.get("type", "unknown")
        
        rpn_tokens.extend(["OBJECT_BEGIN", obj_type])
        
        # Process object properties
        if "mesh" in obj:
            mesh_ref = obj["mesh"]
            rpn_tokens.extend([str(mesh_ref), "OBJECT_MESH"])
        
        if "material" in obj:
            material_ref = obj["material"]
            rpn_tokens.extend([str(material_ref), "OBJECT_MATERIAL"])
        
        if "transform" in obj:
            transform = obj["transform"]
            if isinstance(transform, dict):
                if "translation" in transform:
                    trans = transform["translation"]
                    if len(trans) >= 3:
                        rpn_tokens.extend([
                            str(trans[0]), str(trans[1]), str(trans[2]), "OBJECT_TRANSLATION"
                        ])
        
        rpn_tokens.append("OBJECT_END")


class Procedural3DAdapter:
    """Adapter for procedural 3D generation without external models."""
    
    def __init__(self):
        self.model_type = MODEL_TYPE_PROCEDURAL
    
    def generate_procedural_3d(self, generation_params: Dict[str, Any], *, task_id: str | None = None) -> TabletEnvelope:
        """Generate procedural 3D content based on parameters."""
        # Extract generation parameters
        primitive_type = generation_params.get("primitive_type", "cube")
        dimensions = generation_params.get("dimensions", [1.0, 1.0, 1.0])
        position = generation_params.get("position", [0.0, 0.0, 0.0])
        
        # Generate procedural RPN program
        rpn_program = self._generate_procedural_rpn(primitive_type, dimensions, position)
        
        # Create tablet envelope
        return TabletIngest.procedural_3d_task(
            task_id=task_id or _stable_task_id("procedural", json.dumps(generation_params, sort_keys=True)),
            source="procedural_3d",
            rpn_program=rpn_program,
            embeddings=[],
            metadata=generation_params
        )
    
    def _generate_procedural_rpn(self, primitive_type: str, dimensions: List[float], position: List[float]) -> str:
        """Generate procedural RPN for basic 3D primitive."""
        rpn_tokens = []
        
        rpn_tokens.append("PROCEDURAL_3D_BEGIN")
        rpn_tokens.extend([primitive_type, "PRIMITIVE_TYPE"])
        
        # Add dimensions
        for dim in dimensions[:3]:  # Max 3 dimensions
            rpn_tokens.extend([str(dim), "DIMENSION"])
        
        # Add position
        for pos in position[:3]:  # Max 3 coordinates
            rpn_tokens.extend([str(pos), "POSITION"])
        
        rpn_tokens.append("PROCEDURAL_3D_END")
        
        return " ".join(rpn_tokens)


# Extend TabletIngest with 3D task method
def _procedural_3d_task(
    *,
    task_id: str,
    source: str,
    rpn_program: str,
    embeddings: List[float],
    metadata: Dict[str, Any],
    specialist: str = "visual_3d",
    domain_hint: str = "3d_model_generation",
    galaxies: Sequence[str] | None = None,
) -> TabletEnvelope:
    """Create tablet envelope for procedural 3D generation tasks."""
    route_galaxies = tuple(
        str(name)
        for name in (galaxies or DEFAULT_3D_GALAXIES)
        if str(name).strip()
    )
    
    task_payload = {
        "surface_kind": "SPATIAL_3D",
        "task_id": str(task_id),
        "query": f"3D model generation from {source}",
        "source": str(source),
        "rpn_program": str(rpn_program),
        "embeddings": list(embeddings),
        "metadata": dict(metadata),
        "program_type": "gpu_task_dispatch_sovereign",
        "trm_dispatch": True,
    }
    
    return TabletEnvelope(
        surface_kind="SPATIAL_3D",
        task_id=str(task_id),
        query=f"3D model generation from {source}",
        specialist=str(specialist),
        domain_hint=str(domain_hint),
        galaxies=route_galaxies,
        task=task_payload,
        metadata={
            "source": source,
            "rpn_program_length": len(rpn_program),
            "embedding_dims": len(embeddings),
        }
    )


# Monkey patch the method into TabletIngest
TabletIngest.procedural_3d_task = staticmethod(_procedural_3d_task)


class External3DWineBridge:
    """WINE bridge for external 3D models - converts old paradigm to sovereign RPN."""
    
    def __init__(self):
        self.model_registry = {
            MODEL_TYPE_TRELLIS: TRELLISWineAdapter(),
            MODEL_TYPE_HUNYUAN: HunyuanWineAdapter(),
            MODEL_TYPE_PROCEDURAL: Procedural3DAdapter(),
        }
    
    def bridge_external_3d(self, model_type: str, external_data: Dict[str, Any], *, task_id: str | None = None) -> Dict[str, Any]:
        """Convert external 3D model output to sovereign RPN program."""
        adapter = self.model_registry.get(model_type)
        if not adapter:
            raise ValueError(f"Unknown 3D model type: {model_type}")
        
        # Route through appropriate adapter
        if model_type == MODEL_TYPE_TRELLIS:
            envelope = adapter.ingest_trellis_output(external_data.get("file_path", ""), task_id=task_id)
        elif model_type == MODEL_TYPE_HUNYUAN:
            envelope = adapter.ingest_hunyuan_output(external_data.get("file_path", ""), task_id=task_id)
        elif model_type == MODEL_TYPE_PROCEDURAL:
            envelope = adapter.generate_procedural_3d(external_data.get("params", {}), task_id=task_id)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Return procedural RPN program for GPU execution
        return {
            "envelope": envelope,
            "rpn_program": envelope.task.get("rpn_program", ""),
            "specialist": envelope.specialist,
            "domain_hint": envelope.domain_hint,
            "galaxy_names": list(envelope.galaxies),
            "source": envelope.task.get("source", "unknown"),
        }


# Convenience functions for external integration
def build_3d_route(
    *,
    specialist: str = "visual_3d",
    domain_hint: str | None = "3d_model_generation",
    galaxies: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Build route configuration for 3D model tasks."""
    route = {
        "specialist": str(specialist or "visual_3d"),
        "domain_hint": str(domain_hint).strip() if domain_hint is not None else None,
    }
    galaxy_names = [
        str(name)
        for name in (galaxies or DEFAULT_3D_GALAXIES)
        if str(name).strip()
    ]
    if galaxy_names:
        route["galaxy_names"] = galaxy_names
    return route


def build_3d_task(
    *,
    task_id: str,
    model_type: str,
    external_data: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build 3D task with external model integration."""
    bridge = External3DWineBridge()
    result = bridge.bridge_external_3d(model_type, external_data, task_id=task_id)
    
    envelope = result["envelope"]
    task = dict(envelope.task)
    if metadata:
        merged = dict(task.get("metadata", {}))
        merged.update(metadata)
        task["metadata"] = merged
    return task, build_3d_route(
        specialist=envelope.specialist,
        domain_hint=envelope.domain_hint,
        galaxies=envelope.galaxies,
    )


# Multi-modal 3D generation with WINE bridge
def create_multimodal_3d_task(
    *,
    task_id: str,
    text_prompt: str,
    image_path: str | None = None,
    video_path: str | None = None,
    model_type: str = MODEL_TYPE_TRELLIS,
    generation_params: Dict[str, Any] | None = None,
) -> TabletEnvelope:
    """Create multi-modal 3D generation task."""
    # Generate embeddings for text prompt
    text_embedding = _generate_text_embedding(text_prompt)
    
    # Create external data structure
    external_data = {
        "text_prompt": text_prompt,
        "text_embedding": text_embedding,
        "generation_params": generation_params or {},
        "multimodal_flags": {
            "text": True,
            "image": bool(image_path),
            "video": bool(video_path),
        }
    }
    
    # Add image/video paths if provided
    if image_path:
        external_data["image_path"] = image_path
    if video_path:
        external_data["video_path"] = video_path
    
    # Create procedural RPN for multi-modal fusion
    rpn_program = _create_multimodal_rpn(external_data)
    
    return TabletIngest.procedural_3d_task(
        task_id=task_id,
        source=f"{model_type}_multimodal",
        rpn_program=rpn_program,
        embeddings=text_embedding,
        metadata=external_data,
        specialist="visual_3d",
        domain_hint="multimodal_3d_generation",
    )


def _generate_text_embedding(text: str) -> List[float]:
    """Generate a deterministic ingestion-side hash embedding.

    This is an ingestion helper only. It does not claim semantic model quality;
    it provides a stable signal for routing, deduplication, and artifact linking
    when an external embedding model is intentionally out of the runtime path.
    """
    payload = (text or "").encode("utf-8")
    if not payload:
        return [0.0] * 512
    values: List[float] = []
    counter = 0
    while len(values) < 512:
        digest = hashlib.sha256(payload + counter.to_bytes(4, "little")).digest()
        for byte in digest:
            values.append((byte / 127.5) - 1.0)
            if len(values) == 512:
                break
        counter += 1
    return values


def _stable_task_id(prefix: str, payload: str) -> str:
    digest = hashlib.sha256(f"{prefix}:{payload}".encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _create_multimodal_rpn(external_data: Dict[str, Any]) -> str:
    """Create RPN program for multi-modal 3D generation."""
    rpn_tokens = []
    
    rpn_tokens.append("MULTIMODAL_3D_BEGIN")
    
    # Text input
    text_prompt = external_data.get("text_prompt", "")
    rpn_tokens.extend([f'"{text_prompt}"', "TEXT_INPUT"])
    
    # Multi-modal flags
    flags = external_data.get("multimodal_flags", {})
    if flags.get("text"):
        rpn_tokens.append("TEXT_ENABLED")
    if flags.get("image"):
        rpn_tokens.append("IMAGE_ENABLED")
    if flags.get("video"):
        rpn_tokens.append("VIDEO_ENABLED")
    
    # Generation parameters
    params = external_data.get("generation_params", {})
    if "temperature" in params:
        rpn_tokens.extend([str(params["temperature"]), "GENERATION_TEMPERATURE"])
    if "style" in params:
        rpn_tokens.extend([f'"{params["style"]}"', "GENERATION_STYLE"])
    
    # Fusion and generation
    rpn_tokens.append("MULTIMODAL_FUSION")
    rpn_tokens.append("TEXT_3D_GENERATE")
    
    rpn_tokens.append("MULTIMODAL_3D_END")
    
    return " ".join(rpn_tokens)
