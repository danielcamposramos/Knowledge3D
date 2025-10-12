import os
import logging
import json
import time
import ctypes
from .sovereign.loader import memcpy_dtoh
from .galaxy_buffer import GalaxyEmbedding, GALAXY_EMBEDDING_SIZE

logger = logging.getLogger(__name__)

# Lazy matplotlib import
_MPL = None
if os.getenv("K3D_ENABLE_THINKING_TAG_VISUALIZATION", "0") == "1":
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _MPL = plt
    except ImportError:
        logger.warning("Matplotlib unavailable, using JSON export only")

class GalaxyVisualizer:
    def __init__(self, resonance_field):
        self.resonance_field = resonance_field
        self.output_dir = os.getenv("K3D_VISUALIZATION_OUTPUT_DIR", "./visualization_output")
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_region_gpu_data(self, layer_id):
        """Sovereign GPU readback (Kimi's fix)"""
        count = self.resonance_field.region_size(layer_id)
        gpu_ptr = self.resonance_field.region_ptr(layer_id)
        cpu_arr = (GalaxyEmbedding * count)()
        memcpy_dtoh(ctypes.byref(cpu_arr), gpu_ptr, count * GALAXY_EMBEDDING_SIZE)
        return cpu_arr

    def visualize_weight_regions(self, layer_id, output_filename=None):
        try:
            weights_data = self.extract_region_gpu_data(layer_id)
            
            if output_filename is None:
                output_filename = f"weight_region_layer_{layer_id}.json"
            output_path = os.path.join(self.output_dir, output_filename)

            export_data = []
            for i, emb in enumerate(weights_data):
                export_data.append({
                    "id": i,
                    "x": float(emb.vector[0]),
                    "y": float(emb.vector[1]),
                    "z": float(emb.vector[2]),
                    "value": float(emb.vector[3]),
                    "access_freq": int(emb.access_freq),
                    "clock": int(emb.galaxy_clock)
                })

            with open(output_path, 'w') as f:
                json.dump(export_data, f)
            logger.info(f"Exported visualization to {output_path}")

        except Exception as e:
            logger.error(f"Visualization failed: {e}")

    def visualize_inference_flow(self, input_embedding, tags):
        try:
            output_filename = f"inference_trace_{int(time.time())}.json"
            output_path = os.path.join(self.output_dir, output_filename)

            trace_data = {
                "input_embedding": input_embedding.tolist(),
                "generated_tags": tags,
                "timestamp": time.time()
            }

            with open(output_path, 'w') as f:
                json.dump(trace_data, f)
            logger.info(f"Exported inference trace to {output_path}")

        except Exception as e:
            logger.error(f"Inference trace failed: {e}")
