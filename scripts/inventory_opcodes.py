#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inventory script to extract all OP codes from rpn_opcodes.py and create a canonical manifest.
"""

import json
import re
import os
import codecs


def extract_opcodes_from_file(filepath):
    """
    Extract all OP codes and their metadata from rpn_opcodes.py.
    """
    with codecs.open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all OP_* constant definitions
    opcode_pattern = r'^(OP_[A-Z_]+)\s*=\s*(0x[0-9A-F]+)'
    opcodes = {}
    
    for match in re.finditer(opcode_pattern, content, re.MULTILINE):
        opcode_name = match.group(1)
        opcode_value = int(match.group(2), 16)
        
        # Extract surrounding context for metadata
        lines = content.split('\n')
        line_num = content[:match.start()].count('\n')
        
        # Get context (5 lines before and after)
        start_line = max(0, line_num - 5)
        end_line = min(len(lines), line_num + 6)
        context = '\n'.join(lines[start_line:end_line])
        
        # Determine tier based on opcode value
        tier = determine_tier(opcode_value)
        
        opcodes[opcode_name] = {
            'value': opcode_value,
            'hex': hex(opcode_value),
            'tier': tier,
            'line_number': line_num + 1,
            'context': context.strip(),
            'file': os.path.relpath(filepath)
        }
    
    return opcodes


def determine_tier(opcode_value):
    """
    Determine the tier/domain based on opcode value.
    """
    if 0x0000 <= opcode_value <= 0x00FF:
        return "tier_0_math"
    elif 0x0100 <= opcode_value <= 0x01FF:
        return "tier_1_cooperative"
    elif 0x0200 <= opcode_value <= 0x02FF:
        return "tier_2_physics"
    elif 0x0300 <= opcode_value <= 0x03FF:
        return "tier_3_cas"
    elif 0x0400 <= opcode_value <= 0x04FF:
        return "tier_4_drawing"
    elif 0x0500 <= opcode_value <= 0x05FF:
        return "tier_5_vector_db"
    elif 0x0600 <= opcode_value <= 0x06FF:
        return "tier_6_filter"
    elif 0x0700 <= opcode_value <= 0x07FF:
        return "tier_7_extension"
    elif 0xFF00 <= opcode_value <= 0xFFFF:
        return "system_control"
    else:
        return "unknown"


def categorize_opcode(opcode_name, opcode_value):
    """
    Categorize opcode by functional domain based on name patterns.
    """
    name_lower = opcode_name.lower()
    
    # Math operations
    if any(op in name_lower for op in ['add', 'sub', 'mul', 'div', 'pow', 'sqrt', 'exp', 'log', 
                                       'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 
                                       'cosh', 'tanh', 'abs', 'ceil', 'floor', 'round',
                                       'max', 'min', 'eq', 'gt', 'lt', 'gte', 'lte',
                                       'dup', 'swap', 'drop', 'mod', 'log2', 'log10']):
        return "math"
    
    # Stack operations
    elif any(op in name_lower for op in ['dup', 'swap', 'drop']):
        return "stack"
    
    # Bitwise operations
    elif any(op in name_lower for op in ['and', 'or', 'xor', 'not', 'neg']):
        return "bitwise"
    
    # Complex numbers
    elif 'complex' in name_lower:
        return "complex"
    
    # Special operations
    elif any(op in name_lower for op in ['sparse', 'smav', 'entropy', 'sigmoid']):
        return "special"
    
    # TRM integration
    elif 'trm' in name_lower:
        return "trm_integration"
    
    # Checkpoint operations
    elif any(op in name_lower for op in ['checkpoint', 'rollback', 'verify']):
        return "checkpoint"
    
    # Programmability
    elif any(op in name_lower for op in ['branch', 'loop', 'next', 'store', 'recall', 
                                        'limit', 'series', 'gradient', 'integrate',
                                        'divergence', 'curl', 'laplacian']):
        return "programmability"
    
    # Clustering
    elif any(op in name_lower for op in ['vec', 'normalize', 'argmax', 'blend', 'cosine', 
                                        'cluster', 'set', 'dot', 'cross', 'outer',
                                        'eigen', 'svd', 'qr', 'cholesky', 'lu',
                                        'quantum']):
        return "clustering"
    
    # GPU Galaxy access
    elif any(op in name_lower for op in ['galaxy', 'load_galaxy', 'galaxy_similarity', 
                                        'galaxy_scan']):
        return "galaxy_access"
    
    # Variable references
    elif opcode_name in ['OP_VAR_X', 'OP_VAR_Y', 'OP_VAR_Z', 'OP_VAR_W', 'OP_CONST']:
        return "variable_reference"
    
    # Grammar evolution
    elif 'grammar' in name_lower:
        return "grammar_evolution"
    
    # Temporal reasoning
    elif 'temporal' in name_lower:
        return "temporal_reasoning"
    
    # Reasoning paradigm
    elif any(op in name_lower for op in ['abduce', 'explain', 'suspect', 'icheck', 
                                        'abdres', 'abdneg', 'ebbelief', 'biduce',
                                        'frame', 'euler', 'dl', 'blocking', 
                                        'ctx_switch', 'alpchain', 'tunify', 
                                        'tresolve', 'torder', 'tsubsum', 
                                        'tsuperpos', 'trewrite', 'tsplit',
                                        'tclose', 'texpand', 'tbcp', 'tlearnt',
                                        'rete', 'agenda', 'halt']):
        return "reasoning_paradigm"
    
    # CBR extension
    elif 'case' in name_lower:
        return "cbr_extension"
    
    # Trit operations
    elif opcode_name.startswith('OP_T') and len(opcode_name) <= 6:
        return "trit_operations"
    
    # Entity behavior
    elif opcode_name.startswith('OP_BH_'):
        return "entity_behavior"
    
    # CAS operations
    elif opcode_name.startswith('OP_POLY_') or opcode_name in [
        'OP_SIMPLIFY', 'OP_SUBSTITUTE', 'OP_COLLECT', 'OP_RATIONALIZE',
        'OP_TRIG_SIMPLIFY', 'OP_LOG_SIMPLIFY', 'OP_SOLVE_LINEAR',
        'OP_SOLVE_QUADRATIC', 'OP_LINSOLVE', 'OP_PATTERN_MATCH',
        'OP_RULE_APPLY', 'OP_COEFF_EXTRACT', 'OP_CAS_PUSH_SYM',
        'OP_CAS_PUSH_CONST', 'OP_CAS_BUILD', 'OP_CAS_EVAL'
    ]:
        return "cas"
    
    # SAS extension
    elif opcode_name in [
        'OP_CANONICALIZE', 'OP_CAS_HASH', 'OP_SEMANTIC_RESOLVE',
        'OP_RULE_SELECT', 'OP_CONTEXTUAL_REWRITE', 'OP_SEMANTIC_EQUIV'
    ]:
        return "sas"
    
    # Procedural drawing
    elif opcode_name.startswith('OP_DRAW_'):
        return "procedural_drawing"
    
    # Advanced drawing primitives
    elif opcode_name in [
        'OP_BEZIER_EVAL', 'OP_SHAPE_UNION', 'OP_SHAPE_INTERSECT', 
        'OP_SHAPE_SUBTRACT', 'OP_DRAW_REL_LINE', 'OP_DRAW_FIELD_COEF',
        'OP_DRAW_DOT_EMIT'
    ]:
        return "advanced_drawing"
    
    # VectorDotMap Codec
    elif opcode_name in [
        'OP_DRAW_VECTORDOTMAP_ENCODE', 'OP_DRAW_VECTORDOTMAP_DECODE'
    ]:
        return "vectordotmap_codec"
    
    # Lighting and Layer Ops
    elif opcode_name in [
        'OP_DRAW_LAYER_NEW', 'OP_LAYER_BLEND', 'OP_BLEND_MULTIPLY',
        'OP_BLEND_SCREEN', 'OP_BLEND_OVERLAY', 'OP_ATMOSPHERE_FOG',
        'OP_VIGNETTE'
    ]:
        return "lighting_layer"
    
    # 3D Technique Suite
    elif opcode_name in [
        'OP_NURBS_EVAL', 'OP_MARCHING_CUBES', 'OP_LSYSTEM_GENERATE',
        'OP_PARAMETRIC_SURFACE', 'OP_CSG_UNION_3D', 'OP_CSG_INTERSECT_3D',
        'OP_CSG_SUBTRACT_3D', 'OP_CROSS_MODAL_LINK', 'OP_PROCEDURAL_TEXTURE'
    ]:
        return "3d_technique"
    
    # Physics meta-dispatch
    elif opcode_name.startswith('OP_PH_'):
        return "physics_meta_dispatch"
    
    # Procedural texture synthesis
    elif opcode_name.startswith('OP_TEX_'):
        return "procedural_texture"
    
    # Drawing Galaxy Layers
    elif opcode_name in [
        'OP_GRADIENT_LINEAR', 'OP_GRADIENT_RADIAL', 'OP_GRADIENT_CONIC',
        'OP_GRADIENT_STOP', 'OP_FILTER_BLUR', 'OP_FILTER_SHARPEN',
        'OP_FILTER_EDGE', 'OP_FILTER_INVERT', 'OP_LIGHT_AMBIENT',
        'OP_LIGHT_DIRECTIONAL', 'OP_LAYER_PUSH', 'OP_LAYER_POP',
        'OP_BLEND_MODE'
    ]:
        return "drawing_galaxy"
    
    else:
        return "other"


def generate_opcode_manifest():
    """
    Generate complete opcode manifest with all metadata.
    """
    opcodes_file = "knowledge3d/cranium/ptx_runtime/rpn_opcodes.py"
    
    if not os.path.exists(opcodes_file):
        raise FileNotFoundError("Could not find %s" % opcodes_file)
    
    opcodes = extract_opcodes_from_file(opcodes_file)
    
    # Enhance with categorization and additional metadata
    enhanced_opcodes = {}
    for name, data in opcodes.items():
        category = categorize_opcode(name, data['value'])
        
        # Create enhanced entry without using ** unpacking
        enhanced_opcodes[name] = data.copy()
        enhanced_opcodes[name]['category'] = category
        enhanced_opcodes[name]['sovereignty_compliant'] = check_sovereignty_compliance(name, data['value'])
        enhanced_opcodes[name]['kernel_mapping'] = find_kernel_mapping(name, data['value'])
    
    # Generate statistics
    stats = generate_statistics(enhanced_opcodes)
    
    manifest = {
        'metadata': {
            'total_opcodes': len(enhanced_opcodes),
            'generated_by': 'scripts/inventory_opcodes.py',
            'source_file': opcodes_file,
            'schema_version': '1.0'
        },
        'statistics': stats,
        'opcodes': enhanced_opcodes
    }
    
    return manifest


def check_sovereignty_compliance(opcode_name, opcode_value):
    """
    Check sovereignty compliance for an opcode.
    """
    # This would be expanded with actual checks
    return {
        'gpu_first': True,  # Assume GPU-first for now
        'determinism': 'unknown',  # strict, weak, probabilistic
        'precision': 'unknown',    # FP16, FP32, FP64, mixed
        'memory_safety': 'unknown', # shared memory requirements
        'provenance': 'unknown'    # algorithm origin, certifications
    }


def find_kernel_mapping(opcode_name, opcode_value):
    """
    Find corresponding kernel file for an opcode.
    To be enhanced with actual kernel discovery.
    """
    # Placeholder - will be enhanced in kernel inventory script
    return None


def generate_statistics(opcodes):
    """
    Generate statistics about the opcode collection.
    """
    tiers = {}
    categories = {}
    
    for opcode_data in opcodes.values():
        tier = opcode_data['tier']
        category = opcode_data['category']
        
        tiers[tier] = tiers.get(tier, 0) + 1
        categories[category] = categories.get(category, 0) + 1
    
    return {
        'by_tier': tiers,
        'by_category': categories,
        'total_count': len(opcodes)
    }


def save_manifest(manifest, output_path):
    """
    Save the opcode manifest to a JSON file.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with codecs.open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def main():
    """
    Main execution function.
    """
    print("Extracting OP codes from rpn_opcodes.py...")
    
    manifest = generate_opcode_manifest()
    
    output_path = "docs/opcodes_manifest.json"
    save_manifest(manifest, output_path)
    
    print("Extracted %d opcodes" % manifest['metadata']['total_opcodes'])
    print("Statistics:")
    print("   By tier: %s" % manifest['statistics']['by_tier'])
    print("   By category: %s" % manifest['statistics']['by_category'])
    print("Saved manifest to: %s" % output_path)


if __name__ == "__main__":
    main()