#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to populate Qdrant with opcode and kernel data from the manifest files.
"""

import json
import os
import sys
import codecs
from datetime import datetime


def load_manifests(opcodes_path, kernels_path):
    """
    Load the opcode and kernel manifests.
    """
    opcodes_manifest = None
    kernels_manifest = None
    
    if os.path.exists(opcodes_path):
        with codecs.open(opcodes_path, 'r', encoding='utf-8') as f:
            opcodes_manifest = json.load(f)
        print("Loaded %d opcodes from manifest" % len(opcodes_manifest.get('opcodes', {})))
    else:
        print("Warning: Opcodes manifest not found at %s" % opcodes_path)
    
    if os.path.exists(kernels_path):
        with codecs.open(kernels_path, 'r', encoding='utf-8') as f:
            kernels_manifest = json.load(f)
        print("Loaded %d kernel files from manifest" % len(kernels_manifest.get('kernel_files', [])))
    else:
        print("Warning: Kernels manifest not found at %s" % kernels_path)
    
    return opcodes_manifest, kernels_manifest


def create_qdrant_points_from_opcodes(opcodes_manifest):
    """
    Create Qdrant points from opcode manifest data.
    """
    points = []
    
    if not opcodes_manifest or 'opcodes' not in opcodes_manifest:
        return points
    
    for opcode_name, opcode_data in opcodes_manifest['opcodes'].items():
        # Create a text description for embedding
        description = create_opcode_description(opcode_name, opcode_data)
        
        point = {
            'id': hash(opcode_name) % (10**8),  # Simple ID generation
            'payload': {
                'type': 'opcode',
                'name': opcode_name,
                'value': opcode_data['value'],
                'hex': opcode_data['hex'],
                'tier': opcode_data['tier'],
                'category': opcode_data['category'],
                'line_number': opcode_data['line_number'],
                'file': opcode_data['file'],
                'sovereignty_compliant': opcode_data.get('sovereignty_compliant', {}),
                'kernel_mapping': opcode_data.get('kernel_mapping'),
                'description': description,
                'created_at': datetime.now().isoformat(),
                'source': 'k3d_opcodes_manifest'
            },
            'vector': [0.0] * 384  # Placeholder for embeddings
        }
        
        points.append(point)
    
    return points


def create_qdrant_points_from_kernels(kernels_manifest):
    """
    Create Qdrant points from kernel manifest data.
    """
    points = []
    
    if not kernels_manifest or 'kernel_files' not in kernels_manifest:
        return points
    
    for kernel_file in kernels_manifest['kernel_files']:
        # Create a text description for embedding
        description = create_kernel_description(kernel_file)
        
        point = {
            'id': hash(kernel_file['relative_path']) % (10**8),  # Simple ID generation
            'payload': {
                'type': 'kernel',
                'filename': kernel_file['filename'],
                'relative_path': kernel_file['relative_path'],
                'type': kernel_file['type'],
                'size_bytes': kernel_file['size_bytes'],
                'functions': kernel_file.get('functions', []),
                'includes': kernel_file.get('includes', []),
                'opcodes_referenced': kernel_file.get('opcodes_referenced', []),
                'description': kernel_file.get('description'),
                'created_at': datetime.now().isoformat(),
                'source': 'k3d_kernels_manifest'
            },
            'vector': [0.0] * 384  # Placeholder for embeddings
        }
        
        points.append(point)
    
    return points


def create_opcode_description(opcode_name, opcode_data):
    """
    Create a text description for an opcode for embedding.
    """
    parts = []
    parts.append("Opcode: %s" % opcode_name)
    parts.append("Value: %s" % opcode_data['hex'])
    parts.append("Tier: %s" % opcode_data['tier'])
    parts.append("Category: %s" % opcode_data['category'])
    
    # Add context information
    if opcode_data.get('context'):
        # Extract meaningful parts from context
        context_lines = opcode_data['context'].split('\n')
        for line in context_lines:
            line = line.strip()
            if line and not line.startswith('OP_') and len(line) < 200:
                parts.append("Context: %s" % line)
                break
    
    return ' '.join(parts)


def create_kernel_description(kernel_file):
    """
    Create a text description for a kernel file for embedding.
    """
    parts = []
    parts.append("Kernel: %s" % kernel_file['filename'])
    parts.append("Type: %s" % kernel_file['type'])
    
    if kernel_file.get('description'):
        parts.append("Description: %s" % kernel_file['description'])
    
    if kernel_file.get('functions'):
        parts.append("Functions: %s" % ', '.join(kernel_file['functions'][:3]))  # Limit to first 3
    
    if kernel_file.get('opcodes_referenced'):
        parts.append("Opcodes: %s" % ', '.join(kernel_file['opcodes_referenced'][:5]))  # Limit to first 5
    
    return ' '.join(parts)


def generate_qdrant_batch_file(opcodes_manifest, kernels_manifest, output_path):
    """
    Generate a JSON file with Qdrant points that can be uploaded.
    """
    all_points = []
    
    # Add opcode points
    opcode_points = create_qdrant_points_from_opcodes(opcodes_manifest)
    all_points.extend(opcode_points)
    
    # Add kernel points
    kernel_points = create_qdrant_points_from_kernels(kernels_manifest)
    all_points.extend(kernel_points)
    
    # Create the batch file structure
    batch_data = {
        'metadata': {
            'total_points': len(all_points),
            'opcode_points': len(opcode_points),
            'kernel_points': len(kernel_points),
            'collection_name': 'k3d_opcodes_kernels_canonical',
            'created_at': datetime.now().isoformat(),
            'schema_version': '1.0'
        },
        'points': all_points
    }
    
    # Save the batch file
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with codecs.open(output_path, 'w', encoding='utf-8') as f:
        json.dump(batch_data, f, indent=2, sort_keys=True)
    
    return batch_data


def generate_summary_report(opcodes_manifest, kernels_manifest, batch_data):
    """
    Generate a summary report of the Qdrant population.
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_opcodes': len(opcodes_manifest.get('opcodes', {}) if opcodes_manifest else {}),
            'total_kernels': len(kernels_manifest.get('kernel_files', []) if kernels_manifest else []),
            'total_qdrant_points': batch_data['metadata']['total_points'],
            'opcode_points': batch_data['metadata']['opcode_points'],
            'kernel_points': batch_data['metadata']['kernel_points']
        },
        'opcode_statistics': opcodes_manifest.get('statistics', {}) if opcodes_manifest else {},
        'kernel_statistics': kernels_manifest.get('statistics', {}) if kernels_manifest else {},
        'orphaned_analysis': {
            'orphaned_kernels': len(kernels_manifest.get('orphaned_kernels', []) if kernels_manifest else []),
            'orphaned_opcodes': len(kernels_manifest.get('orphaned_opcodes', []) if kernels_manifest else [])
        }
    }
    
    return report


def main():
    """
    Main execution function.
    """
    print("Populating Qdrant with opcode and kernel data...")
    
    # Load manifests
    opcodes_manifest_path = "docs/opcodes_manifest.json"
    kernels_manifest_path = "docs/kernels_manifest.json"
    
    opcodes_manifest, kernels_manifest = load_manifests(opcodes_manifest_path, kernels_manifest_path)
    
    if not opcodes_manifest and not kernels_manifest:
        print("Error: No manifests found. Please run inventory scripts first.")
        sys.exit(1)
    
    # Generate Qdrant batch file
    output_path = "docs/qdrant_batch.json"
    batch_data = generate_qdrant_batch_file(opcodes_manifest, kernels_manifest, output_path)
    
    # Generate summary report
    report = generate_summary_report(opcodes_manifest, kernels_manifest, batch_data)
    report_path = "docs/qdrant_population_report.json"
    
    with codecs.open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, sort_keys=True)
    
    print("Qdrant population completed:")
    print("  Total points: %d" % batch_data['metadata']['total_points'])
    print("  Opcode points: %d" % batch_data['metadata']['opcode_points'])
    print("  Kernel points: %d" % batch_data['metadata']['kernel_points'])
    print("  Batch file: %s" % output_path)
    print("  Report: %s" % report_path)
    
    # Print orphaned analysis
    if kernels_manifest:
        print("\nOrphaned Analysis:")
        print("  Orphaned kernels: %d" % len(kernels_manifest.get('orphaned_kernels', [])))
        print("  Orphaned opcodes: %d" % len(kernels_manifest.get('orphaned_opcodes', [])))


if __name__ == "__main__":
    main()