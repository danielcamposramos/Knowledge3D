#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inventory script to extract all kernel files (.cu and .ptx) and map them to opcodes.
"""

import json
import re
import os
import codecs


def find_kernel_files(root_dir):
    """
    Find all .cu and .ptx files in the kernels directory.
    """
    kernel_files = []
    
    kernels_dir = os.path.join(root_dir, "knowledge3d", "cranium", "kernels")
    
    if not os.path.exists(kernels_dir):
        print("Warning: Kernels directory not found: %s" % kernels_dir)
        return kernel_files
    
    # Walk through the kernels directory
    for root, dirs, files in os.walk(kernels_dir):
        for file in files:
            if file.endswith(('.cu', '.ptx')):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, root_dir)
                
                kernel_files.append({
                    'filename': file,
                    'full_path': full_path,
                    'relative_path': relative_path,
                    'type': 'cuda' if file.endswith('.cu') else 'ptx',
                    'size_bytes': os.path.getsize(full_path),
                    'last_modified': os.path.getmtime(full_path)
                })
    
    return kernel_files


def extract_kernel_metadata(kernel_file):
    """
    Extract metadata from a kernel file.
    """
    metadata = {
        'functions': [],
        'includes': [],
        'opcodes_referenced': [],
        'description': None
    }
    
    try:
        with codecs.open(kernel_file['full_path'], 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract function names
        function_pattern = r'__global__\s+void\s+(\w+)\s*\('
        for match in re.finditer(function_pattern, content):
            metadata['functions'].append(match.group(1))
        
        # Extract include files
        include_pattern = r'#include\s+["<]([^">]+)[">]'
        for match in re.finditer(include_pattern, content):
            metadata['includes'].append(match.group(1))
        
        # Look for opcode references
        opcode_pattern = r'OP_[A-Z_]+'
        opcodes_found = set()
        for match in re.finditer(opcode_pattern, content):
            opcodes_found.add(match.group(0))
        metadata['opcodes_referenced'] = list(opcodes_found)
        
        # Extract description from comments at the top
        lines = content.split('\n')
        description_lines = []
        in_header_comment = False
        
        for line in lines[:20]:  # Check first 20 lines
            line = line.strip()
            if line.startswith('/*') or line.startswith('//'):
                in_header_comment = True
                desc_line = line.replace('/*', '').replace('*/', '').replace('//', '').strip()
                if desc_line:
                    description_lines.append(desc_line)
            elif in_header_comment and not line:
                break
            elif in_header_comment and not (line.startswith('/*') or line.startswith('//')):
                break
        
        if description_lines:
            metadata['description'] = ' '.join(description_lines)
        
    except Exception as e:
        print("Error processing %s: %s" % (kernel_file['filename'], str(e)))
        metadata['error'] = str(e)
    
    return metadata


def map_kernels_to_opcodes(kernel_files, opcode_manifest):
    """
    Map kernel files to their corresponding opcodes based on references.
    """
    kernel_opcode_mapping = {}
    
    # Create a lookup of all known opcodes
    known_opcodes = set()
    if opcode_manifest and 'opcodes' in opcode_manifest:
        known_opcodes = set(opcode_manifest['opcodes'].keys())
    
    for kernel_file in kernel_files:
        metadata = extract_kernel_metadata(kernel_file)
        kernel_file.update(metadata)
        
        # Find valid opcode references
        valid_opcodes = []
        for opcode in metadata['opcodes_referenced']:
            if opcode in known_opcodes:
                valid_opcodes.append(opcode)
        
        kernel_opcode_mapping[kernel_file['relative_path']] = {
            'kernel_file': kernel_file,
            'referenced_opcodes': valid_opcodes,
            'orphaned_opcodes': [op for op in metadata['opcodes_referenced'] if op not in known_opcodes]
        }
    
    return kernel_opcode_mapping


def find_orphaned_kernels(kernel_opcode_mapping):
    """
    Find kernels that don't have corresponding opcode bindings.
    """
    orphaned = []
    
    for relative_path, mapping in kernel_opcode_mapping.items():
        kernel_file = mapping['kernel_file']
        
        # Check if this kernel has any opcode references
        if not mapping['referenced_opcodes']:
            # Check if it's a utility kernel (might not need opcode binding)
            is_utility = any(keyword in kernel_file['filename'].lower() 
                           for keyword in ['utility', 'helper', 'internal', 'warp', 'block'])
            
            orphaned.append({
                'kernel_file': relative_path,
                'reason': 'utility' if is_utility else 'no_opcode_references',
                'functions': kernel_file.get('functions', [])
            })
    
    return orphaned


def find_orphaned_opcodes(opcode_manifest, kernel_opcode_mapping):
    """
    Find opcodes that are referenced but don't have corresponding kernel implementations.
    """
    if not opcode_manifest or 'opcodes' not in opcode_manifest:
        return []
    
    # Get all opcodes referenced by kernels
    referenced_opcodes = set()
    for mapping in kernel_opcode_mapping.values():
        referenced_opcodes.update(mapping['referenced_opcodes'])
    
    # Find opcodes in manifest that aren't referenced
    orphaned = []
    for opcode_name, opcode_data in opcode_manifest['opcodes'].items():
        if opcode_name not in referenced_opcodes:
            orphaned.append({
                'opcode': opcode_name,
                'value': opcode_data['value'],
                'tier': opcode_data['tier'],
                'category': opcode_data['category']
            })
    
    return orphaned


def generate_kernel_manifest(root_dir, opcode_manifest_path=None):
    """
    Generate complete kernel manifest with all metadata and mappings.
    """
    # Load opcode manifest if provided
    opcode_manifest = None
    if opcode_manifest_path and os.path.exists(opcode_manifest_path):
        with codecs.open(opcode_manifest_path, 'r', encoding='utf-8') as f:
            opcode_manifest = json.load(f)
    
    # Find all kernel files
    kernel_files = find_kernel_files(root_dir)
    print("Found %d kernel files" % len(kernel_files))
    
    # Map kernels to opcodes
    kernel_opcode_mapping = map_kernels_to_opcodes(kernel_files, opcode_manifest)
    
    # Find orphaned kernels and opcodes
    orphaned_kernels = find_orphaned_kernels(kernel_opcode_mapping)
    orphaned_opcodes = find_orphaned_opcodes(opcode_manifest, kernel_opcode_mapping)
    
    # Generate statistics
    stats = {
        'total_kernels': len(kernel_files),
        'cuda_kernels': len([k for k in kernel_files if k['type'] == 'cuda']),
        'ptx_kernels': len([k for k in kernel_files if k['type'] == 'ptx']),
        'kernels_with_opcodes': len([m for m in kernel_opcode_mapping.values() if m['referenced_opcodes']]),
        'orphaned_kernels': len(orphaned_kernels),
        'orphaned_opcodes': len(orphaned_opcodes)
    }
    
    manifest = {
        'metadata': {
            'total_kernels': len(kernel_files),
            'generated_by': 'scripts/inventory_kernels.py',
            'root_directory': root_dir,
            'schema_version': '1.0'
        },
        'statistics': stats,
        'kernel_files': kernel_files,
        'kernel_opcode_mapping': kernel_opcode_mapping,
        'orphaned_kernels': orphaned_kernels,
        'orphaned_opcodes': orphaned_opcodes
    }
    
    return manifest


def save_manifest(manifest, output_path):
    """
    Save the kernel manifest to a JSON file.
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
    print("Inventorying kernel files...")
    
    root_dir = "."
    opcode_manifest_path = "docs/opcodes_manifest.json"
    
    manifest = generate_kernel_manifest(root_dir, opcode_manifest_path)
    
    output_path = "docs/kernels_manifest.json"
    save_manifest(manifest, output_path)
    
    print("Processed %d kernel files" % manifest['metadata']['total_kernels'])
    print("Statistics:")
    print("   CUDA kernels: %d" % manifest['statistics']['cuda_kernels'])
    print("   PTX kernels: %d" % manifest['statistics']['ptx_kernels'])
    print("   Kernels with opcode mappings: %d" % manifest['statistics']['kernels_with_opcodes'])
    print("   Orphaned kernels: %d" % manifest['statistics']['orphaned_kernels'])
    print("   Orphaned opcodes: %d" % manifest['statistics']['orphaned_opcodes'])
    print("Saved manifest to: %s" % output_path)


if __name__ == "__main__":
    main()