#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script to verify the integrity of the opcode-kernel registry.
"""

import json
import os
import codecs
import sys
from datetime import datetime


def validate_opcodes(opcodes_manifest):
    """
    Validate opcode manifest integrity.
    """
    issues = []
    
    if not opcodes_manifest or 'opcodes' not in opcodes_manifest:
        issues.append("Missing or invalid opcodes manifest")
        return issues
    
    opcodes = opcodes_manifest['opcodes']
    
    # Check for duplicate opcode values
    value_counts = {}
    for opcode_name, opcode_data in opcodes.items():
        value = opcode_data['value']
        if value in value_counts:
            value_counts[value].append(opcode_name)
        else:
            value_counts[value] = [opcode_name]
    
    for value, names in value_counts.items():
        if len(names) > 1:
            issues.append("Duplicate opcode value 0x%04X: %s" % (value, ', '.join(names)))
    
    # Check for required fields
    required_fields = ['value', 'hex', 'tier', 'category', 'line_number']
    for opcode_name, opcode_data in opcodes.items():
        for field in required_fields:
            if field not in opcode_data:
                issues.append("Missing field '%s' in opcode %s" % (field, opcode_name))
    
    # Check hex consistency
    for opcode_name, opcode_data in opcodes.items():
        expected_hex = hex(opcode_data['value'])
        if opcode_data['hex'] != expected_hex:
            issues.append("Hex mismatch for %s: expected %s, got %s" % 
                         (opcode_name, expected_hex, opcode_data['hex']))
    
    # Check tier ranges
    tier_ranges = {
        'tier_0_math': (0x0000, 0x00FF),
        'tier_1_cooperative': (0x0100, 0x01FF),
        'tier_2_physics': (0x0200, 0x02FF),
        'tier_3_cas': (0x0300, 0x03FF),
        'tier_4_drawing': (0x0400, 0x04FF),
        'tier_5_vector_db': (0x0500, 0x05FF),
        'tier_6_filter': (0x0600, 0x06FF),
        'tier_7_extension': (0x0700, 0x07FF),
        'system_control': (0xFF00, 0xFFFF)
    }
    
    for opcode_name, opcode_data in opcodes.items():
        tier = opcode_data['tier']
        value = opcode_data['value']
        
        if tier in tier_ranges:
            min_val, max_val = tier_ranges[tier]
            if not (min_val <= value <= max_val):
                issues.append("Opcode %s value 0x%04X outside tier %s range [0x%04X-0x%04X]" % 
                             (opcode_name, value, tier, min_val, max_val))
    
    return issues


def validate_kernels(kernels_manifest):
    """
    Validate kernel manifest integrity.
    """
    issues = []
    
    if not kernels_manifest or 'kernel_files' not in kernels_manifest:
        issues.append("Missing or invalid kernels manifest")
        return issues
    
    kernel_files = kernels_manifest['kernel_files']
    
    # Check for duplicate file paths
    path_counts = {}
    for kernel_file in kernel_files:
        path = kernel_file['relative_path']
        if path in path_counts:
            path_counts[path].append(kernel_file['filename'])
        else:
            path_counts[path] = [kernel_file['filename']]
    
    for path, filenames in path_counts.items():
        if len(filenames) > 1:
            issues.append("Duplicate kernel path %s: %s" % (path, ', '.join(filenames)))
    
    # Check for required fields
    required_fields = ['filename', 'relative_path', 'type', 'size_bytes']
    for i, kernel_file in enumerate(kernel_files):
        for field in required_fields:
            if field not in kernel_file:
                issues.append("Missing field '%s' in kernel file %d (%s)" % 
                             (field, i, kernel_file.get('filename', 'unknown')))
    
    # Check file types
    valid_types = ['cuda', 'ptx']
    for kernel_file in kernel_files:
        if kernel_file['type'] not in valid_types:
            issues.append("Invalid kernel type '%s' for %s" % 
                         (kernel_file['type'], kernel_file['filename']))
    
    # Check physical file existence
    for kernel_file in kernel_files:
        full_path = kernel_file['full_path']
        if not os.path.exists(full_path):
            issues.append("Kernel file not found: %s" % full_path)
    
    return issues


def validate_mappings(opcodes_manifest, kernels_manifest):
    """
    Validate opcode-kernel mappings.
    """
    issues = []
    
    if not opcodes_manifest or not kernels_manifest:
        issues.append("Missing manifests for mapping validation")
        return issues
    
    # Get all known opcodes
    known_opcodes = set(opcodes_manifest.get('opcodes', {}).keys())
    
    # Get kernel-opcode mapping
    kernel_mapping = kernels_manifest.get('kernel_opcode_mapping', {})
    
    # Check for opcodes referenced by kernels but not in manifest
    referenced_opcodes = set()
    for mapping in kernel_mapping.values():
        referenced_opcodes.update(mapping.get('referenced_opcodes', []))
    
    missing_in_manifest = referenced_opcodes - known_opcodes
    if missing_in_manifest:
        issues.append("Opcodes referenced by kernels but missing from manifest: %s" % 
                     ', '.join(sorted(missing_in_manifest)))
    
    # Check for orphaned opcodes (in manifest but not referenced)
    orphaned_opcodes = kernels_manifest.get('orphaned_opcodes', [])
    if orphaned_opcodes:
        issues.append("%d orphaned opcodes found (in manifest but not referenced by kernels)" % 
                     len(orphaned_opcodes))
        # Show first 5 as examples
        if len(orphaned_opcodes) > 5:
            sample = orphaned_opcodes[:5]
            issues.append("Sample orphaned opcodes: %s" % ', '.join(
                "%s (0x%04X)" % (op['opcode'], op['value']) for op in sample))
    
    # Check for orphaned kernels
    orphaned_kernels = kernels_manifest.get('orphaned_kernels', [])
    if orphaned_kernels:
        issues.append("%d orphaned kernels found (exist but lack opcode bindings)" % 
                     len(orphaned_kernels))
        # Show first 5 as examples
        if len(orphaned_kernels) > 5:
            sample = orphaned_kernels[:5]
            issues.append("Sample orphaned kernels: %s" % ', '.join(
                op['kernel_file'] for op in sample))
    
    return issues


def validate_sovereignty_compliance(opcodes_manifest, kernels_manifest):
    """
    Validate sovereignty compliance for opcodes and kernels.
    """
    issues = []
    
    # Check opcodes sovereignty compliance
    if opcodes_manifest and 'opcodes' in opcodes_manifest:
        non_compliant_opcodes = []
        for opcode_name, opcode_data in opcodes_manifest['opcodes'].items():
            compliance = opcode_data.get('sovereignty_compliant', {})
            if not compliance or compliance.get('gpu_first') is not True:
                non_compliant_opcodes.append(opcode_name)
        
        if non_compliant_opcodes:
            issues.append("%d opcodes lack sovereignty compliance data" % len(non_compliant_opcodes))
    
    # Check for CPU fallbacks in kernel implementations
    if kernels_manifest and 'kernel_files' in kernels_manifest:
        cpu_fallback_kernels = []
        for kernel_file in kernels_manifest['kernel_files']:
            # This is a simplified check - in reality, you'd analyze the kernel code
            description = kernel_file.get('description', '') or ''
            if 'cpu_fallback' in description.lower():
                cpu_fallback_kernels.append(kernel_file['filename'])
        
        if cpu_fallback_kernels:
            issues.append("Kernels with potential CPU fallbacks: %s" % ', '.join(cpu_fallback_kernels))
    
    return issues


def generate_validation_report(opcodes_manifest, kernels_manifest):
    """
    Generate a comprehensive validation report.
    """
    timestamp = datetime.now().isoformat()
    
    # Run all validations
    opcode_issues = validate_opcodes(opcodes_manifest)
    kernel_issues = validate_kernels(kernels_manifest)
    mapping_issues = validate_mappings(opcodes_manifest, kernels_manifest)
    sovereignty_issues = validate_sovereignty_compliance(opcodes_manifest, kernels_manifest)
    
    # Calculate statistics
    total_opcodes = len(opcodes_manifest.get('opcodes', {}) if opcodes_manifest else {})
    total_kernels = len(kernels_manifest.get('kernel_files', []) if kernels_manifest else {})
    
    report = {
        'timestamp': timestamp,
        'summary': {
            'total_opcodes': total_opcodes,
            'total_kernels': total_kernels,
            'total_issues': len(opcode_issues) + len(kernel_issues) + len(mapping_issues) + len(sovereignty_issues),
            'opcode_issues': len(opcode_issues),
            'kernel_issues': len(kernel_issues),
            'mapping_issues': len(mapping_issues),
            'sovereignty_issues': len(sovereignty_issues),
            'status': 'PASS' if len(opcode_issues) + len(kernel_issues) + len(mapping_issues) + len(sovereignty_issues) == 0 else 'FAIL'
        },
        'issues': {
            'opcode_issues': opcode_issues,
            'kernel_issues': kernel_issues,
            'mapping_issues': mapping_issues,
            'sovereignty_issues': sovereignty_issues
        }
    }
    
    return report


def main():
    """
    Main execution function.
    """
    print("Validating Knowledge3D opcode and kernel registry...")
    
    # Load manifests
    opcodes_manifest_path = "docs/opcodes_manifest.json"
    kernels_manifest_path = "docs/kernels_manifest.json"
    
    opcodes_manifest = None
    kernels_manifest = None
    
    if os.path.exists(opcodes_manifest_path):
        with codecs.open(opcodes_manifest_path, 'r', encoding='utf-8') as f:
            opcodes_manifest = json.load(f)
        print("Loaded opcodes manifest with %d entries" % len(opcodes_manifest.get('opcodes', {})))
    
    if os.path.exists(kernels_manifest_path):
        with codecs.open(kernels_manifest_path, 'r', encoding='utf-8') as f:
            kernels_manifest = json.load(f)
        print("Loaded kernels manifest with %d entries" % len(kernels_manifest.get('kernel_files', [])))
    
    # Generate validation report
    report = generate_validation_report(opcodes_manifest, kernels_manifest)
    
    # Save report
    report_path = "docs/validation_report.json"
    with codecs.open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, sort_keys=True)
    
    # Print summary
    print("\nValidation Summary:")
    print("  Status: %s" % report['summary']['status'])
    print("  Total Issues: %d" % report['summary']['total_issues'])
    print("  Opcode Issues: %d" % report['summary']['opcode_issues'])
    print("  Kernel Issues: %d" % report['summary']['kernel_issues'])
    print("  Mapping Issues: %d" % report['summary']['mapping_issues'])
    print("  Sovereignty Issues: %d" % report['summary']['sovereignty_issues'])
    print("  Report saved to: %s" % report_path)
    
    # Print detailed issues if any
    if report['summary']['total_issues'] > 0:
        print("\nDetailed Issues:")
        for category, issues in report['issues'].items():
            if issues:
                print("  %s:" % category.replace('_', ' ').title())
                for issue in issues[:5]:  # Show first 5 issues
                    print("    - %s" % issue)
                if len(issues) > 5:
                    print("    ... and %d more issues" % (len(issues) - 5))
    
    # Exit with appropriate code
    sys.exit(0 if report['summary']['status'] == 'PASS' else 1)


if __name__ == "__main__":
    main()