#!/usr/bin/env python3
"""Test script for standalone DeepSeek OCR analysis."""

import sys
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.append('.')

# Import the standalone analyzer directly
from knowledge3d.ingestion.deepseek_ocr_standalone import StandaloneDeepSeekAnalyzer


def main():
    """Run standalone DeepSeek OCR analysis test."""
    
    print("=" * 60)
    print("STANDALONE DEESEEK OCR ANALYSIS TEST")
    print("=" * 60)
    
    try:
        # Initialize analyzer
        print("Initializing StandaloneDeepSeekAnalyzer...")
        analyzer = StandaloneDeepSeekAnalyzer(enable_gpu=False)
        print("✅ Analyzer initialized successfully!")
        
        # Test with sample texts
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Mathematical equations: E = mc² and a² + b² = c²",
            "Historical reference: In 1492, Columbus sailed the ocean blue.",
            "Scientific concept: Photosynthesis converts light energy into chemical energy.",
            "Technical specification: The RTX 3060 has 12GB of VRAM and 3584 CUDA cores."
        ]
        
        print(f"\nAnalyzing {len(test_texts)} sample texts...")
        
        # Run analysis
        results = analyzer.analyze_deepseek_capabilities(
            test_texts,
            generate_test_images=True
        )
        
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        
        # Display key metrics
        compression = results.get('compression_metrics', {})
        embedding = results.get('embedding_analysis', {})
        performance = results.get('performance_summary', {})
        
        print(f"\n📊 COMPRESSION METRICS:")
        print(f"  Average compression ratio: {compression.get('average_compression_ratio', 0):.1f}x")
        print(f"  Average fidelity: {compression.get('average_fidelity', 0):.3f}")
        print(f"  Target achieved: {'✅' if compression.get('deepseek_target_achieved') else '❌'}")
        
        print(f"\n🔤 EMBEDDING ANALYSIS:")
        print(f"  Average embedding dimension: {embedding.get('average_embedding_dimension', 0):.0f}")
        print(f"  Average token count: {embedding.get('average_token_count', 0):.0f}")
        print(f"  Embedding efficiency: {embedding.get('embedding_efficiency', 0):.2f}")
        
        print(f"\n⚡ PERFORMANCE SUMMARY:")
        print(f"  Average processing time: {performance.get('average_processing_time_ms', 0):.1f}ms")
        print(f"  Throughput: {performance.get('throughput_pages_per_second', 0):.1f} pages/second")
        print(f"  Performance grade: {performance.get('performance_grade', 'Unknown')}")
        
        print(f"\n🔍 DETAILED RESULTS:")
        for i, analysis in enumerate(results.get('deepseek_analysis', [])):
            print(f"\n  Text {i+1}: \"{analysis['original_text'][:50]}...\"")
            print(f"    Compression ratio: {analysis['compression_ratio']:.1f}x")
            print(f"    Fidelity: {analysis['fidelity']:.3f}")
            print(f"    Processing time: {analysis['processing_time_ms']:.1f}ms")
            print(f"    Token count: {analysis['token_count']}")
        
        # Generate and save report
        print(f"\n📋 GENERATING REPORT...")
        report = analyzer.generate_analysis_report(results)
        
        # Save report to file
        output_dir = Path("/K3D/Knowledge3D.local/tests/deepseek_standalone")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = output_dir / "test_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Report saved to: {report_file}")
        
        # Save JSON results
        results_file = output_dir / "test_results.json"
        import json
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Results saved to: {results_file}")
        
        print(f"\n{'=' * 60}")
        print("✅ STANDALONE DEESEEK OCR ANALYSIS TEST COMPLETED SUCCESSFULLY!")
        print(f"{'=' * 60}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())