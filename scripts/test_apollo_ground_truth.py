#!/usr/bin/env python3
"""
Test GPU OCR against APOLLO.PDF ground truth.

This script validates OCR accuracy using known ground truth bounding boxes
and text from the first page of APOLLO.PDF.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from knowledge3d.cranium.bridges.pdf_ingestion_bridge_phase_g import PhaseGPDFIngestionBridge


# Ground truth from APOLLO.PDF-Expected_Text.md (Page 0)
APOLLO_GROUND_TRUTH = [
    {"text": "I", "bbox": [363, 62, 417, 107]},
    {"text": "C", "bbox": [321, 108, 375, 153]},
    {"text": "A", "bbox": [435, 62, 489, 107]},
    {"text": "S", "bbox": [389, 108, 443, 153]},
    {"text": "E", "bbox": [507, 62, 561, 107]},
    {"text": "APOLLO 11", "bbox": [373, 163, 627, 192]},
    {"text": "A Teacher Resource Book", "bbox": [280, 212, 720, 238]},
    {"text": "Commemorating the", "bbox": [325, 332, 674, 353]},
    {"text": "20th anniversary of the", "bbox": [304, 362, 696, 383]},
    {"text": "Apollo 11 Moon Landing,", "bbox": [291, 392, 709, 413]},
    {"text": "1969 - 1989", "bbox": [414, 422, 586, 443]},
    {"text": "INTERNATIONAL COUNCIL OF ASSOCIATIONS FOR SCIENCE EDUCATION", "bbox": [102, 1353, 898, 1370]},
]


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """Calculate Intersection over Union for two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0

    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using edit distance."""
    text1 = text1.strip().lower()
    text2 = text2.strip().lower()

    if len(text1) == 0 and len(text2) == 0:
        return 1.0
    if len(text1) == 0 or len(text2) == 0:
        return 0.0

    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    edit_distance = dp[m][n]
    return 1.0 - (edit_distance / max(m, n))


def test_apollo_pdf():
    """Test OCR on APOLLO.PDF page 0 against ground truth."""
    apollo_pdf_path = Path("/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/Apollo 11/APOLLO.PDF")

    if not apollo_pdf_path.exists():
        print(f"ERROR: APOLLO.PDF not found at {apollo_pdf_path}")
        return

    print("="*80)
    print("APOLLO.PDF Ground Truth Validation")
    print("="*80)
    print()

    print("[1/4] Initializing Phase G PDF bridge...")
    bridge = PhaseGPDFIngestionBridge()
    print(f"      Glyph database: {len(bridge.glyph_metadata)} variants loaded")
    print(f"      GPU OCR enabled: {bridge.gpu_ocr_enabled}")
    print()

    print("[2/4] Processing APOLLO.PDF page 0 with GPU OCR...")

    try:
        result = bridge.ingest_pdf_page(apollo_pdf_path, page_num=0)
        print(f"      Objects extracted: {result.get('object_count', 0)}")
        print(f"      OCR stats: {bridge.ocr_stats}")
        print()
    except Exception as e:
        print(f"      ERROR during OCR: {e}")
        import traceback
        traceback.print_exc()
        return

    print("[3/4] Extracting predicted text and bounding boxes...")
    predicted_objects = []
    layout_graph = result.get("layout_graph", {}) if 'result' in locals() else {}
    nodes = layout_graph.get("nodes", [])

    for node in nodes:
        node_type = float(node.get("type", 0.0))
        if node_type != 1.0:
            continue

        bbox = node.get("bbox", (0.0, 0.0, 0.0, 0.0))
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        x, y, w, h = bbox
        text_content = (node.get("text_sample") or "").strip()
        if not text_content:
            continue

        predicted_objects.append(
            {
                "text": text_content,
                "bbox": [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))],
                "confidence": float(node.get("importance", 0.0)),
            }
        )

    print(f"      Predicted objects with text: {len(predicted_objects)}")
    for sample in predicted_objects[:12]:
        print(f"        -> text='{sample['text']}' bbox={sample['bbox']} conf={sample['confidence']:.2f}")
    print()

    print("[4/4] Matching predictions to ground truth...")
    matches = []
    matched_gt = set()

    for pred_idx, pred in enumerate(predicted_objects):
        best_iou, best_gt_idx = 0.0, -1
        for gt_idx, gt in enumerate(APOLLO_GROUND_TRUTH):
            if gt_idx not in matched_gt:
                iou = calculate_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_gt_idx = iou, gt_idx

        if best_iou >= 0.5:
            matched_gt.add(best_gt_idx)
            text_sim = calculate_text_similarity(pred["text"], APOLLO_GROUND_TRUTH[best_gt_idx]["text"])
            matches.append({
                "pred_idx": pred_idx,
                "gt_idx": best_gt_idx,
                "iou": best_iou,
                "text_similarity": text_sim,
                "predicted_text": pred["text"],
                "ground_truth_text": APOLLO_GROUND_TRUTH[best_gt_idx]["text"],
                "match_quality": "correct" if text_sim > 0.8 else "partial"
            })

    tp = sum(1 for m in matches if m["match_quality"] == "correct")
    fp = len(predicted_objects) - len(matches)
    fn = len(APOLLO_GROUND_TRUTH) - len(matched_gt)

    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Ground Truth Objects:  {len(APOLLO_GROUND_TRUTH)}")
    print(f"Predicted Objects:     {len(predicted_objects)}")
    print(f"True Positives:        {tp}")
    print(f"False Positives:       {fp}")
    print(f"False Negatives:       {fn}")
    print()
    print(f"Precision:             {precision:.2%}")
    print(f"Recall:                {recall:.2%}")
    print(f"F1 Score:              {f1_score:.2%}")
    print()

    if matches:
        print("="*80)
        print("DETAILED MATCHES (First 10)")
        print("="*80)
        print()
        for i, match in enumerate(matches[:10], 1):
            print(f"Match {i}:")
            print(f"  Ground Truth: \"{match['ground_truth_text']}\"")
            print(f"  Predicted:    \"{match['predicted_text']}\"")
            print(f"  IoU:          {match['iou']:.2f}")
            print(f"  Text Sim:     {match['text_similarity']:.2%}")
            print(f"  Quality:      {match['match_quality']}")
            print()

    print("="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_apollo_pdf()
