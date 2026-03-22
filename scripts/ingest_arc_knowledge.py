#!/usr/bin/env python3
"""Populate Drawing and Language galaxies with ARC reasoning anchors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse  # noqa: E402


DEFAULT_STORAGE_ROOT = Path("/K3D/Knowledge3D.local")


def _slug(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "_", lowered)
    lowered = lowered.strip("_")
    return lowered or "arc_anchor"


def _dedup_strs(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        resolved = str(value or "").strip()
        if not resolved:
            continue
        lowered = resolved.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(resolved)
    return result


def _arc_anchor_spec(
    *,
    ident: str,
    name: str,
    family_group: str,
    transform_family: str,
    summary: str,
    content: str,
    rpn_program: str,
    query_terms: list[str],
    keywords: list[str],
    semantics: str,
    layer: int = 2,
    confidence: float = 0.91,
    category: str = "arc_transform_primitive",
) -> dict[str, Any]:
    return {
        "id": f"arc_anchor_{ident}",
        "name": name,
        "family_group": family_group,
        "transform_family": transform_family,
        "summary": summary,
        "content": content,
        "rpn_program": rpn_program,
        "query_anchor": " ".join(_dedup_strs(query_terms)),
        "keywords": _dedup_strs(keywords),
        "semantics": semantics,
        "layer": int(layer),
        "confidence": float(confidence),
        "category": category,
    }


def _arc_anchor_entry(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": spec["id"],
        "name": spec["name"],
        "domain": "drawing",
        "category": spec["category"],
        "layer": int(spec.get("layer", 2)),
        "content": spec["content"],
        "summary": spec["summary"],
        "description": spec["summary"],
        "rpn_program": spec["rpn_program"],
        "metadata": {
            "subject": "arc_transform",
            "arc_family_group": spec["family_group"],
            "transform_family": spec["transform_family"],
            "primitive_plan": spec["transform_family"],
            "query_anchor": spec["query_anchor"],
            "semantics": spec["semantics"],
            "keywords": list(spec["keywords"]),
            "tags": list(spec["keywords"]),
            "confidence": float(spec["confidence"]),
            "ingest_source": "ingest_arc_knowledge",
            "specialist": "visual",
        },
    }


def _arc_language_symlink_entry(spec: dict[str, Any]) -> dict[str, Any]:
    content = (
        f"Language-side bridge for {spec['name']}. "
        f"Route natural-language ARC reasoning into Drawing anchor {spec['id']}."
    )
    return {
        "id": f"lang_arc_symlink_{spec['id'][11:]}",
        "name": f"{spec['name']} language bridge",
        "domain": "language",
        "category": "meaning_symlink",
        "content": content,
        "summary": f"{spec['name']} language→drawing bridge",
        "description": "Natural-language anchor that routes the TRM from Language Galaxy into the matching ARC Drawing anchor.",
        "rpn_program": "",
        "answer_text": "",
        "symlink_to": spec["id"],
        "metadata": {
            "ingest_source": "ingest_arc_knowledge",
            "bridge_role": "language_to_drawing_anchor",
            "symlink_target": spec["id"],
            "symlink_galaxy": "Drawing",
            "cross_modal": "language_to_drawing",
            "modalities": ["language", "visual", "drawing"],
            "specialist": "visual",
            "arc_family_group": spec["family_group"],
            "transform_family": spec["transform_family"],
            "query_anchor": spec["query_anchor"],
            "aliases": [str(spec["name"]).strip().lower()],
            "keywords": list(spec["keywords"]),
            "tags": list(spec["keywords"]),
            "semantics": f"Language meaning routes into ARC concept anchor {spec['id']}.",
            "direct_eval": False,
        },
    }


def _expand_specs(
    *,
    family_group: str,
    transform_family: str,
    items: list[tuple[str, str, str, str, list[str], list[str], str]],
    category: str = "arc_transform_primitive",
    layer: int = 2,
    confidence: float = 0.91,
) -> list[dict[str, Any]]:
    return [
        _arc_anchor_spec(
            ident=ident,
            name=name,
            family_group=family_group,
            transform_family=transform_family,
            summary=summary,
            content=content,
            rpn_program=rpn_program,
            query_terms=query_terms,
            keywords=keywords,
            semantics=semantics,
            layer=layer,
            confidence=confidence,
            category=category,
        )
        for ident, name, summary, content, query_terms, keywords, rpn_program in items
        for semantics in [f"{name} anchor for ARC {family_group.replace('_', ' ')} reasoning."]
    ]


def _object_centric_specs() -> list[dict[str, Any]]:
    items = [
        ("connected_component_same_color_4conn", "Connected component same-color 4-connected", "Detect same-color 4-connected objects.", "Flood-fill 4-connected regions of the same color and label each discrete object.", ["find objects", "same color regions", "4 connected", "separate groups", "segment by color"], ["connected", "component", "object", "flood", "fill", "4conn"], "GRID_ITERATE CELL_COLOR FLOOD_FILL_4CONN OBJECT_LABEL"),
        ("connected_component_same_color_8conn", "Connected component same-color 8-connected", "Detect same-color 8-connected objects.", "Flood-fill 8-connected regions of the same color and label diagonal-touching cells as one object.", ["find objects", "same color regions", "8 connected", "diagonal groups", "segment by color"], ["connected", "component", "object", "flood", "fill", "8conn"], "GRID_ITERATE CELL_COLOR FLOOD_FILL_8CONN OBJECT_LABEL"),
        ("connected_component_any_color_4conn", "Connected component any-color 4-connected", "Detect object masks ignoring palette.", "Segment contiguous non-background cells into objects while ignoring per-cell color identity.", ["find shapes", "ignore color", "4 connected", "foreground objects"], ["object", "shape", "mask", "foreground", "4conn"], "GRID_MASK_NONZERO FLOOD_FILL_4CONN OBJECT_LABEL"),
        ("connected_component_any_color_8conn", "Connected component any-color 8-connected", "Detect diagonal object masks ignoring palette.", "Segment contiguous non-background cells into diagonal-aware objects while ignoring per-cell color identity.", ["find shapes", "ignore color", "8 connected", "foreground objects"], ["object", "shape", "mask", "foreground", "8conn"], "GRID_MASK_NONZERO FLOOD_FILL_8CONN OBJECT_LABEL"),
        ("background_foreground_separation", "Background foreground separation", "Separate background from active objects.", "Infer the dominant background color, then isolate the active foreground objects against it.", ["background color", "foreground objects", "separate active cells", "majority color background"], ["background", "foreground", "segmentation", "majority"], "GRID_COLOR_HISTOGRAM COLOR_MODE FOREGROUND_MASK"),
        ("figure_ground_reversal", "Figure ground reversal", "Swap which color acts as figure versus ground.", "Reinterpret the grid by promoting background cells to figure and treating prior figure cells as support.", ["reverse figure ground", "swap background foreground", "invert object role"], ["figure", "ground", "invert", "segmentation"], "GRID COLOR_MODE FIGURE_GROUND_SWAP"),
        ("touching_groups_4conn", "Touching groups 4-connected", "Detect objects touching by edges.", "Build an adjacency graph of objects that share 4-connected edge contact.", ["objects touching", "edge contact", "4 connected neighbors"], ["touching", "adjacency", "edge", "objects"], "OBJECT_LABEL OBJECT_ADJACENCY_4CONN"),
        ("touching_groups_8conn", "Touching groups 8-connected", "Detect objects touching by edges or corners.", "Build an adjacency graph of objects that share 8-connected contact including corners.", ["objects touching", "corner contact", "8 connected neighbors"], ["touching", "adjacency", "corner", "objects"], "OBJECT_LABEL OBJECT_ADJACENCY_8CONN"),
        ("isolated_objects_only", "Isolated objects only", "Keep only isolated objects.", "Filter out objects that touch or cluster so only isolated singleton components remain.", ["isolated object", "standalone shape", "remove touching groups"], ["isolated", "singleton", "object", "filter"], "OBJECT_LABEL OBJECT_KEEP_ISOLATED"),
        ("marker_seed_region", "Marker-seeded region detection", "Use a marker to choose the active object region.", "Start a flood-fill from marker cells to identify the object or region indicated by the marker.", ["marker chooses object", "seed region", "trace from marker"], ["marker", "seed", "region", "object"], "MARKER_CELLS FLOOD_FILL_FROM_SEEDS"),
        ("hole_preserving_component", "Hole-preserving component extraction", "Extract objects while preserving internal holes.", "Treat shells and holes as one structured object so symbolic hole counts remain visible.", ["object with holes", "preserve empty center", "frame object"], ["hole", "frame", "component", "object"], "OBJECT_LABEL HOLE_MAP OBJECT_WITH_HOLES"),
        ("object_bounding_box", "Object bounding box", "Compute object bounding rectangles.", "Find the smallest axis-aligned rectangle enclosing each object.", ["bounding box", "object rectangle", "crop around shape"], ["bounding", "box", "rectangle", "crop"], "OBJECT_LABEL OBJECT_BBOX"),
        ("bounding_box_union", "Bounding box union", "Compute a box enclosing multiple objects.", "Merge multiple object bounds into a single union box for shared cropping or framing.", ["union bounding box", "group crop", "objects together"], ["bounding", "box", "union", "group"], "OBJECT_BBOX_LIST BBOX_UNION"),
        ("bounding_box_intersection", "Bounding box intersection", "Intersect object bounds.", "Find the overlapping aligned window shared by multiple bounding boxes.", ["intersect bounding boxes", "shared crop region", "overlap rectangle"], ["bounding", "intersection", "overlap", "box"], "OBJECT_BBOX_LIST BBOX_INTERSECTION"),
        ("object_centroid", "Object centroid", "Compute object centroids.", "Compute each object's center of mass from its occupied cells.", ["object center", "centroid", "middle of shape"], ["centroid", "center", "object", "mass"], "OBJECT_MASK CENTROID"),
        ("color_weighted_centroid", "Color-weighted centroid", "Compute centroids biased by color importance.", "Use cell color weights to emphasize symbolic marker colors when locating the center.", ["weighted centroid", "marker biased center", "color weighted object center"], ["centroid", "color", "weight", "marker"], "OBJECT_MASK COLOR_WEIGHTED_CENTROID"),
        ("object_size_counting", "Object size counting", "Measure cells per object.", "Count how many colored cells belong to each detected object.", ["object size", "cells per object", "count shape area"], ["size", "count", "area", "object"], "OBJECT_LABEL OBJECT_CELL_COUNT"),
        ("object_area_rank", "Object area ranking", "Rank objects by area.", "Sort objects by cell count to identify the largest, smallest, and median shapes.", ["largest object", "smallest object", "rank by size"], ["rank", "size", "largest", "smallest"], "OBJECT_LABEL OBJECT_CELL_COUNT SORT_BY_SIZE"),
        ("object_perimeter", "Object perimeter", "Measure boundary length per object.", "Count each object's exterior boundary cells to compare outline complexity.", ["object perimeter", "boundary length", "outline count"], ["perimeter", "boundary", "outline", "object"], "OBJECT_MASK OBJECT_PERIMETER"),
        ("object_aspect_ratio", "Object aspect ratio", "Measure width-height proportion.", "Derive each object's aspect ratio from its bounding width and height.", ["aspect ratio", "width height", "object proportions"], ["aspect", "ratio", "width", "height"], "OBJECT_BBOX OBJECT_ASPECT_RATIO"),
        ("symmetry_axis_detection", "Object symmetry axis detection", "Detect object symmetry axes.", "Detect whether an object is symmetric horizontally, vertically, or diagonally.", ["symmetry axis", "balanced object", "mirror shape"], ["symmetry", "axis", "mirror", "object"], "OBJECT_MASK DETECT_SYMMETRY_AXIS"),
        ("containment_detection", "Object containment detection", "Detect nested objects.", "Determine whether one object lies completely inside the boundary of another object.", ["object inside object", "containment", "nested shape"], ["containment", "inside", "nested", "object"], "OBJECT_BBOX OBJECT_CONTAINMENT"),
        ("adjacency_detection_4conn", "Object adjacency detection 4-connected", "Detect edge-adjacent objects.", "Determine when two objects touch along a side without diagonal-only contact.", ["adjacent objects", "side touch", "4 connected neighbor objects"], ["adjacent", "touch", "edge", "object"], "OBJECT_LABEL OBJECT_ADJACENCY_4CONN"),
        ("adjacency_detection_8conn", "Object adjacency detection 8-connected", "Detect corner-adjacent objects.", "Determine when two objects touch along a side or corner.", ["adjacent objects", "corner touch", "8 connected neighbor objects"], ["adjacent", "touch", "corner", "object"], "OBJECT_LABEL OBJECT_ADJACENCY_8CONN"),
        ("alignment_horizontal", "Horizontal object alignment", "Detect row alignment.", "Test whether objects share the same y band or horizon line.", ["objects on same row", "horizontal alignment", "aligned across"], ["alignment", "horizontal", "row", "object"], "OBJECT_CENTROIDS ALIGNMENT_HORIZONTAL"),
        ("alignment_vertical", "Vertical object alignment", "Detect column alignment.", "Test whether objects share the same x band or vertical spine.", ["objects on same column", "vertical alignment", "aligned down"], ["alignment", "vertical", "column", "object"], "OBJECT_CENTROIDS ALIGNMENT_VERTICAL"),
        ("alignment_diagonal", "Diagonal object alignment", "Detect diagonal object chains.", "Test whether object centroids lie on a shared diagonal direction.", ["objects on diagonal", "diagonal alignment", "staircase pattern"], ["alignment", "diagonal", "chain", "object"], "OBJECT_CENTROIDS ALIGNMENT_DIAGONAL"),
        ("sort_by_size_ascending", "Sort objects by size ascending", "Order objects from smallest to largest.", "Produce an object ordering from smallest to largest area.", ["sort small to large", "object order by size", "ascending sizes"], ["sort", "size", "ascending", "object"], "OBJECT_CELL_COUNT SORT_ASC"),
        ("sort_by_size_descending", "Sort objects by size descending", "Order objects from largest to smallest.", "Produce an object ordering from largest to smallest area.", ["sort large to small", "object order by size", "descending sizes"], ["sort", "size", "descending", "object"], "OBJECT_CELL_COUNT SORT_DESC"),
        ("sort_by_position_x", "Sort objects by x position", "Order objects left to right.", "Sort objects by their horizontal centroid or bounding-box origin.", ["sort left to right", "x position order", "object horizontal order"], ["sort", "position", "x", "object"], "OBJECT_CENTROIDS SORT_BY_X"),
        ("sort_by_position_y", "Sort objects by y position", "Order objects top to bottom.", "Sort objects by their vertical centroid or bounding-box origin.", ["sort top to bottom", "y position order", "object vertical order"], ["sort", "position", "y", "object"], "OBJECT_CENTROIDS SORT_BY_Y"),
        ("shape_match_ignore_color", "Shape matching ignoring color", "Match same shapes despite palette changes.", "Compare normalized masks so objects with the same shape match even when recolored.", ["same shape different color", "match by shape", "ignore palette"], ["match", "shape", "ignore color", "object"], "OBJECT_MASK NORMALIZE_SHAPE SHAPE_MATCH"),
        ("color_match_ignore_shape", "Color matching ignoring shape", "Match same colors despite geometry changes.", "Find objects sharing a color identity even if their shape differs.", ["same color different shape", "match by color", "palette identity"], ["match", "color", "ignore shape", "object"], "OBJECT_PALETTE COLOR_MATCH"),
        ("mask_match", "Binary mask matching", "Match objects by binary footprint.", "Compare binary occupancy masks after removing color information.", ["binary footprint", "same mask", "object silhouette match"], ["mask", "footprint", "silhouette", "match"], "OBJECT_MASK BINARY_MASK_MATCH"),
        ("hole_count_match", "Hole-count object matching", "Match objects by number of holes.", "Classify frames and shells using the number of enclosed voids.", ["hole count", "match framed objects", "object void count"], ["holes", "count", "frame", "match"], "OBJECT_WITH_HOLES HOLE_COUNT_MATCH"),
        ("orientation_match", "Orientation-aware object matching", "Match objects by orientation state.", "Match objects after detecting which way each motif points or opens.", ["orientation match", "pointing same way", "directional shape"], ["orientation", "match", "direction", "shape"], "OBJECT_MASK ORIENTATION_SIGNATURE_MATCH"),
        ("size_match", "Size-based object matching", "Match objects by equal area or span.", "Find objects whose area or dimensions match a target object.", ["equal size object", "same dimensions", "match by span"], ["size", "match", "dimensions", "area"], "OBJECT_BBOX OBJECT_SIZE_MATCH"),
        ("template_match_object", "Template object matching", "Match objects against a template motif.", "Use a known object template to locate repeated motif instances in the grid.", ["template match object", "find repeated motif", "locate same shape"], ["template", "match", "motif", "object"], "OBJECT_TEMPLATE OBJECT_TEMPLATE_MATCH"),
        ("nearest_shape_clone", "Nearest shape clone detection", "Find the nearest reused shape.", "Identify the closest object that reuses the same normalized mask as a reference object.", ["nearest repeated shape", "closest clone object", "same motif nearest"], ["nearest", "clone", "shape", "object"], "OBJECT_MASK SHAPE_MATCH NEAREST_NEIGHBOR"),
        ("repeated_motif_instance", "Repeated motif instance detection", "Find every motif instance.", "Locate all translated, reflected, or recolored copies of a base motif.", ["repeated motif", "all copies of shape", "motif instances"], ["motif", "repeat", "instance", "object"], "OBJECT_MASK MOTIF_INSTANCE_SCAN"),
        ("object_role_marker", "Marker-object role pairing", "Bind markers to objects they refer to.", "Pair marker cells with their nearest or aligned target objects.", ["marker points to object", "pair marker and target", "object role binding"], ["marker", "role", "bind", "target"], "MARKER_OBJECT_PAIRING"),
        ("extract_object", "Object extraction", "Isolate one object as a subgrid.", "Cut out one labeled object into its own focused working grid.", ["extract object", "isolate shape", "subgrid from object"], ["extract", "object", "subgrid", "crop"], "OBJECT_LABEL OBJECT_EXTRACT"),
        ("place_object", "Object placement", "Paste an object into a target location.", "Place an extracted object into a new grid coordinate or frame slot.", ["place object", "paste shape", "move object to location"], ["place", "object", "paste", "target"], "OBJECT_SUBGRID TARGET_X TARGET_Y OBJECT_PLACE"),
        ("delete_object", "Object deletion", "Remove selected objects.", "Erase objects matching a chosen rule while leaving the rest of the grid intact.", ["delete object", "remove selected shape", "erase one object"], ["delete", "remove", "erase", "object"], "OBJECT_LABEL OBJECT_DELETE"),
        ("clone_object", "Object cloning", "Duplicate a selected object.", "Copy one object and spawn duplicates at guided positions.", ["clone object", "duplicate shape", "copy motif"], ["clone", "duplicate", "copy", "object"], "OBJECT_EXTRACT CLONE_OBJECT"),
        ("move_object", "Object movement", "Translate a selected object.", "Move one object along an inferred direction or offset while preserving shape.", ["move object", "shift shape", "translate selected object"], ["move", "translate", "object", "shift"], "OBJECT_EXTRACT DX DY OBJECT_TRANSLATE"),
        ("swap_objects", "Object swapping", "Exchange two objects.", "Swap the positions or colors of two matched objects.", ["swap objects", "exchange shapes", "trade positions"], ["swap", "exchange", "object", "pair"], "OBJECT_PAIR OBJECT_SWAP"),
        ("merge_objects", "Object merging", "Merge object masks.", "Fuse multiple selected objects into one composite region or frame.", ["merge objects", "combine shapes", "fuse regions"], ["merge", "combine", "fuse", "object"], "OBJECT_SET OBJECT_MERGE"),
        ("split_object", "Object splitting", "Split an object into subparts.", "Break a compound object into logical sub-objects by color, holes, or separators.", ["split object", "divide shape", "separate compound object"], ["split", "divide", "object", "parts"], "OBJECT_MASK OBJECT_SPLIT"),
        ("normalize_object_origin", "Object origin normalization", "Move an object to canonical origin.", "Translate an object so its bounding box starts at the canonical top-left origin.", ["normalize object origin", "anchor shape at top left", "canonical object position"], ["normalize", "origin", "canonical", "object"], "OBJECT_BBOX NORMALIZE_OBJECT_ORIGIN"),
        ("canonical_object_pose", "Canonical object pose", "Normalize object orientation.", "Rotate or mirror an object into a canonical pose for comparison and reuse.", ["canonical object pose", "normalize orientation", "standardized shape pose"], ["canonical", "pose", "orientation", "object"], "OBJECT_MASK CANONICAL_OBJECT_POSE"),
        ("object_count", "Object counting", "Count discrete objects.", "Count how many labeled connected components exist in the active grid.", ["count objects", "how many shapes", "number of groups"], ["count", "objects", "groups", "components"], "OBJECT_LABEL OBJECT_COUNT"),
        ("cell_count_per_object", "Cell count per object", "Count cells within each object.", "Return a per-object list of occupied cell counts.", ["cells per object", "object area list", "size list"], ["cell", "count", "area", "object"], "OBJECT_LABEL OBJECT_CELL_COUNT"),
        ("color_count_per_object", "Color count per object", "Count palette cells within each object.", "Measure how many cells of each color appear inside each object.", ["color histogram in object", "palette per shape", "count colors per object"], ["color", "count", "histogram", "object"], "OBJECT_MASK OBJECT_COLOR_HISTOGRAM"),
        ("foreground_majority_object", "Foreground majority object", "Find the most dominant object.", "Select the object with the strongest foreground presence by area or salience.", ["dominant object", "biggest foreground shape", "main object"], ["foreground", "majority", "dominant", "object"], "OBJECT_SALIENCE SELECT_MAX"),
        ("minority_object", "Minority object", "Find the least dominant object.", "Select the smallest or rarest object relative to the task's dominant motifs.", ["outlier small object", "minority object", "rare shape"], ["minority", "outlier", "rare", "object"], "OBJECT_SALIENCE SELECT_MIN"),
        ("object_labeling", "Object labeling", "Assign stable IDs to objects.", "Label each object deterministically for later extraction and rule application.", ["label objects", "stable object ids", "index all shapes"], ["label", "id", "index", "object"], "OBJECT_LABEL ASSIGN_IDS"),
        ("representative_object", "Representative object selection", "Pick a canonical exemplar object.", "Choose one object as the prototype for shape copying or analogy.", ["prototype object", "representative shape", "choose exemplar"], ["prototype", "representative", "exemplar", "object"], "OBJECT_CLUSTER SELECT_REPRESENTATIVE"),
        ("outlier_object", "Outlier object detection", "Find the one object unlike the others.", "Detect the object whose shape, color, or size deviates from the dominant set.", ["odd object out", "different shape", "outlier motif"], ["outlier", "different", "odd", "object"], "OBJECT_FEATURES FIND_OUTLIER"),
        ("singleton_detection", "Singleton object detection", "Detect one-cell or one-off objects.", "Locate singleton cells or tiny components that act as markers or noise.", ["singleton object", "one cell marker", "tiny outlier"], ["singleton", "tiny", "marker", "object"], "OBJECT_LABEL FILTER_SINGLETONS"),
        ("component_trace_order", "Component trace order", "Trace objects in deterministic order.", "Enumerate objects in a stable read order so later rules can reference first, second, or last.", ["trace object order", "first second last object", "stable scan order"], ["trace", "order", "enumerate", "object"], "OBJECT_LABEL TRACE_OBJECT_ORDER"),
        ("color_segment_objects", "Color-segment object detection", "Segment objects by color family.", "Split the grid into object groups organized primarily by palette identity.", ["segment by color family", "objects per color", "palette groups"], ["color", "segment", "palette", "object"], "GRID COLOR_SEGMENT_OBJECTS"),
        ("pattern_segment_objects", "Pattern-segment object detection", "Segment objects by repeated internal pattern.", "Group objects that share repeated stripes, holes, or internal tiling.", ["segment by internal pattern", "group patterned objects", "same motif family"], ["pattern", "segment", "motif", "object"], "OBJECT_MASK INTERNAL_PATTERN_SEGMENT"),
        ("occlusion_part_separation", "Occlusion part separation", "Separate overlapping object parts.", "Infer when an object is partially hidden and recover its visible subparts.", ["occluded object parts", "split overlap", "recover hidden shape"], ["occlusion", "overlap", "parts", "object"], "OBJECT_MASK OCCLUSION_PARTS"),
        ("anchor_object_pairing", "Anchor-object pairing", "Bind anchors to the objects they command.", "Pair anchor marks, arrows, or frames with the object they modify.", ["anchor paired with object", "which object the marker commands", "anchor target"], ["anchor", "pairing", "command", "object"], "ANCHOR_OBJECT_PAIRING"),
        ("object_trail_tracing", "Object trail tracing", "Trace motion trails of repeated objects.", "Follow repeated object positions across a line or path to infer direction.", ["object trail", "trace moved copies", "shape path"], ["trail", "trace", "motion", "object"], "OBJECT_CENTROIDS TRACE_TRAIL"),
        ("outline_extraction", "Object outline extraction", "Extract object outlines only.", "Reduce filled objects to their boundary shells when the rule depends on outline structure.", ["outline of object", "boundary only shape", "shell extraction"], ["outline", "boundary", "shell", "object"], "OBJECT_MASK EXTRACT_OUTLINE"),
        ("fill_state_detection", "Fill-state detection", "Detect filled versus hollow objects.", "Classify whether an object is solid, framed, striped, or hollow.", ["filled or hollow object", "solid frame detection", "interior state"], ["fill", "hollow", "solid", "frame"], "OBJECT_MASK DETECT_FILL_STATE"),
        ("hole_extraction", "Hole extraction", "Extract the holes inside objects.", "Treat object holes as symbolic subregions that can drive color or action rules.", ["extract holes", "interior void regions", "object empty spaces"], ["holes", "void", "interior", "object"], "OBJECT_WITH_HOLES EXTRACT_HOLES"),
        ("shell_core_split", "Shell-core split", "Separate object shell from object core.", "Split framed objects into outer shell and inner fill regions.", ["shell and core", "frame versus interior", "split framed object"], ["shell", "core", "frame", "interior"], "OBJECT_WITH_HOLES SHELL_CORE_SPLIT"),
        ("mirrored_pair_detection", "Mirrored pair detection", "Detect object pairs reflected across an axis.", "Find two objects that are mirror mates across a task-defined symmetry axis.", ["mirrored object pair", "reflected shapes", "matching across axis"], ["mirror", "pair", "reflection", "object"], "OBJECT_MASK MIRRORED_PAIR_DETECT"),
    ]
    return _expand_specs(
        family_group="object_centric",
        transform_family="object_detection",
        items=items,
    )


def _geometric_specs() -> list[dict[str, Any]]:
    items = [
        ("rotate_90_cw", "Rotate 90 clockwise", "Rotate content by a quarter-turn clockwise.", "Rotate the active object or grid 90 degrees clockwise.", ["rotate clockwise", "quarter turn right", "arc rotation 90"], ["rotate", "90", "clockwise", "transform"], "GRID ROTATE_90_CW"),
        ("rotate_90_ccw", "Rotate 90 counterclockwise", "Rotate content by a quarter-turn counterclockwise.", "Rotate the active object or grid 90 degrees counterclockwise.", ["rotate counterclockwise", "quarter turn left", "arc rotation 90"], ["rotate", "90", "counterclockwise", "transform"], "GRID ROTATE_90_CCW"),
        ("rotate_180", "Rotate 180", "Rotate content by a half turn.", "Rotate the active object or grid by 180 degrees.", ["rotate 180", "flip upside down", "half turn"], ["rotate", "180", "half turn", "transform"], "GRID ROTATE_180"),
        ("rotate_270", "Rotate 270", "Rotate content by three quarter-turns.", "Rotate the active object or grid 270 degrees clockwise.", ["rotate 270", "three quarter turn", "rotate left 90 inverse"], ["rotate", "270", "transform"], "GRID ROTATE_270"),
        ("mirror_horizontal", "Mirror horizontal", "Reflect content across a horizontal axis.", "Mirror the active object or grid top-to-bottom.", ["mirror horizontal", "flip vertically", "reflect across horizontal axis"], ["mirror", "horizontal", "flip", "reflect"], "GRID MIRROR_HORIZONTAL"),
        ("mirror_vertical", "Mirror vertical", "Reflect content across a vertical axis.", "Mirror the active object or grid left-to-right.", ["mirror vertical", "flip horizontally", "reflect across vertical axis"], ["mirror", "vertical", "flip", "reflect"], "GRID MIRROR_VERTICAL"),
        ("mirror_diagonal_main", "Mirror main diagonal", "Reflect content across the main diagonal.", "Transpose and reflect the grid across the top-left to bottom-right axis.", ["mirror main diagonal", "transpose shape", "reflect tl br diagonal"], ["mirror", "diagonal", "main", "transpose"], "GRID MIRROR_DIAGONAL_MAIN"),
        ("mirror_diagonal_anti", "Mirror anti-diagonal", "Reflect content across the anti-diagonal.", "Reflect the grid across the top-right to bottom-left diagonal.", ["mirror anti diagonal", "reflect tr bl diagonal", "anti transpose"], ["mirror", "diagonal", "anti", "reflect"], "GRID MIRROR_DIAGONAL_ANTI"),
        ("transpose", "Transpose grid", "Swap rows and columns.", "Transpose the active grid or object coordinate frame.", ["transpose rows columns", "swap x y", "grid transpose"], ["transpose", "swap", "rows", "columns"], "GRID TRANSPOSE"),
        ("identity", "Identity transform", "Keep content unchanged.", "Recognize when the correct transform is to preserve the object or grid exactly.", ["no change", "keep same", "identity transform"], ["identity", "preserve", "same", "unchanged"], "GRID IDENTITY"),
        ("translate_up", "Translate up", "Move content upward.", "Shift the active object or pattern upward by a learned offset.", ["move up", "translate upward", "shift object up"], ["translate", "up", "shift", "move"], "GRID TRANSLATE_UP"),
        ("translate_down", "Translate down", "Move content downward.", "Shift the active object or pattern downward by a learned offset.", ["move down", "translate downward", "shift object down"], ["translate", "down", "shift", "move"], "GRID TRANSLATE_DOWN"),
        ("translate_left", "Translate left", "Move content left.", "Shift the active object or pattern left by a learned offset.", ["move left", "translate left", "shift object left"], ["translate", "left", "shift", "move"], "GRID TRANSLATE_LEFT"),
        ("translate_right", "Translate right", "Move content right.", "Shift the active object or pattern right by a learned offset.", ["move right", "translate right", "shift object right"], ["translate", "right", "shift", "move"], "GRID TRANSLATE_RIGHT"),
        ("crop_to_object", "Crop to object", "Shrink a grid to object bounds.", "Crop the grid to the bounding box of the active object or content region.", ["crop to object", "trim border", "tight bounding box"], ["crop", "bounding box", "trim", "object"], "GRID CROP_TO_OBJECT"),
        ("crop_to_nonzero", "Crop to nonzero content", "Crop to all non-background cells.", "Trim away empty borders and keep only the active non-background region.", ["crop to content", "remove empty border", "trim zeros"], ["crop", "content", "nonzero", "trim"], "GRID CROP_TO_NONZERO"),
        ("scale_up_2x", "Scale up 2x", "Double object size.", "Expand the object or motif by a factor of two in each dimension.", ["scale 2x", "double size", "enlarge pattern"], ["scale", "2x", "expand", "enlarge"], "GRID SCALE_UP_2X"),
        ("scale_up_3x", "Scale up 3x", "Triple object size.", "Expand the object or motif by a factor of three in each dimension.", ["scale 3x", "triple size", "enlarge pattern"], ["scale", "3x", "expand", "enlarge"], "GRID SCALE_UP_3X"),
        ("scale_down_2x", "Scale down 2x", "Downsample content by two.", "Compress a pattern using 2x2 blocks and a majority or representative vote.", ["scale down 2x", "downsample", "compress pattern"], ["scale", "downsample", "2x", "compress"], "GRID SCALE_DOWN_2X"),
        ("tile_to_fill", "Tile to fill", "Repeat a motif until the frame is filled.", "Use repeated tiling to expand a small pattern across a target output frame.", ["tile to fill", "repeat pattern to target size", "fill frame with motif"], ["tile", "repeat", "fill", "motif"], "GRID TILE_TO_FILL"),
        ("complete_reflective_horizontal", "Complete horizontal reflection symmetry", "Finish a horizontally mirrored pattern.", "Use a horizontal symmetry axis to complete the missing half of a shape.", ["complete mirror horizontally", "finish reflected shape", "symmetry completion"], ["complete", "symmetry", "mirror", "horizontal"], "GRID COMPLETE_REFLECTIVE_HORIZONTAL"),
        ("complete_reflective_vertical", "Complete vertical reflection symmetry", "Finish a vertically mirrored pattern.", "Use a vertical symmetry axis to complete the missing half of a shape.", ["complete mirror vertically", "finish reflected shape", "symmetry completion"], ["complete", "symmetry", "mirror", "vertical"], "GRID COMPLETE_REFLECTIVE_VERTICAL"),
        ("complete_rotational_two_way", "Complete 180-degree rotational symmetry", "Finish a half-turn rotational pattern.", "Use 180-degree rotational symmetry to infer missing cells.", ["complete rotational symmetry 180", "half turn completion", "rotational mirror"], ["rotational", "symmetry", "180", "complete"], "GRID COMPLETE_ROTATIONAL_180"),
        ("complete_rotational_four_way", "Complete 90-degree rotational symmetry", "Finish a quarter-turn rotational pattern.", "Use 90-degree rotational symmetry to infer missing quadrants.", ["complete rotational symmetry 90", "quarter turn completion", "four way rotation"], ["rotational", "symmetry", "90", "complete"], "GRID COMPLETE_ROTATIONAL_90"),
        ("detect_symmetry_axis_horizontal", "Detect horizontal symmetry axis", "Locate a horizontal mirror axis.", "Find the y axis around which the shape reflects.", ["find horizontal symmetry axis", "mirror line across rows", "reflection axis"], ["symmetry", "axis", "horizontal", "detect"], "GRID DETECT_SYMMETRY_AXIS_H"),
        ("detect_symmetry_axis_vertical", "Detect vertical symmetry axis", "Locate a vertical mirror axis.", "Find the x axis around which the shape reflects.", ["find vertical symmetry axis", "mirror line across columns", "reflection axis"], ["symmetry", "axis", "vertical", "detect"], "GRID DETECT_SYMMETRY_AXIS_V"),
        ("break_symmetry_cell", "Break-symmetry cell detection", "Find the one cell breaking symmetry.", "Locate the cell or region whose removal or recolor would restore symmetry.", ["cell that breaks symmetry", "one wrong pixel", "repair symmetry"], ["break", "symmetry", "repair", "cell"], "GRID FIND_BREAK_SYMMETRY_CELL"),
        ("transform_per_object", "Transform per object", "Apply transforms object by object.", "Choose a transform independently for each object instead of for the whole grid.", ["transform each object separately", "per object rotate mirror move", "independent object transform"], ["per object", "transform", "independent", "objects"], "OBJECT_LABEL TRANSFORM_PER_OBJECT"),
        ("transform_conditional", "Conditional transform", "Transform only objects matching a condition.", "Apply the transform only when an object matches color, size, or position criteria.", ["conditional transform", "only some objects change", "if object matches"], ["conditional", "transform", "criteria", "object"], "OBJECT_LABEL CONDITIONAL_TRANSFORM"),
        ("rotate_then_mirror", "Rotate then mirror", "Compose rotation followed by reflection.", "Apply a rotation first and then reflect the resulting shape.", ["rotate then mirror", "compose rotation and reflection", "two step transform"], ["compose", "rotate", "mirror", "two step"], "GRID ROTATE_THEN_MIRROR"),
        ("mirror_then_rotate", "Mirror then rotate", "Compose reflection followed by rotation.", "Reflect the shape first and then rotate the result.", ["mirror then rotate", "reflection before rotation", "two step transform"], ["compose", "mirror", "rotate", "two step"], "GRID MIRROR_THEN_ROTATE"),
        ("scale_then_translate", "Scale then translate", "Compose scaling followed by movement.", "Resize the object and then place it into a destination slot or corner.", ["scale then move", "resize and translate", "enlarge then place"], ["compose", "scale", "translate", "two step"], "GRID SCALE_THEN_TRANSLATE"),
        ("translate_then_crop", "Translate then crop", "Compose movement followed by cropping.", "Shift the active content, then crop the target region around it.", ["move then crop", "translate and trim", "shift before crop"], ["compose", "translate", "crop", "two step"], "GRID TRANSLATE_THEN_CROP"),
        ("per_object_rotate", "Per-object rotation", "Rotate each object independently.", "Choose or infer a rotation separately for each labeled object.", ["rotate each object", "independent object rotation", "objectwise rotate"], ["per object", "rotate", "objectwise", "transform"], "OBJECT_LABEL ROTATE_EACH_OBJECT"),
        ("per_object_mirror", "Per-object mirror", "Mirror each object independently.", "Choose or infer a reflection separately for each labeled object.", ["mirror each object", "independent object reflection", "objectwise mirror"], ["per object", "mirror", "objectwise", "transform"], "OBJECT_LABEL MIRROR_EACH_OBJECT"),
        ("preserve_largest_transform_rest", "Preserve largest transform the rest", "Leave largest object fixed while transforming others.", "Use the dominant object as anchor and transform the remaining objects around it.", ["keep largest object fixed", "transform smaller objects", "anchor dominant shape"], ["preserve", "largest", "anchor", "transform"], "OBJECT_LABEL PRESERVE_LARGEST_TRANSFORM_REST"),
        ("conditional_identity", "Conditional identity", "Some objects remain unchanged.", "Recognize when only a subset transforms and the rest should stay identical.", ["some objects unchanged", "conditional identity", "leave others same"], ["identity", "conditional", "preserve", "objects"], "OBJECT_LABEL CONDITIONAL_IDENTITY"),
        ("orientation_normalize", "Orientation normalize", "Normalize motifs to a shared orientation.", "Rotate or mirror objects until they share a canonical orientation for comparison.", ["normalize orientation", "same facing direction", "canonical rotation"], ["orientation", "normalize", "canonical", "object"], "OBJECT_MASK ORIENTATION_NORMALIZE"),
        ("bounding_frame_align", "Bounding-frame alignment", "Align objects to a shared frame.", "Translate objects so their bounding boxes align to a common guide frame.", ["align bounding boxes", "shared frame", "line up objects"], ["align", "frame", "bounding box", "objects"], "OBJECT_BBOX ALIGN_TO_FRAME"),
        ("rotate_around_anchor", "Rotate around anchor", "Rotate content around a marker anchor.", "Treat a marker or centroid as pivot and rotate a shape around it.", ["rotate around marker", "pivot rotation", "anchor as center"], ["rotate", "anchor", "pivot", "marker"], "GRID ROTATE_AROUND_ANCHOR"),
        ("mirror_around_marker_axis", "Mirror around marker axis", "Reflect content across a marker-defined axis.", "Use marker cells to infer where the reflection axis lies.", ["mirror using marker axis", "reflect by guide marker", "marker chooses axis"], ["mirror", "marker", "axis", "reflect"], "GRID MIRROR_AROUND_MARKER_AXIS"),
        ("scale_to_fit_frame", "Scale to fit frame", "Resize an object to fill a target frame.", "Scale an object until it exactly fits the destination box or border.", ["scale to fit box", "resize into frame", "fit target bounds"], ["scale", "fit", "frame", "bounds"], "OBJECT_EXTRACT SCALE_TO_FIT_FRAME"),
        ("scale_per_color", "Scale per color family", "Resize objects differently by color.", "Apply distinct scale factors based on object color roles.", ["scale by color", "different resize per color", "palette conditioned scale"], ["scale", "color", "conditional", "objects"], "OBJECT_LABEL SCALE_PER_COLOR"),
        ("translate_to_center", "Translate to center", "Move content to the grid center.", "Compute the grid center and reposition the object or pattern there.", ["move to center", "center object", "reposition middle"], ["translate", "center", "middle", "object"], "OBJECT_CENTROID TRANSLATE_TO_CENTER"),
        ("normalize_orientation_by_anchor", "Normalize orientation by anchor", "Use an anchor to choose canonical orientation.", "A marker or asymmetric clue determines which way the object should face.", ["orientation chosen by marker", "anchor decides facing", "normalize by clue"], ["orientation", "anchor", "normalize", "marker"], "OBJECT_MASK NORMALIZE_ORIENTATION_BY_ANCHOR"),
        ("diagonal_stretch", "Diagonal stretch", "Stretch content along a diagonal axis.", "Project object cells outward along a diagonal growth direction.", ["stretch along diagonal", "diagonal growth", "expand on slant"], ["diagonal", "stretch", "expand", "growth"], "GRID DIAGONAL_STRETCH"),
        ("quadrant_replicate", "Quadrant replicate", "Copy a motif into quadrants.", "Take one quadrant or seed motif and replicate it into the other quadrants.", ["replicate to quadrants", "copy motif four ways", "fill missing quadrants"], ["quadrant", "replicate", "tile", "copy"], "GRID QUADRANT_REPLICATE"),
        ("crop_margin", "Crop margin", "Remove a known border margin.", "Crop away an outer margin of fixed thickness before solving the interior task.", ["remove outer border", "crop margin", "trim frame"], ["crop", "margin", "border", "trim"], "GRID CROP_MARGIN"),
        ("reframe_content", "Reframe content", "Place content inside a new frame.", "Wrap, pad, or relocate the pattern into a new bounding frame or canvas size.", ["put shape in new frame", "reframe content", "pad around object"], ["frame", "reframe", "pad", "canvas"], "GRID REFRAME_CONTENT"),
        ("reflect_about_center", "Reflect about center", "Reflect content through the grid center.", "Use the grid center as a pivot for a center-reflection or point-symmetry transform.", ["reflect about center", "point symmetry around center", "centered inversion"], ["reflect", "center", "point symmetry", "transform"], "GRID REFLECT_ABOUT_CENTER"),
    ]
    return _expand_specs(
        family_group="geometric_transform",
        transform_family="geometric_transform",
        items=items,
    )


def _color_pattern_specs() -> list[dict[str, Any]]:
    items = [
        ("color_histogram_global", "Global color histogram", "Count colors over the full grid.", "Measure the global palette distribution to detect dominant and rare colors.", ["global color histogram", "count colors in grid", "palette frequency"], ["color", "histogram", "palette", "count"], "GRID COLOR_HISTOGRAM"),
        ("color_histogram_region", "Regional color histogram", "Count colors inside a region.", "Measure the palette distribution of a selected region, subgrid, or object.", ["color histogram in region", "count palette in subgrid", "regional colors"], ["color", "histogram", "region", "palette"], "REGION COLOR_HISTOGRAM"),
        ("color_histogram_object", "Object color histogram", "Count colors inside each object.", "Measure the palette composition per object to detect symbolic multi-color motifs.", ["color histogram per object", "palette inside shape", "object colors"], ["color", "histogram", "object", "palette"], "OBJECT_MASK OBJECT_COLOR_HISTOGRAM"),
        ("majority_color_region", "Majority color in region", "Find the most frequent color in a region.", "Use color frequency to infer background, fill, or target replacement color.", ["most common color", "majority color region", "dominant palette"], ["majority", "dominant", "color", "region"], "REGION COLOR_MODE"),
        ("minority_color_region", "Minority color in region", "Find the rarest color in a region.", "Use rare colors as markers, anomalies, or control tokens.", ["rarest color", "minority color region", "marker color"], ["minority", "rare", "color", "region"], "REGION COLOR_MINORITY"),
        ("color_remap_by_rule", "Color remap by rule", "Apply a deterministic palette mapping.", "Replace colors according to a learned source-to-target mapping.", ["color remap rule", "recolor by mapping", "palette substitution"], ["color", "remap", "mapping", "recolor"], "GRID COLOR_REMAP_RULE"),
        ("color_remap_by_position", "Color remap by position", "Change colors based on cell position.", "Use row, column, border, or quadrant position to determine new colors.", ["color depends on position", "recolor by row column", "positional palette rule"], ["color", "position", "row", "column"], "GRID COLOR_REMAP_POSITION"),
        ("color_remap_by_context", "Color remap by context", "Change colors based on neighbors or objects.", "Use surrounding colors, object membership, or adjacency to recolor cells.", ["color depends on context", "neighbor based recolor", "context palette rule"], ["color", "context", "neighbor", "recolor"], "GRID COLOR_REMAP_CONTEXT"),
        ("palette_detection", "Palette detection", "Detect which colors the task uses.", "Enumerate the active palette and compare it to unused colors or target palettes.", ["which colors are used", "palette detection", "active colors"], ["palette", "detect", "active", "colors"], "GRID DETECT_PALETTE"),
        ("unused_color_detection", "Unused color detection", "Find colors absent from the task.", "Infer which palette slots from 0-9 remain unused and may be available as outputs.", ["unused colors", "missing palette entries", "which colors absent"], ["unused", "missing", "palette", "color"], "GRID UNUSED_COLORS"),
        ("dominant_border_color", "Dominant border color", "Read the border color as a cue.", "Use the frame color as a symbolic command for fill, crop, or action selection.", ["border color cue", "frame color command", "read outer border"], ["border", "frame", "color", "cue"], "GRID BORDER_COLOR_MODE"),
        ("color_as_count", "Color as count", "Interpret color values as counts.", "Use a color value to encode how many cells, steps, or objects to create.", ["color value means count", "palette encodes number", "count by color"], ["color", "count", "symbolic", "value"], "GRID COLOR_AS_COUNT"),
        ("color_as_pointer", "Color as pointer", "Interpret color as a pointer to another object.", "Use a marker color to choose a matching target color or object class.", ["color points to target", "palette as pointer", "color chooses object"], ["color", "pointer", "target", "symbolic"], "GRID COLOR_AS_POINTER"),
        ("color_as_target", "Color as target selector", "Interpret color as target selection.", "Use a color token to indicate which object, region, or output slot should change.", ["color chooses target", "target color marker", "palette selects object"], ["color", "target", "selector", "symbolic"], "GRID COLOR_AS_TARGET"),
        ("color_priority_ordering", "Color priority ordering", "Order colors by task-specific priority.", "Resolve conflicts by color rank, salience, or explicit palette ordering.", ["color priority", "which color wins", "palette ranking"], ["color", "priority", "ranking", "conflict"], "GRID COLOR_PRIORITY_ORDER"),
        ("color_swap_pair", "Color swap pair", "Swap a pair of colors.", "Exchange two role colors while leaving the rest of the palette untouched.", ["swap two colors", "exchange palette pair", "recolor pair"], ["color", "swap", "pair", "palette"], "GRID COLOR_SWAP_PAIR"),
        ("color_cycle", "Color cycle", "Rotate colors through a cycle.", "Advance colors along a learned palette cycle such as 1→2→3→1.", ["cycle colors", "rotate palette", "color ring"], ["color", "cycle", "palette", "rotation"], "GRID COLOR_CYCLE"),
        ("color_isolate", "Color isolate", "Keep only one target color.", "Suppress non-target colors so only the requested palette class remains visible.", ["keep one color", "isolate palette target", "filter by color"], ["color", "isolate", "filter", "target"], "GRID COLOR_ISOLATE"),
        ("erase_background_color", "Erase background color", "Remove the background color.", "Strip away background cells so only active shapes or signals remain.", ["remove background color", "erase empty palette", "drop dominant color"], ["background", "erase", "remove", "color"], "GRID ERASE_BACKGROUND_COLOR"),
        ("fill_with_missing_color", "Fill with missing color", "Use the missing palette color as fill.", "Infer the fill color from the palette slot that is absent from the input.", ["missing color fill", "use absent palette", "fill with unused color"], ["missing", "unused", "fill", "color"], "GRID FILL_WITH_MISSING_COLOR"),
        ("color_by_rank", "Color by rank", "Assign colors according to count rank.", "Map the first, second, or third ranked color counts to new actions or outputs.", ["ranked colors", "color by frequency order", "palette ranking rule"], ["color", "rank", "frequency", "mapping"], "GRID COLOR_BY_RANK"),
        ("periodic_horizontal", "Horizontal periodic pattern", "Detect repeating horizontal bands.", "Identify motifs that repeat along rows or horizontal stripes.", ["horizontal repeating pattern", "row periodicity", "stripe repeat"], ["pattern", "periodic", "horizontal", "rows"], "GRID DETECT_PERIODIC_HORIZONTAL"),
        ("periodic_vertical", "Vertical periodic pattern", "Detect repeating vertical bands.", "Identify motifs that repeat along columns or vertical stripes.", ["vertical repeating pattern", "column periodicity", "stripe repeat"], ["pattern", "periodic", "vertical", "columns"], "GRID DETECT_PERIODIC_VERTICAL"),
        ("periodic_two_dimensional", "2D periodic pattern", "Detect repeating 2D tiles.", "Identify motifs that tile in both axes to cover a larger region.", ["2d tile repeat", "periodic grid pattern", "motif tiles both ways"], ["pattern", "periodic", "2d", "tile"], "GRID DETECT_PERIODIC_2D"),
        ("checkerboard_pattern", "Checkerboard pattern", "Detect alternating checker cells.", "Recognize parity-based alternation across a rectangular region.", ["checkerboard", "alternating cells", "parity pattern"], ["checkerboard", "alternating", "parity", "pattern"], "GRID DETECT_CHECKERBOARD"),
        ("diagonal_pattern", "Diagonal pattern", "Detect diagonal repetition.", "Recognize motifs arranged along primary diagonals.", ["diagonal pattern", "staircase motif", "repeat on diagonal"], ["diagonal", "pattern", "motif", "staircase"], "GRID DETECT_DIAGONAL_PATTERN"),
        ("antidiagonal_pattern", "Anti-diagonal pattern", "Detect anti-diagonal repetition.", "Recognize motifs arranged along anti-diagonals.", ["anti diagonal pattern", "reverse staircase motif", "repeat on anti diagonal"], ["antidiagonal", "pattern", "motif", "staircase"], "GRID DETECT_ANTIDIAGONAL_PATTERN"),
        ("border_pattern", "Border pattern", "Detect border-only motifs.", "Recognize when the rule is encoded on the outer frame or perimeter.", ["border pattern", "frame motif", "perimeter encoding"], ["border", "frame", "perimeter", "pattern"], "GRID DETECT_BORDER_PATTERN"),
        ("spiral_pattern", "Spiral pattern", "Detect spiral ordering.", "Recognize cells or colors arranged in a clockwise or counterclockwise spiral.", ["spiral pattern", "clockwise ordering", "ring turns"], ["spiral", "clockwise", "ordering", "pattern"], "GRID DETECT_SPIRAL_PATTERN"),
        ("fractal_pattern", "Fractal self-similarity", "Detect self-similar subpatterns.", "Recognize when a region contains scaled copies of itself or a seed motif.", ["fractal pattern", "self similar motif", "pattern contains smaller copy"], ["fractal", "self similar", "recursive", "pattern"], "GRID DETECT_FRACTAL_PATTERN"),
        ("gradient_pattern", "Gradient pattern", "Detect monotonic color or value gradients.", "Recognize smooth or stepwise palette progression across space.", ["color gradient", "step gradient", "palette progression"], ["gradient", "progression", "color", "pattern"], "GRID DETECT_GRADIENT_PATTERN"),
        ("stripe_parity_pattern", "Stripe parity pattern", "Detect alternating stripe parity.", "Recognize even-odd row or column rules that alternate colors or actions.", ["alternating stripes", "row parity", "column parity"], ["stripe", "parity", "alternating", "pattern"], "GRID DETECT_STRIPE_PARITY"),
        ("ring_pattern", "Ring pattern", "Detect concentric ring motifs.", "Recognize nested rectangular or circular ring-like layers.", ["concentric rings", "nested border layers", "ring motif"], ["ring", "concentric", "nested", "pattern"], "GRID DETECT_RING_PATTERN"),
        ("corner_marker_pattern", "Corner marker pattern", "Detect meaningful corner markers.", "Read special colors or shapes placed in corners as task instructions.", ["corner markers", "special corners", "corner cues"], ["corner", "marker", "cue", "pattern"], "GRID DETECT_CORNER_MARKERS"),
        ("center_marker_pattern", "Center marker pattern", "Detect a meaningful center marker.", "Read the central cell or cluster as a symbolic command or pivot.", ["center marker", "middle cue", "central symbol"], ["center", "marker", "pivot", "pattern"], "GRID DETECT_CENTER_MARKER"),
        ("sparse_lattice_pattern", "Sparse lattice pattern", "Detect sparse repeated lattice points.", "Recognize evenly spaced sparse signals that define a scaffold or grid.", ["sparse lattice", "repeated anchor points", "grid scaffold"], ["lattice", "sparse", "anchor", "pattern"], "GRID DETECT_SPARSE_LATTICE"),
        ("repeated_tile_pattern", "Repeated tile motif", "Detect a repeated motif tile.", "Extract the tile that repeats to generate the larger scene.", ["repeated tile motif", "base tile extraction", "find motif tile"], ["tile", "repeat", "motif", "pattern"], "GRID EXTRACT_REPEATED_TILE"),
        ("mirror_motif_pattern", "Mirror motif pattern", "Detect mirrored motif pairs.", "Recognize paired motifs that are mirrored rather than copied verbatim.", ["mirrored motifs", "reflection motif pair", "paired mirrored shapes"], ["mirror", "motif", "pair", "pattern"], "GRID DETECT_MIRROR_MOTIF"),
        ("rotation_motif_pattern", "Rotation motif pattern", "Detect rotated motif families.", "Recognize motif copies generated by rotations rather than translations.", ["rotated motif copies", "same shape rotated", "rotation family"], ["rotate", "motif", "family", "pattern"], "GRID DETECT_ROTATION_MOTIF"),
        ("line_extension_pattern", "Line extension pattern", "Detect extendable lines.", "Recognize seed segments that should extend in a consistent direction.", ["extend line pattern", "continue segment", "grow line"], ["line", "extend", "segment", "pattern"], "GRID DETECT_LINE_EXTENSION"),
        ("alternating_band_pattern", "Alternating band pattern", "Detect alternating bands.", "Recognize repeated band regions whose color or shape alternates by index.", ["alternating bands", "band parity", "striped regions"], ["bands", "alternating", "stripe", "pattern"], "GRID DETECT_ALTERNATING_BANDS"),
        ("palette_complement", "Palette complement detection", "Detect colors missing from a local palette.", "Use complement colors relative to a palette subset to fill or recolor outputs.", ["palette complement", "colors missing from subset", "complementary palette"], ["palette", "complement", "missing", "color"], "GRID DETECT_PALETTE_COMPLEMENT"),
        ("palette_union", "Palette union", "Combine palettes from multiple regions.", "Take colors from multiple objects or examples and form a union palette.", ["palette union", "merge colors", "combine region palettes"], ["palette", "union", "merge", "color"], "GRID PALETTE_UNION"),
        ("palette_intersection", "Palette intersection", "Intersect palettes from multiple regions.", "Find which colors are shared across objects, examples, or frames.", ["palette intersection", "shared colors", "common palette"], ["palette", "intersection", "shared", "color"], "GRID PALETTE_INTERSECTION"),
        ("periodic_offset", "Periodic offset pattern", "Detect repeating motifs with a phase offset.", "Recognize motifs that repeat but shift phase every row or column.", ["shifted repeating pattern", "phase offset tile", "staggered repetition"], ["periodic", "offset", "phase", "pattern"], "GRID DETECT_PERIODIC_OFFSET"),
        ("nested_borders", "Nested border pattern", "Detect multiple nested borders.", "Recognize layered frames whose order or colors encode rules.", ["nested borders", "multiple frames", "layered border cue"], ["nested", "borders", "frames", "pattern"], "GRID DETECT_NESTED_BORDERS"),
        ("concentric_rings", "Concentric ring pattern", "Detect concentric growth rings.", "Recognize rings centered on a pivot with changing palette or thickness.", ["concentric rings", "centered layered frame", "ring growth"], ["concentric", "rings", "centered", "pattern"], "GRID DETECT_CONCENTRIC_RINGS"),
        ("zigzag_pattern", "Zigzag pattern", "Detect zigzag traversal or structure.", "Recognize motifs that progress in alternating diagonal directions.", ["zigzag pattern", "alternating diagonal steps", "sawtooth motif"], ["zigzag", "sawtooth", "alternating", "pattern"], "GRID DETECT_ZIGZAG_PATTERN"),
        ("step_gradient", "Step gradient pattern", "Detect discrete step gradients.", "Recognize palette changes that proceed in discrete staircase-like steps.", ["step gradient", "staircase palette progression", "discrete color gradient"], ["step", "gradient", "staircase", "pattern"], "GRID DETECT_STEP_GRADIENT"),
        ("color_adjacency_matrix", "Color adjacency matrix", "Measure which colors touch.", "Build a color-touch graph to reason about palette neighborhoods.", ["which colors touch", "color adjacency graph", "palette neighborhood"], ["color", "adjacency", "graph", "palette"], "GRID COLOR_ADJACENCY_MATRIX"),
        ("background_noise_template", "Background-noise template detection", "Recover a motif from noisy background.", "Separate stable motif structure from distractor background noise.", ["motif hidden in noise", "recover pattern from background", "template despite clutter"], ["noise", "template", "motif", "background"], "GRID TEMPLATE_FROM_NOISE"),
    ]
    return _expand_specs(
        family_group="color_pattern",
        transform_family="color_pattern",
        items=items,
    )


def _symbolic_specs() -> list[dict[str, Any]]:
    items = [
        ("marker_indicates_position", "Marker indicates position", "Use a marker to choose a location.", "Interpret a small marker object as the place where the main action must occur.", ["marker indicates position", "do action here", "target location marker"], ["marker", "position", "target", "symbolic"], "GRID MARKER_TO_POSITION"),
        ("marker_indicates_direction", "Marker indicates direction", "Use a marker to choose a direction.", "Interpret a marker's orientation or placement as a direction of motion or growth.", ["marker indicates direction", "arrow-like cue", "grow this way"], ["marker", "direction", "arrow", "symbolic"], "GRID MARKER_TO_DIRECTION"),
        ("marker_indicates_object", "Marker indicates object", "Use a marker to choose an object.", "Interpret a marker as selecting the object nearest to it or aligned with it.", ["marker points to object", "selected object by marker", "target shape cue"], ["marker", "object", "select", "symbolic"], "GRID MARKER_TO_OBJECT"),
        ("marker_indicates_group", "Marker indicates group", "Use a marker to choose an object group.", "Interpret a marker as selecting a whole group, row, column, or cluster.", ["marker chooses group", "select cluster by marker", "group cue"], ["marker", "group", "cluster", "symbolic"], "GRID MARKER_TO_GROUP"),
        ("marker_indicates_boundary", "Marker indicates boundary", "Use a marker to choose a border or limit.", "Interpret a marker as defining the boundary where fill or crop should stop.", ["marker defines boundary", "fill until marker", "crop by marker"], ["marker", "boundary", "limit", "symbolic"], "GRID MARKER_TO_BOUNDARY"),
        ("count_encodes_action", "Count encodes action", "Use object counts to choose the action.", "Interpret a count as which transform or output pattern should be used.", ["count chooses action", "number of objects means operation", "action by count"], ["count", "action", "symbolic", "rule"], "GRID COUNT_TO_ACTION"),
        ("count_encodes_color", "Count encodes color", "Use counts to choose a color.", "Interpret an object or hole count as the output color or recolor value.", ["count chooses color", "number means palette", "color by count"], ["count", "color", "symbolic", "rule"], "GRID COUNT_TO_COLOR"),
        ("count_encodes_size", "Count encodes size", "Use counts to choose a size.", "Interpret a count as how large the output object or pattern should be.", ["count chooses size", "number means scale", "size by count"], ["count", "size", "symbolic", "rule"], "GRID COUNT_TO_SIZE"),
        ("shape_encodes_rule", "Shape encodes rule", "Use object shape to choose the rule.", "Interpret whether an object is a square, line, frame, or cross as which action to take.", ["shape chooses rule", "object type means operation", "symbolic shape cue"], ["shape", "rule", "symbolic", "object"], "GRID SHAPE_TO_RULE"),
        ("shape_encodes_priority", "Shape encodes priority", "Use shape to choose precedence.", "Interpret one shape family as dominant when multiple rules compete.", ["shape chooses priority", "which object wins by shape", "dominant motif family"], ["shape", "priority", "dominant", "symbolic"], "GRID SHAPE_TO_PRIORITY"),
        ("shape_encodes_target", "Shape encodes target", "Use shape to choose target object.", "Interpret a control shape as specifying which other shape should change.", ["shape chooses target", "control shape selects object", "target by motif"], ["shape", "target", "control", "symbolic"], "GRID SHAPE_TO_TARGET"),
        ("color_encodes_target", "Color encodes target", "Use color to choose the target.", "Interpret control colors as selecting which object or region changes.", ["color encodes target", "palette chooses object", "target by color"], ["color", "target", "symbolic", "selector"], "GRID COLOR_TO_TARGET"),
        ("color_encodes_action", "Color encodes action", "Use color to choose the action.", "Interpret control colors as deciding which transform gets applied.", ["color encodes action", "palette chooses transform", "action by color"], ["color", "action", "symbolic", "selector"], "GRID COLOR_TO_ACTION"),
        ("color_encodes_boundary", "Color encodes boundary", "Use color to choose boundary semantics.", "Interpret a frame or separator color as defining where operations begin or end.", ["color encodes boundary", "frame color means limit", "border color command"], ["color", "boundary", "frame", "symbolic"], "GRID COLOR_TO_BOUNDARY"),
        ("hole_count_encodes_color", "Hole count encodes color", "Use hole count to choose color.", "Interpret the number of holes in a frame or shell as the output palette value.", ["holes choose color", "hole count means palette", "frame void count"], ["holes", "color", "count", "symbolic"], "GRID HOLE_COUNT_TO_COLOR"),
        ("hole_count_encodes_action", "Hole count encodes action", "Use hole count to choose action.", "Interpret a shell's number of holes as which transform or placement rule to apply.", ["holes choose action", "hole count means operation", "void count rule"], ["holes", "action", "count", "symbolic"], "GRID HOLE_COUNT_TO_ACTION"),
        ("border_color_encodes_action", "Border color encodes action", "Use border color to choose action.", "Interpret the outer frame color as which transform or fill rule should run.", ["border color chooses action", "frame command color", "outer border rule"], ["border", "color", "action", "symbolic"], "GRID BORDER_COLOR_TO_ACTION"),
        ("border_color_encodes_priority", "Border color encodes priority", "Use border color to choose precedence.", "Interpret the frame color as deciding which objects or rules outrank others.", ["border color priority", "frame decides winner", "outer border precedence"], ["border", "priority", "color", "symbolic"], "GRID BORDER_COLOR_TO_PRIORITY"),
        ("size_encodes_priority", "Size encodes priority", "Use object size to choose precedence.", "Interpret larger or smaller objects as more important control signals.", ["size means priority", "larger object wins", "small marker has lower priority"], ["size", "priority", "symbolic", "object"], "GRID SIZE_TO_PRIORITY"),
        ("size_encodes_action", "Size encodes action", "Use object size to choose action.", "Interpret an object's size as mapping to a transform family or count.", ["size chooses action", "object area means transform", "action by size"], ["size", "action", "symbolic", "object"], "GRID SIZE_TO_ACTION"),
        ("example_defines_mapping", "Example defines mapping", "Learn a lookup table from examples.", "Infer a direct mapping from input motifs to output motifs by comparing training pairs.", ["example defines mapping", "learn lookup from examples", "training pair mapping"], ["example", "mapping", "lookup", "generalize"], "EXAMPLE_PAIR BUILD_MAPPING"),
        ("input_output_delta", "Input-output delta", "Focus on the change between input and output.", "Treat the difference between train input and output as the reusable rule itself.", ["input output delta", "what changed", "difference map rule"], ["delta", "difference", "change", "rule"], "EXAMPLE_PAIR DELTA_MAP"),
        ("invariant_detection", "Invariant detection", "Find what stays the same across examples.", "Detect structural invariants such as preserved objects, counts, or palette relations.", ["what stays same", "invariant across examples", "preserved structure"], ["invariant", "preserve", "same", "examples"], "EXAMPLE_SET DETECT_INVARIANTS"),
        ("variable_detection", "Variable detection", "Find what changes across examples.", "Detect the varying dimensions such as color, count, position, or orientation.", ["what changes", "variable across examples", "changing parameter"], ["variable", "change", "parameter", "examples"], "EXAMPLE_SET DETECT_VARIABLES"),
        ("rule_generalization", "Rule generalization", "Abstract the rule from examples.", "Generalize the smallest rule that explains all train examples.", ["generalize rule", "abstract from examples", "common transformation"], ["generalize", "abstract", "rule", "examples"], "EXAMPLE_SET GENERALIZE_RULE"),
        ("rule_composition", "Rule composition", "Compose multiple simple rules.", "Infer that two or more primitives must be chained together to explain the examples.", ["compose multiple rules", "two step example rule", "chained transformation"], ["compose", "chain", "multiple", "rule"], "EXAMPLE_SET COMPOSE_RULES"),
        ("symbolic_substitution", "Symbolic substitution", "Replace a symbol with a learned meaning.", "Treat one motif as a symbolic token that must be substituted by another structure.", ["symbolic substitution", "motif means other motif", "replace symbol"], ["symbolic", "substitute", "replace", "motif"], "EXAMPLE_SET SYMBOLIC_SUBSTITUTE"),
        ("reference_exemplar_retrieval", "Reference exemplar retrieval", "Retrieve the exemplar controlling the action.", "Use the closest training exemplar or control object as a prototype for the output.", ["retrieve exemplar", "reference example", "prototype object"], ["reference", "exemplar", "prototype", "symbolic"], "EXAMPLE_SET RETRIEVE_EXEMPLAR"),
        ("exception_marker", "Exception marker", "Detect symbols that signal an exception.", "Treat one object, color, or frame as a do-not-follow-the-main-rule exception.", ["exception marker", "special case symbol", "rule exception cue"], ["exception", "marker", "override", "symbolic"], "EXAMPLE_SET DETECT_EXCEPTION_MARKER"),
        ("context_switch_by_marker", "Context switch by marker", "Switch rules when a marker appears.", "Interpret one marker as changing which rule family applies in that region.", ["marker switches context", "different rule by cue", "context sensitive symbol"], ["context", "switch", "marker", "rule"], "EXAMPLE_SET CONTEXT_SWITCH_MARKER"),
        ("multi_example_vote", "Multi-example vote", "Use multiple examples to vote on the rule.", "Resolve ambiguity by letting repeated train examples reinforce one mapping.", ["majority across examples", "example vote", "repeated training signal"], ["vote", "examples", "majority", "rule"], "EXAMPLE_SET MULTI_EXAMPLE_VOTE"),
        ("training_example_alignment", "Training example alignment", "Align corresponding train examples.", "Match objects or regions across train pairs before extracting the rule.", ["align train examples", "corresponding regions", "example matching"], ["align", "examples", "correspond", "rule"], "EXAMPLE_SET ALIGN_TRAIN_EXAMPLES"),
        ("latent_slot_binding", "Latent slot binding", "Bind role slots across examples.", "Treat objects as role slots that preserve meaning even when their palette changes.", ["role slots across examples", "bind object roles", "latent slots"], ["slot", "bind", "role", "symbolic"], "EXAMPLE_SET BIND_ROLE_SLOTS"),
        ("symbol_role_assignment", "Symbol role assignment", "Assign roles to control symbols.", "Determine whether a symbol acts as a source, target, count, or boundary cue.", ["assign symbol role", "what does this marker mean", "control token role"], ["symbol", "role", "assignment", "control"], "EXAMPLE_SET ASSIGN_SYMBOL_ROLE"),
        ("analogy_transfer", "Analogy transfer", "Transfer the same relation to new objects.", "Apply the relation seen in one pair of objects to another analogous pair.", ["analogy transfer", "same relation new objects", "by analogy"], ["analogy", "transfer", "relation", "symbolic"], "EXAMPLE_SET ANALOGY_TRANSFER"),
        ("symbolic_override_by_frame", "Symbolic override by frame", "Use a frame to override the main rule.", "Treat a containing frame or border as changing the local semantics of enclosed objects.", ["frame overrides rule", "enclosed region different meaning", "context by border"], ["frame", "override", "context", "symbolic"], "EXAMPLE_SET FRAME_OVERRIDE"),
        ("symbolic_negation", "Symbolic negation", "Interpret a symbol as negating the default action.", "Treat one cue as meaning do the opposite, remove, or suppress the usual rule.", ["symbolic negation", "opposite action cue", "cancel default rule"], ["negation", "cancel", "opposite", "symbolic"], "EXAMPLE_SET SYMBOLIC_NEGATION"),
        ("marker_means_noop", "Marker means no-op", "Interpret a marker as keep unchanged.", "Treat one cue as explicitly instructing the system to preserve content.", ["marker means keep same", "no action cue", "leave unchanged marker"], ["noop", "preserve", "marker", "symbolic"], "EXAMPLE_SET MARKER_MEANS_NOOP"),
        ("example_chooses_transform_family", "Example chooses transform family", "Use examples to choose between transform families.", "Infer whether the task is about recolor, movement, extraction, tiling, or symmetry.", ["examples choose transform family", "which primitive family fits", "family selection"], ["family", "transform", "choose", "examples"], "EXAMPLE_SET CHOOSE_TRANSFORM_FAMILY"),
        ("context_dependent_color_semantics", "Context-dependent color semantics", "Same color means different things in different contexts.", "Interpret the same palette value differently depending on region, role, or example pairing.", ["same color different meaning", "context dependent color", "palette semantics switch"], ["context", "color", "semantics", "symbolic"], "EXAMPLE_SET CONTEXT_COLOR_SEMANTICS"),
        ("count_chooses_rotation", "Count chooses rotation", "Use a count to choose rotation amount.", "Interpret counts as quarter turns or rotation family selection.", ["count chooses rotation", "number means turn amount", "rotation by count"], ["count", "rotation", "quarter turn", "symbolic"], "GRID COUNT_TO_ROTATION"),
        ("count_chooses_translation", "Count chooses translation", "Use a count to choose movement distance.", "Interpret counts as dx, dy, or number of growth steps.", ["count chooses movement", "distance by count", "translation by number"], ["count", "translation", "distance", "symbolic"], "GRID COUNT_TO_TRANSLATION"),
        ("orientation_encodes_order", "Orientation encodes order", "Use orientation to choose processing order.", "Interpret which way an arrow or object points as the order of operations.", ["orientation means order", "pointing direction decides sequence", "order by arrow"], ["orientation", "order", "sequence", "symbolic"], "GRID ORIENTATION_TO_ORDER"),
        ("nearest_marker_chooses_target", "Nearest marker chooses target", "Nearest marker selects the target object.", "Resolve target choice using nearest-marker proximity.", ["nearest marker chooses object", "closest cue selects target", "target by nearest"], ["nearest", "marker", "target", "symbolic"], "GRID NEAREST_MARKER_TARGET"),
        ("farthest_marker_chooses_target", "Farthest marker chooses target", "Farthest marker selects the target object.", "Resolve target choice using farthest-marker or global relation.", ["farthest marker chooses object", "furthest cue selects target", "target by farthest"], ["farthest", "marker", "target", "symbolic"], "GRID FARTHEST_MARKER_TARGET"),
        ("hole_position_encodes_direction", "Hole position encodes direction", "Use hole placement to infer direction.", "A hole on one side of a frame indicates where growth or motion should occur.", ["hole position means direction", "frame opening points way", "direction from hole"], ["hole", "direction", "frame", "symbolic"], "GRID HOLE_POSITION_TO_DIRECTION"),
        ("blank_space_encodes_keepout", "Blank space encodes keep-out region", "Use empty space as a no-go area.", "Treat intentionally blank regions as symbolic keep-out or preserve zones.", ["blank space means do not fill", "keep out region", "empty area cue"], ["blank", "empty", "keepout", "symbolic"], "GRID BLANK_SPACE_KEEPOUT"),
        ("border_gap_encodes_opening", "Border gap encodes opening", "Use a border gap as an opening cue.", "Treat a break in a frame as indicating where a path, fill, or object should exit.", ["gap in border means opening", "frame break indicates path", "opening cue"], ["border", "gap", "opening", "symbolic"], "GRID BORDER_GAP_TO_OPENING"),
        ("exemplar_ranking_by_similarity", "Exemplar ranking by similarity", "Choose the best example by similarity.", "Rank train exemplars and apply the most similar transformation trace.", ["pick nearest example", "best matching exemplar", "similarity chooses mapping"], ["exemplar", "similarity", "rank", "symbolic"], "EXAMPLE_SET EXEMPLAR_RANK_SIMILARITY"),
        ("train_example_majority_rule", "Train example majority rule", "Use the majority rule across examples.", "When train pairs disagree locally, follow the most repeated relation.", ["majority rule from training examples", "most common mapping", "vote over train pairs"], ["majority", "examples", "vote", "symbolic"], "EXAMPLE_SET MAJORITY_RULE"),
    ]
    return _expand_specs(
        family_group="symbolic_interpretation",
        transform_family="symbolic_interpretation",
        items=items,
    )


def _spatial_specs() -> list[dict[str, Any]]:
    items = [
        ("grid_subdivision_2x2", "Grid subdivision 2x2", "Divide the grid into four regions.", "Split the grid into a 2x2 arrangement of equal or nearly equal regions.", ["divide grid into quadrants", "2 by 2 subgrids", "split into four"], ["grid", "subdivision", "2x2", "quadrants"], "GRID SUBDIVIDE_2X2"),
        ("grid_subdivision_3x3", "Grid subdivision 3x3", "Divide the grid into nine regions.", "Split the grid into a 3x3 arrangement of equal or nearly equal regions.", ["divide grid into 3 by 3", "nine subgrids", "split into nine"], ["grid", "subdivision", "3x3", "regions"], "GRID SUBDIVIDE_3X3"),
        ("grid_region_by_lines", "Grid region by lines", "Use separator lines to divide the grid.", "Detect drawn separators and treat them as region boundaries.", ["separator lines divide grid", "regions by drawn lines", "split by barrier"], ["separator", "regions", "lines", "grid"], "GRID REGIONS_BY_LINES"),
        ("grid_overlay", "Grid overlay", "Overlay two grids or layers.", "Combine two aligned grids using an overlay priority rule.", ["overlay grids", "combine layers", "superimpose patterns"], ["overlay", "combine", "layers", "grid"], "GRID OVERLAY"),
        ("grid_concat_horizontal", "Horizontal grid concatenation", "Join grids horizontally.", "Place two or more subgrids side by side in output order.", ["join grids left right", "horizontal concatenation", "side by side"], ["concat", "horizontal", "join", "grid"], "GRID CONCAT_HORIZONTAL"),
        ("grid_concat_vertical", "Vertical grid concatenation", "Join grids vertically.", "Place two or more subgrids top to bottom in output order.", ["join grids top bottom", "vertical concatenation", "stack grids"], ["concat", "vertical", "join", "grid"], "GRID CONCAT_VERTICAL"),
        ("grid_interleave_horizontal", "Horizontal interleave", "Interleave columns from multiple grids.", "Weave source grids column by column into one output.", ["interleave columns", "weave grids horizontally", "column by column"], ["interleave", "horizontal", "columns", "grid"], "GRID INTERLEAVE_HORIZONTAL"),
        ("grid_interleave_vertical", "Vertical interleave", "Interleave rows from multiple grids.", "Weave source grids row by row into one output.", ["interleave rows", "weave grids vertically", "row by row"], ["interleave", "vertical", "rows", "grid"], "GRID INTERLEAVE_VERTICAL"),
        ("grid_difference", "Grid difference", "Keep cells that differ.", "Construct the difference map between two grids or stages.", ["grid difference", "what changed between grids", "difference map"], ["difference", "delta", "grid", "compare"], "GRID DIFFERENCE"),
        ("grid_intersection", "Grid intersection", "Keep cells that agree.", "Construct the overlap of cells or motifs shared by multiple grids.", ["grid intersection", "shared cells", "common overlap"], ["intersection", "shared", "overlap", "grid"], "GRID INTERSECTION"),
        ("grid_xor", "Grid exclusive difference", "Keep cells present in only one source.", "Construct an exclusive-or map over two candidate layers.", ["grid xor", "exclusive difference", "one or the other cells"], ["xor", "exclusive", "difference", "grid"], "GRID XOR"),
        ("grid_union", "Grid union", "Keep cells present in any source.", "Construct a union map over multiple sources or layers.", ["grid union", "combine any active cell", "merge layers"], ["union", "merge", "layers", "grid"], "GRID UNION"),
        ("above_below_relation", "Above-below relation", "Reason about vertical ordering.", "Detect when one object is above or below another object.", ["above below objects", "vertical relation", "object over under"], ["above", "below", "vertical", "relation"], "OBJECTS RELATION_ABOVE_BELOW"),
        ("left_right_relation", "Left-right relation", "Reason about horizontal ordering.", "Detect when one object is left or right of another object.", ["left right objects", "horizontal relation", "object beside"], ["left", "right", "horizontal", "relation"], "OBJECTS RELATION_LEFT_RIGHT"),
        ("inside_outside_relation", "Inside-outside relation", "Reason about enclosure.", "Detect whether an object or color lies inside or outside another object or frame.", ["inside outside relation", "enclosed object", "frame interior"], ["inside", "outside", "enclosure", "relation"], "OBJECTS RELATION_INSIDE_OUTSIDE"),
        ("between_relation", "Between relation", "Reason about one object between two others.", "Detect when an object lies spatially between two anchors.", ["object between two others", "middle relation", "between anchors"], ["between", "middle", "relation", "objects"], "OBJECTS RELATION_BETWEEN"),
        ("nearest_neighbor_relation", "Nearest-neighbor relation", "Reason about nearest objects.", "Find the closest object to a target object or marker.", ["nearest object", "closest neighbor", "which shape is nearest"], ["nearest", "neighbor", "distance", "relation"], "OBJECTS NEAREST_NEIGHBOR"),
        ("farthest_neighbor_relation", "Farthest-neighbor relation", "Reason about farthest objects.", "Find the farthest object relative to a target or marker.", ["farthest object", "furthest neighbor", "most distant shape"], ["farthest", "distance", "neighbor", "relation"], "OBJECTS FARTHEST_NEIGHBOR"),
        ("path_between", "Path between objects", "Reason about paths connecting objects.", "Find or build a path between source and destination anchors.", ["path between objects", "connect source to target", "route through grid"], ["path", "connect", "route", "objects"], "GRID PATH_BETWEEN"),
        ("line_of_sight", "Line-of-sight relation", "Reason about unobstructed straight paths.", "Test whether two points can see each other along a straight line without obstacles.", ["line of sight", "unobstructed line", "straight connection"], ["line of sight", "straight", "visibility", "relation"], "GRID LINE_OF_SIGHT"),
        ("gravity_fall", "Gravity simulation", "Let objects fall toward the bottom.", "Move unsupported objects downward until they rest on another object or the border.", ["gravity fall", "drop objects", "objects fall down"], ["gravity", "fall", "support", "simulation"], "GRID GRAVITY_FALL"),
        ("flood_expansion", "Flood expansion", "Expand a region until it hits boundaries.", "Grow a region outward subject to blockers or palette rules.", ["expand region until blocked", "flood expansion", "grow fill"], ["flood", "expand", "grow", "region"], "GRID FLOOD_EXPANSION"),
        ("attract_to_center", "Attract to center", "Move objects toward the center.", "Shift objects inward toward a central pivot or centroid.", ["move toward center", "attract to middle", "center pull"], ["attract", "center", "move", "spatial"], "GRID ATTRACT_TO_CENTER"),
        ("repel_from_border", "Repel from border", "Move objects away from the border.", "Shift objects inward or outward based on distance from the frame.", ["move away from border", "repel from frame", "border push"], ["repel", "border", "frame", "spatial"], "GRID REPEL_FROM_BORDER"),
        ("pack_to_corner", "Pack to corner", "Pack objects into a corner.", "Collect objects toward a canonical corner while preserving order.", ["pack to corner", "collect shapes in corner", "move to top left"], ["pack", "corner", "collect", "spatial"], "GRID PACK_TO_CORNER"),
        ("spread_evenly", "Spread evenly", "Distribute objects with even spacing.", "Place objects at regular intervals across a row, column, or frame.", ["spread evenly", "regular spacing", "distribute shapes"], ["spread", "spacing", "regular", "spatial"], "GRID SPREAD_EVENLY"),
        ("align_to_guide_line", "Align to guide line", "Snap objects to a guide line.", "Use a visible guide or separator to align objects.", ["align to line", "snap objects to guide", "use separator as guide"], ["align", "guide", "line", "spatial"], "GRID ALIGN_TO_GUIDE_LINE"),
        ("project_to_border", "Project to border", "Extend features until they hit the border.", "Project lines, colors, or objects outward toward a frame edge.", ["project to border", "extend until frame", "grow toward edge"], ["project", "border", "extend", "spatial"], "GRID PROJECT_TO_BORDER"),
        ("bridge_across_gap", "Bridge across gap", "Connect two regions across a gap.", "Build a shortest or straight bridge across empty space or separators.", ["bridge across gap", "connect separated regions", "span empty cells"], ["bridge", "gap", "connect", "spatial"], "GRID BRIDGE_ACROSS_GAP"),
        ("follow_path", "Follow path", "Move or draw along a discovered path.", "Follow an existing line, trail, or corridor through the grid.", ["follow path", "trace corridor", "move along line"], ["follow", "path", "trace", "spatial"], "GRID FOLLOW_PATH"),
        ("fill_corridor", "Fill corridor", "Fill a path-shaped corridor.", "Fill the interior of a corridor or tunnel bounded by walls.", ["fill corridor", "color tunnel", "path interior fill"], ["fill", "corridor", "tunnel", "spatial"], "GRID FILL_CORRIDOR"),
        ("expand_until_obstacle", "Expand until obstacle", "Grow until hitting a blocker.", "Apply growth repeatedly until a wall, object, or palette constraint stops it.", ["grow until obstacle", "expand until blocked", "fill until wall"], ["expand", "obstacle", "blocked", "spatial"], "GRID EXPAND_UNTIL_OBSTACLE"),
        ("count_objects_spatial", "Count objects", "Count spatially distinct objects.", "Count objects as a spatial measurement primitive.", ["count spatial objects", "how many shapes in grid", "object tally"], ["count", "objects", "spatial", "measure"], "OBJECT_LABEL OBJECT_COUNT"),
        ("count_cells_per_color", "Count cells per color", "Count how many cells each color occupies.", "Build a per-color occupancy histogram over the grid or region.", ["cells per color", "count palette occupancy", "color cell histogram"], ["count", "cells", "color", "measure"], "GRID COUNT_CELLS_PER_COLOR"),
        ("measure_distance_manhattan", "Measure Manhattan distance", "Use grid-step distance between points.", "Measure the number of orthogonal steps between anchors.", ["manhattan distance", "grid step distance", "taxicab distance"], ["distance", "manhattan", "grid", "measure"], "POINTS DISTANCE_MANHATTAN"),
        ("measure_distance_chebyshev", "Measure Chebyshev distance", "Use king-move distance between points.", "Measure the number of diagonal-or-orthogonal steps between anchors.", ["chebyshev distance", "king move distance", "max axis distance"], ["distance", "chebyshev", "grid", "measure"], "POINTS DISTANCE_CHEBYSHEV"),
        ("measure_area", "Measure area", "Count cells in a region.", "Measure the filled area of a region or object.", ["measure area", "count region cells", "shape area"], ["area", "cells", "measure", "region"], "REGION MEASURE_AREA"),
        ("measure_perimeter", "Measure perimeter", "Count boundary cells.", "Measure the perimeter or outline length of an object.", ["measure perimeter", "boundary count", "outline length"], ["perimeter", "boundary", "measure", "object"], "REGION MEASURE_PERIMETER"),
        ("measure_aspect_ratio", "Measure aspect ratio", "Measure width-to-height proportion.", "Compute the aspect ratio of a bounding box or region.", ["measure aspect ratio", "width height proportion", "object shape proportion"], ["aspect", "ratio", "measure", "object"], "REGION MEASURE_ASPECT_RATIO"),
        ("measure_width_height_span", "Measure width-height span", "Measure bounding extents.", "Compute width and height span as separate values.", ["measure width and height", "bounding extents", "span of object"], ["width", "height", "span", "measure"], "REGION MEASURE_WIDTH_HEIGHT"),
        ("measure_connected_path_length", "Measure connected path length", "Count cells along a path.", "Measure the length of a corridor, bridge, or traced line.", ["path length", "corridor length", "line segment count"], ["path", "length", "measure", "spatial"], "PATH MEASURE_LENGTH"),
        ("quadrant_occupancy", "Quadrant occupancy", "Measure which quadrants contain content.", "Treat occupancy in each quadrant as a feature or symbolic signal.", ["which quadrants filled", "quadrant occupancy", "content by quadrant"], ["quadrant", "occupancy", "content", "spatial"], "GRID QUADRANT_OCCUPANCY"),
        ("row_occupancy_profile", "Row occupancy profile", "Measure filled cells per row.", "Build a row-wise profile describing where content appears.", ["row occupancy profile", "cells per row", "horizontal density"], ["row", "occupancy", "profile", "spatial"], "GRID ROW_OCCUPANCY_PROFILE"),
        ("column_occupancy_profile", "Column occupancy profile", "Measure filled cells per column.", "Build a column-wise profile describing where content appears.", ["column occupancy profile", "cells per column", "vertical density"], ["column", "occupancy", "profile", "spatial"], "GRID COLUMN_OCCUPANCY_PROFILE"),
        ("shortest_bridge", "Shortest bridge", "Build the minimum connection across a gap.", "Find the minimum connector between two regions.", ["shortest bridge", "minimum connection", "closest join"], ["shortest", "bridge", "connect", "spatial"], "GRID SHORTEST_BRIDGE"),
        ("corridor_width", "Corridor width", "Measure corridor thickness.", "Measure how wide a path or tunnel is between boundaries.", ["corridor width", "tunnel thickness", "passage size"], ["corridor", "width", "tunnel", "measure"], "GRID CORRIDOR_WIDTH"),
        ("enclosure_depth", "Enclosure depth", "Measure how deeply nested a cell or object is.", "Count how many border layers enclose a point or object.", ["enclosure depth", "nested border levels", "how deep inside"], ["enclosure", "depth", "nested", "measure"], "GRID ENCLOSURE_DEPTH"),
        ("nearest_border", "Nearest border", "Find the nearest border or frame edge.", "Measure which frame edge is closest to the object or marker.", ["nearest border", "closest frame edge", "edge proximity"], ["nearest", "border", "edge", "spatial"], "GRID NEAREST_BORDER"),
        ("farthest_border", "Farthest border", "Find the farthest border or frame edge.", "Measure which frame edge is farthest from the object or marker.", ["farthest border", "most distant edge", "edge distance"], ["farthest", "border", "edge", "spatial"], "GRID FARTHEST_BORDER"),
        ("anchor_relative_coordinate", "Anchor-relative coordinate", "Express locations relative to an anchor.", "Represent object positions in coordinates centered on a marker or pivot.", ["coordinates relative to marker", "anchor based position", "relative location"], ["anchor", "relative", "coordinate", "spatial"], "GRID ANCHOR_RELATIVE_COORDINATE"),
        ("subgrid_repetition_origin", "Subgrid repetition origin", "Find where a repeated subgrid starts.", "Infer the origin or phase of a repeating subgrid pattern.", ["where repeated subgrid starts", "motif origin", "repetition phase"], ["subgrid", "origin", "phase", "spatial"], "GRID SUBGRID_REPETITION_ORIGIN"),
    ]
    return _expand_specs(
        family_group="spatial_reasoning",
        transform_family="spatial_reasoning",
        items=items,
    )


def _meta_specs() -> list[dict[str, Any]]:
    items = [
        ("one_to_one_mapping", "One-to-one mapping rule", "Infer a one-to-one mapping.", "Assume each input control state maps to exactly one output action or motif.", ["one to one mapping", "unique output per input", "deterministic mapping"], ["one to one", "mapping", "deterministic", "meta"], "EXAMPLE_SET RULE_ONE_TO_ONE"),
        ("one_to_many_mapping", "One-to-many expansion rule", "Infer an expansion from one input to many outputs.", "Assume one seed object or signal expands into multiple output instances.", ["one input to many outputs", "expansion rule", "seed replicates"], ["one to many", "expansion", "mapping", "meta"], "EXAMPLE_SET RULE_ONE_TO_MANY"),
        ("many_to_one_mapping", "Many-to-one reduction rule", "Infer a reduction from many inputs to one output.", "Assume multiple objects collapse into one representative or summary output.", ["many inputs one output", "reduction rule", "merge to one"], ["many to one", "reduction", "mapping", "meta"], "EXAMPLE_SET RULE_MANY_TO_ONE"),
        ("conditional_rule", "Conditional rule", "Infer if-then branching.", "Infer that different objects or contexts trigger different transforms.", ["if then rule", "conditional transform", "branching logic"], ["conditional", "if then", "branch", "meta"], "EXAMPLE_SET RULE_CONDITIONAL"),
        ("exception_rule", "Exception rule", "Infer a general rule with exceptions.", "Infer that one or more outliers override the dominant mapping.", ["general rule plus exception", "special case", "outlier override"], ["exception", "special case", "override", "meta"], "EXAMPLE_SET RULE_EXCEPTION"),
        ("priority_rule", "Priority rule", "Infer priority among competing rules.", "Infer which rule should win when multiple transformations are possible.", ["priority among rules", "which rule wins", "precedence"], ["priority", "precedence", "competing", "meta"], "EXAMPLE_SET RULE_PRIORITY"),
        ("recursive_rule", "Recursive rule", "Infer repeated application of a rule.", "Infer that the same transform should be applied iteratively until a stop condition.", ["recursive rule", "repeat until stable", "iterative transform"], ["recursive", "iterative", "repeat", "meta"], "EXAMPLE_SET RULE_RECURSIVE"),
        ("boundary_rule", "Boundary rule", "Infer special behavior on the boundary.", "Infer that edges, borders, or corners obey a different rule than interior cells.", ["boundary rule", "edge behavior differs", "interior versus border"], ["boundary", "edge", "interior", "meta"], "EXAMPLE_SET RULE_BOUNDARY"),
        ("stateful_rule", "Stateful rule", "Infer dependence on intermediate state.", "Infer that the next step depends on previously generated output state.", ["stateful rule", "depends on prior step", "intermediate state"], ["stateful", "memory", "sequence", "meta"], "EXAMPLE_SET RULE_STATEFUL"),
        ("noop_rule", "No-op rule", "Infer that nothing should change.", "Infer that preserving the input exactly is the correct output for some cases.", ["no op rule", "output same as input", "identity case"], ["noop", "identity", "preserve", "meta"], "EXAMPLE_SET RULE_NOOP"),
        ("output_same_size", "Output same size", "Infer output matches input dimensions.", "Infer that the output grid should preserve the input size.", ["output same size as input", "preserve dimensions", "same grid shape"], ["output", "same size", "dimensions", "meta"], "EXAMPLE_SET OUTPUT_SAME_SIZE"),
        ("output_fixed_size", "Output fixed size", "Infer output size is fixed regardless of input.", "Infer that a constant NxM output canvas is required.", ["output fixed size", "constant dimensions", "same canvas every example"], ["output", "fixed size", "constant", "meta"], "EXAMPLE_SET OUTPUT_FIXED_SIZE"),
        ("output_derived_size", "Output derived size", "Infer output size from content.", "Infer that counts, bounds, or object spans determine output dimensions.", ["output size derived", "size from content", "dimensions from objects"], ["output", "derived size", "content", "meta"], "EXAMPLE_SET OUTPUT_DERIVED_SIZE"),
        ("output_subgrid", "Output is a subgrid", "Infer output crops the input.", "Infer that the result is a selected crop or extracted region.", ["output is crop", "subgrid output", "extract region"], ["output", "subgrid", "crop", "meta"], "EXAMPLE_SET OUTPUT_SUBGRID"),
        ("output_supergrid", "Output is a supergrid", "Infer output expands beyond input.", "Infer that the result is a tiled, padded, or enlarged version of the input.", ["output larger than input", "supergrid", "expand canvas"], ["output", "supergrid", "expand", "meta"], "EXAMPLE_SET OUTPUT_SUPERGRID"),
        ("output_overlay", "Output is an overlay", "Infer output combines multiple layers.", "Infer that multiple intermediate products must be overlaid into the final result.", ["output overlay", "combine layers", "merge intermediate outputs"], ["output", "overlay", "combine", "meta"], "EXAMPLE_SET OUTPUT_OVERLAY"),
        ("output_tiled", "Output is tiled", "Infer output repeats a motif.", "Infer that the final result is generated by repeating a smaller motif.", ["output tiled", "repeat motif output", "pattern filling canvas"], ["output", "tile", "repeat", "meta"], "EXAMPLE_SET OUTPUT_TILED"),
        ("output_cropped", "Output is cropped", "Infer output is trimmed content.", "Infer that borders or irrelevant context are removed from the result.", ["output cropped", "trimmed result", "remove borders"], ["output", "crop", "trim", "meta"], "EXAMPLE_SET OUTPUT_CROPPED"),
        ("output_summary_grid", "Output summary grid", "Infer output summarizes the input.", "Infer that the output is a reduced symbolic summary instead of a direct transform.", ["output summary grid", "reduced symbolic output", "legend style result"], ["summary", "reduced", "symbolic", "meta"], "EXAMPLE_SET OUTPUT_SUMMARY"),
        ("output_legend_grid", "Output legend grid", "Infer output acts like a legend or key.", "Infer that the result re-encodes input structure into a legend-like layout.", ["output legend", "key grid", "encoded summary"], ["legend", "key", "summary", "meta"], "EXAMPLE_SET OUTPUT_LEGEND"),
        ("compare_examples_first", "Compare examples first", "Use example comparison before solving.", "Infer rules by contrasting train pairs before touching the test grid.", ["compare examples first", "learn from train pairs", "difference between examples"], ["compare", "examples", "first", "meta"], "EXAMPLE_SET STRATEGY_COMPARE_EXAMPLES"),
        ("difference_map_first", "Difference-map first", "Start from the input-output delta.", "Infer rules by computing change maps before choosing primitives.", ["difference map first", "start from what changed", "delta before solve"], ["difference", "delta", "strategy", "meta"], "EXAMPLE_SET STRATEGY_DELTA_FIRST"),
        ("invariant_preservation_first", "Invariant-preservation first", "Start from preserved structure.", "Infer rules by identifying what must stay unchanged.", ["start from invariant", "preserved structure first", "what stays same"], ["invariant", "preserve", "strategy", "meta"], "EXAMPLE_SET STRATEGY_INVARIANT_FIRST"),
        ("object_budget_first", "Object-budget first", "Track object counts before geometry.", "Use how many objects survive, split, or merge as a primary cue.", ["object budget", "count objects first", "object conservation"], ["object", "budget", "count", "meta"], "EXAMPLE_SET STRATEGY_OBJECT_BUDGET"),
        ("color_budget_first", "Color-budget first", "Track palette counts before geometry.", "Use palette conservation or shift as a primary cue.", ["color budget", "palette conservation", "track colors first"], ["color", "budget", "palette", "meta"], "EXAMPLE_SET STRATEGY_COLOR_BUDGET"),
        ("symmetry_budget_first", "Symmetry-budget first", "Track symmetry before action.", "Use preserved or broken symmetry as a leading clue.", ["symmetry budget", "check symmetry first", "preserved mirror"], ["symmetry", "budget", "mirror", "meta"], "EXAMPLE_SET STRATEGY_SYMMETRY_BUDGET"),
        ("composition_order", "Composition-order rule", "Infer the order of multi-step rules.", "Infer whether recolor happens before move, crop before tile, or vice versa.", ["order of operations", "which rule comes first", "composition order"], ["composition", "order", "multi step", "meta"], "EXAMPLE_SET STRATEGY_COMPOSITION_ORDER"),
        ("rule_ranking", "Rule ranking", "Rank candidate rules before execution.", "Score plausible rule families before committing to one.", ["rank candidate rules", "best transform family", "rule scoring"], ["rule", "rank", "candidate", "meta"], "EXAMPLE_SET STRATEGY_RULE_RANKING"),
        ("solve_by_elimination", "Solve by elimination", "Reject impossible rules early.", "Narrow the search by eliminating families that violate examples.", ["eliminate impossible rules", "narrow candidate set", "reject bad transforms"], ["elimination", "reject", "candidate", "meta"], "EXAMPLE_SET STRATEGY_ELIMINATION"),
        ("solve_by_sanity_check", "Solve by sanity check", "Validate candidates against obvious constraints.", "Use size, palette, object count, and boundary sanity checks before finalizing.", ["sanity check candidate", "validate obvious constraints", "cheap consistency checks"], ["sanity", "check", "constraints", "meta"], "EXAMPLE_SET STRATEGY_SANITY_CHECK"),
        ("simplest_rule_preference", "Simplest-rule preference", "Prefer the simplest rule that fits.", "When several rules explain the examples, choose the least complex one.", ["prefer simplest rule", "minimum complexity mapping", "simpler explanation"], ["simplest", "preference", "complexity", "meta"], "EXAMPLE_SET STRATEGY_SIMPLEST_RULE"),
        ("conservative_rule_preference", "Conservative-rule preference", "Prefer the least destructive rule.", "When several rules fit, prefer the one that preserves the most structure.", ["prefer conservative rule", "least destructive transform", "preserve more structure"], ["conservative", "preserve", "preference", "meta"], "EXAMPLE_SET STRATEGY_CONSERVATIVE"),
        ("composition_depth_detection", "Composition-depth detection", "Infer how many steps are needed.", "Estimate whether the task needs one, two, or many chained primitives.", ["how many steps", "composition depth", "single or multi step"], ["composition", "depth", "steps", "meta"], "EXAMPLE_SET STRATEGY_COMPOSITION_DEPTH"),
        ("candidate_against_all_examples", "Candidate-against-all-examples check", "Check every candidate against all train pairs.", "Reject candidates that fit only a subset of the training evidence.", ["candidate must fit all examples", "validate against every train pair", "global consistency"], ["candidate", "all examples", "consistency", "meta"], "EXAMPLE_SET STRATEGY_ALL_EXAMPLES"),
        ("exact_object_conservation", "Exact-object conservation", "Prefer rules conserving object identities exactly.", "Use exact object preservation as a strong prior when train pairs support it.", ["exact object conservation", "preserve identities", "objects survive unchanged"], ["object", "conservation", "exact", "meta"], "EXAMPLE_SET STRATEGY_OBJECT_CONSERVATION"),
    ]
    return _expand_specs(
        family_group="meta_reasoning",
        transform_family="meta_reasoning",
        items=items,
    )


def _interactive_specs() -> list[dict[str, Any]]:
    items = [
        ("enumerate_actions", "Enumerate actions", "List plausible next actions.", "Generate a controlled frontier of candidate ARC actions instead of jumping to one guess.", ["enumerate actions", "list candidate moves", "interactive exploration"], ["interactive", "enumerate", "actions", "arcagi3"], "ARC_INTERACTIVE ENUMERATE_ACTIONS"),
        ("probe_one_object", "Probe one object at a time", "Test hypotheses object by object.", "Interactively inspect one object, then update the hypothesis before moving on.", ["probe one object", "inspect shape by shape", "interactive object scan"], ["interactive", "probe", "object", "arcagi3"], "ARC_INTERACTIVE PROBE_ONE_OBJECT"),
        ("compare_candidate_outputs", "Compare candidate outputs", "Score multiple candidate outputs.", "Keep several candidate outputs alive and compare them against train constraints.", ["compare candidate outputs", "multiple hypotheses", "interactive output ranking"], ["interactive", "candidates", "compare", "arcagi3"], "ARC_INTERACTIVE COMPARE_CANDIDATES"),
        ("scratch_memory", "Scratch memory", "Keep temporary symbolic notes.", "Maintain a scratchpad of counts, object roles, and tested hypotheses.", ["scratch memory", "temporary notes", "interactive working memory"], ["interactive", "scratch", "memory", "arcagi3"], "ARC_INTERACTIVE SCRATCH_MEMORY"),
        ("hypothesis_update", "Hypothesis update", "Update the active rule after each probe.", "Revise the current rule based on what the last probe confirmed or falsified.", ["update hypothesis", "revise rule after probe", "interactive refinement"], ["interactive", "hypothesis", "update", "arcagi3"], "ARC_INTERACTIVE UPDATE_HYPOTHESIS"),
        ("state_diff_tracking", "State-diff tracking", "Track state changes between actions.", "Measure what changed after each exploratory action or transform.", ["track state changes", "diff after action", "interactive state delta"], ["interactive", "state", "diff", "arcagi3"], "ARC_INTERACTIVE TRACK_STATE_DIFF"),
        ("reward_from_invariant", "Reward from invariant", "Reward actions that preserve invariants.", "Bias exploration toward actions that keep observed train invariants intact.", ["reward invariant preserving action", "prefer stable constraints", "interactive invariant guidance"], ["interactive", "reward", "invariant", "arcagi3"], "ARC_INTERACTIVE REWARD_INVARIANT"),
        ("ask_what_changed", "Ask what changed", "Explicitly query the output delta.", "Use change-focused inspection as a first interactive question.", ["what changed", "interactive delta query", "output difference question"], ["interactive", "change", "delta", "arcagi3"], "ARC_INTERACTIVE ASK_WHAT_CHANGED"),
        ("ask_what_stayed_same", "Ask what stayed same", "Explicitly query preserved structure.", "Use invariant-focused inspection as an interactive question.", ["what stayed same", "interactive invariant query", "preserved structure question"], ["interactive", "invariant", "preserve", "arcagi3"], "ARC_INTERACTIVE ASK_WHAT_STAYED_SAME"),
        ("store_failed_attempts", "Store failed attempts", "Remember failed candidate rules.", "Use failed transforms as evidence so the same bad branch is not retried.", ["remember failed attempts", "do not retry bad rule", "interactive negative memory"], ["interactive", "failed", "memory", "arcagi3"], "ARC_INTERACTIVE STORE_FAILED_ATTEMPTS"),
        ("cell_probe", "Cell probe", "Inspect one cell or small patch.", "Interactively query a cell, patch, or coordinate for its role.", ["probe cell", "inspect one coordinate", "interactive patch query"], ["interactive", "cell", "probe", "arcagi3"], "ARC_INTERACTIVE CELL_PROBE"),
        ("region_probe", "Region probe", "Inspect one region at a time.", "Interactively test a selected subgrid or object region.", ["probe region", "inspect subgrid", "interactive region query"], ["interactive", "region", "probe", "arcagi3"], "ARC_INTERACTIVE REGION_PROBE"),
        ("object_selection_memory", "Object-selection memory", "Remember which object was selected.", "Persist the currently selected object so later actions refer to it coherently.", ["remember selected object", "interactive object memory", "selection persistence"], ["interactive", "selection", "memory", "arcagi3"], "ARC_INTERACTIVE OBJECT_SELECTION_MEMORY"),
        ("action_rollback", "Action rollback", "Undo an exploratory action.", "Support reversible exploration by rolling back a failed transform.", ["rollback action", "undo probe", "interactive reversible step"], ["interactive", "rollback", "undo", "arcagi3"], "ARC_INTERACTIVE ACTION_ROLLBACK"),
        ("action_replay", "Action replay", "Replay a promising action sequence.", "Reapply a successful action trace to another object or region.", ["replay action sequence", "reuse successful steps", "interactive trace replay"], ["interactive", "replay", "trace", "arcagi3"], "ARC_INTERACTIVE ACTION_REPLAY"),
        ("cursor_follow_object", "Cursor follows object", "Keep the active cursor attached to an object.", "Track the object of interest as it moves or transforms during exploration.", ["cursor follows object", "track active shape", "interactive target lock"], ["interactive", "cursor", "track", "arcagi3"], "ARC_INTERACTIVE CURSOR_FOLLOW_OBJECT"),
        ("exploration_frontier", "Exploration frontier", "Maintain a frontier of unexplored actions.", "Track what branches remain to test and which are exhausted.", ["exploration frontier", "remaining branches", "interactive search frontier"], ["interactive", "frontier", "search", "arcagi3"], "ARC_INTERACTIVE EXPLORATION_FRONTIER"),
        ("candidate_branching", "Candidate branching", "Split exploration into branches.", "Allow several rule branches to progress in parallel until evidence separates them.", ["candidate branching", "parallel hypotheses", "interactive branch search"], ["interactive", "branch", "parallel", "arcagi3"], "ARC_INTERACTIVE CANDIDATE_BRANCHING"),
        ("compress_discovered_rule", "Compress discovered rule", "Summarize a discovered rule compactly.", "Reduce a multi-step exploration trace into a compact reusable rule.", ["compress discovered rule", "summarize exploration", "compact hypothesis"], ["interactive", "compress", "rule", "arcagi3"], "ARC_INTERACTIVE COMPRESS_RULE"),
        ("stop_when_stable", "Stop when stable", "Halt when the rule stabilizes.", "End exploration when new probes no longer change the hypothesis.", ["stop when stable", "interactive halting", "no more change in rule"], ["interactive", "stable", "halt", "arcagi3"], "ARC_INTERACTIVE STOP_WHEN_STABLE"),
        ("uncertainty_monitor", "Uncertainty monitor", "Track remaining uncertainty.", "Measure whether the current hypothesis is still ambiguous.", ["monitor uncertainty", "how unsure is current rule", "interactive ambiguity"], ["interactive", "uncertainty", "ambiguity", "arcagi3"], "ARC_INTERACTIVE UNCERTAINTY_MONITOR"),
        ("trigger_refinement_round", "Trigger refinement round", "Launch a refinement round when needed.", "Escalate to a second exploration pass when the first pass yields conflicting evidence.", ["trigger refinement", "second exploration pass", "interactive re-evaluation"], ["interactive", "refine", "second pass", "arcagi3"], "ARC_INTERACTIVE TRIGGER_REFINEMENT"),
        ("preserve_partial_observations", "Preserve partial observations", "Keep useful partial findings.", "Retain partial detections such as likely axes or target objects even before the full rule is known.", ["preserve partial findings", "keep useful clues", "interactive partial memory"], ["interactive", "partial", "observations", "arcagi3"], "ARC_INTERACTIVE PRESERVE_PARTIALS"),
        ("merge_worker_hypotheses", "Merge worker hypotheses", "Combine partial hypotheses from different probes.", "Fuse compatible partial rules instead of forcing a single winner too early.", ["merge partial hypotheses", "combine worker findings", "interactive fusion"], ["interactive", "merge", "hypotheses", "arcagi3"], "ARC_INTERACTIVE MERGE_HYPOTHESES"),
        ("confirm_reversible_action", "Confirm with reversible action", "Test a reversible step before committing.", "Prefer exploratory actions that can be undone and checked safely.", ["reversible action first", "safe exploratory step", "confirm before commit"], ["interactive", "reversible", "confirm", "arcagi3"], "ARC_INTERACTIVE CONFIRM_REVERSIBLE"),
    ]
    return _expand_specs(
        family_group="interactive_strategy",
        transform_family="arc_agi_3_interactive",
        items=items,
        category="arc_interactive_strategy",
        layer=3,
        confidence=0.89,
    )


def build_arc_anchor_catalog() -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    catalog.extend(_object_centric_specs())
    catalog.extend(_geometric_specs())
    catalog.extend(_color_pattern_specs())
    catalog.extend(_symbolic_specs())
    catalog.extend(_spatial_specs())
    catalog.extend(_meta_specs())
    catalog.extend(_interactive_specs())
    return catalog


def build_arc_language_symlink_entries() -> list[dict[str, Any]]:
    return [_arc_language_symlink_entry(spec) for spec in build_arc_anchor_catalog()]


def build_arc_anchor_entries() -> list[dict[str, Any]]:
    return [_arc_anchor_entry(spec) for spec in build_arc_anchor_catalog()]


def build_all_arc_entries() -> dict[str, list[dict[str, Any]]]:
    return {
        "Drawing": build_arc_anchor_entries(),
        "Language": build_arc_language_symlink_entries(),
    }


def _is_arc_related_entry(entry: dict[str, Any]) -> bool:
    if not isinstance(entry, dict):
        return False
    entry_id = str(entry.get("id", "")).strip().lower()
    category = str(entry.get("category", "")).strip().lower()
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    subject = str(metadata.get("subject", "")).strip().lower()
    symlink_target = str(entry.get("symlink_to") or metadata.get("symlink_target") or "").strip().lower()
    return (
        entry_id.startswith("arc_")
        or entry_id.startswith("lang_arc_symlink_")
        or category.startswith("arc_")
        or subject == "arc_transform"
        or symlink_target.startswith("arc_")
    )


def ingest_arc_knowledge(
    knowledgeverse: Knowledgeverse,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    manager = knowledgeverse.galaxy_manager
    anchor_specs = build_arc_anchor_catalog()
    all_entries = build_all_arc_entries()
    drawing_entries = list(all_entries["Drawing"])
    language_entries = list(all_entries["Language"])

    counts_by_group: dict[str, int] = {}
    for spec in anchor_specs:
        group = str(spec.get("family_group", "")).strip() or "unknown"
        counts_by_group[group] = int(counts_by_group.get(group, 0)) + 1

    inserted = 0
    updated = 0
    with manager.bulk_disk_sync():
        for galaxy_name, entries in (("Drawing", drawing_entries), ("Language", language_entries)):
            for entry in entries:
                status = manager.upsert_entry(galaxy_name, entry)
                if status == "inserted":
                    inserted += 1
                else:
                    updated += 1
            if progress is not None:
                progress(f"ARC {galaxy_name}: {len(entries)} entries staged")

    arc_related_total = 0
    galaxy_names = {str(path.stem).strip() for path in manager.storage_root.glob("*.jsonl")}
    galaxy_names.update(str(name).strip() for name in knowledgeverse.DEFAULT_GALAXIES)
    galaxy_names.add("Language")
    for galaxy_name in sorted(name for name in galaxy_names if name):
        galaxy = manager.get_galaxy(galaxy_name)
        arc_related_total += sum(1 for entry in galaxy.entries if _is_arc_related_entry(entry))

    summary = {
        "anchor_entries": len(drawing_entries),
        "language_symlinks": len(language_entries),
        "total_entries": len(drawing_entries) + len(language_entries),
        "inserted": int(inserted),
        "updated": int(updated),
        "counts_by_group": dict(sorted(counts_by_group.items())),
        "arc_related_total": int(arc_related_total),
    }
    if progress is not None:
        progress(
            "ARC knowledge: "
            f"{summary['anchor_entries']} anchors + {summary['language_symlinks']} language bridges "
            f"({summary['arc_related_total']} ARC-related entries across galaxies)"
        )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, default=DEFAULT_STORAGE_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    knowledgeverse = Knowledgeverse(storage_root=args.storage_root, eager_load_default_galaxies=False)
    knowledgeverse.ensure_default_galaxies_loaded()
    summary = ingest_arc_knowledge(knowledgeverse, progress=lambda message: print(message, flush=True))
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
