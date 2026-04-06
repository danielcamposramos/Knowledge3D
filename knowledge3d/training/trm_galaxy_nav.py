from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_GALAXY_ORDER: tuple[str, ...] = (
    "Drawing",
    "Character",
    "Word",
    "Number",
    "Grammar",
    "Math",
    "Reality",
    "Audio",
    "3DObjects",
    "Tool",
)

TRM_INIT_SEED = 314159
TRM_WEIGHT_SHAPES: dict[str, tuple[int, int]] = {
    "W1": (1024, 512),
    "W2": (512, 1024),
    "W3": (1024, 512),
    "W4": (512, 1024),
}
CHECKPOINT_METADATA_DTYPE = "<U65535"


def trace_target_distribution(
    python_galaxies: Iterable[str],
    *,
    galaxy_contribution: dict[str, Any] | None = None,
    galaxy_order: tuple[str, ...] = DEFAULT_GALAXY_ORDER,
) -> np.ndarray:
    if isinstance(galaxy_contribution, dict):
        weighted = np.zeros(len(galaxy_order), dtype=np.float32)
        total = 0.0
        for idx, galaxy_name in enumerate(galaxy_order):
            try:
                value = float(galaxy_contribution.get(galaxy_name, 0.0))
            except Exception:
                value = 0.0
            if value <= 0.0:
                continue
            weighted[idx] = np.float32(value)
            total += float(value)
        if total > 0.0:
            return weighted / np.float32(total)
    target = np.zeros(len(galaxy_order), dtype=np.float32)
    python_names = {str(g) for g in python_galaxies}
    indexes = [idx for idx, name in enumerate(galaxy_order) if str(name) in python_names]
    if not indexes:
        return target
    value = np.float32(1.0 / float(len(indexes)))
    for idx in indexes:
        target[idx] = value
    return target


def trace_target_logits(
    python_galaxies: Iterable[str],
    *,
    galaxy_contribution: dict[str, Any] | None = None,
    galaxy_idf: dict[str, float] | None = None,
    target_blend_alpha: float = 1.0,
    galaxy_order: tuple[str, ...] = DEFAULT_GALAXY_ORDER,
) -> np.ndarray:
    distribution = trace_target_distribution(
        python_galaxies,
        galaxy_contribution=galaxy_contribution,
        galaxy_order=galaxy_order,
    )
    if not np.any(distribution > 0.0):
        return np.zeros(len(galaxy_order), dtype=np.float32)
    logits = np.full(len(galaxy_order), -1.0, dtype=np.float32)
    positive_count = int(np.count_nonzero(distribution > 0.0))
    positive_value = np.float32(max(1.0, (len(galaxy_order) - positive_count) / float(max(1, positive_count))))
    positive_indexes = np.flatnonzero(distribution > 0.0)
    if isinstance(galaxy_contribution, dict) and any(float(galaxy_contribution.get(name, 0.0)) > 0.0 for name in galaxy_order):
        scale = np.float32(positive_value * max(positive_count, 1))
        for idx in positive_indexes:
            logits[int(idx)] = scale * distribution[int(idx)]
        return logits
    blend_alpha = np.float32(float(np.clip(target_blend_alpha, 0.0, 1.0)))
    idf_scores = np.zeros(len(galaxy_order), dtype=np.float32)
    for idx in positive_indexes:
        galaxy_name = str(galaxy_order[int(idx)])
        idf_weight = np.float32(float(galaxy_idf.get(galaxy_name, 1.0))) if galaxy_idf is not None else np.float32(1.0)
        idf_scores[int(idx)] = max(idf_weight, np.float32(0.0))
    idf_sum = float(np.sum(idf_scores[positive_indexes]))
    if idf_sum <= 0.0:
        idf_distribution = distribution.astype(np.float32, copy=False)
    else:
        idf_distribution = idf_scores / np.float32(idf_sum)
    blended_distribution = (
        (blend_alpha * idf_distribution)
        + ((np.float32(1.0) - blend_alpha) * distribution.astype(np.float32, copy=False))
    ).astype(np.float32, copy=False)
    scale = np.float32(positive_value * max(positive_count, 1))
    for idx in positive_indexes:
        logits[int(idx)] = scale * blended_distribution[int(idx)]
    return logits


def softmax(logits: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(logits), dtype=np.float32).reshape(-1)
    if values.size == 0:
        return values
    shifted = values - float(np.max(values))
    weights = np.exp(shifted, dtype=np.float32)
    denom = float(np.sum(weights))
    if not np.isfinite(denom) or denom <= 0.0:
        return np.full(values.shape, 1.0 / float(values.size), dtype=np.float32)
    return weights / np.float32(denom)


def apply_galaxy_decoder(y_vector_512: Iterable[float], decoder: dict[str, Any]) -> np.ndarray:
    vector = np.asarray(list(y_vector_512), dtype=np.float32).reshape(-1)
    weights = np.asarray(decoder["W_galaxy"], dtype=np.float32)
    bias = np.asarray(decoder.get("b_galaxy", np.zeros(weights.shape[0], dtype=np.float32)), dtype=np.float32)
    return (weights @ vector) + bias


def initialize_trm_weight_matrices(seed: int = TRM_INIT_SEED) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights: dict[str, np.ndarray] = {}
    for name, shape in TRM_WEIGHT_SHAPES.items():
        host = (rng.standard_normal(shape, dtype=np.float32) * np.float32(0.02)).astype(
            np.float32,
            copy=False,
        )
        weights[name] = host
    return weights


def _swish(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    sigma = 1.0 / (1.0 + np.exp(-clipped, dtype=np.float32))
    return values * sigma


def _swish_grad(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    sigma = 1.0 / (1.0 + np.exp(-clipped, dtype=np.float32))
    return sigma + values * sigma * (1.0 - sigma)


def trm_forward_numpy(
    q_vectors: np.ndarray,
    weights: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    q_batch = np.asarray(q_vectors, dtype=np.float32)
    squeeze = False
    if q_batch.ndim == 1:
        q_batch = q_batch.reshape(1, -1)
        squeeze = True
    w1 = np.asarray(weights["W1"], dtype=np.float32)
    w2 = np.asarray(weights["W2"], dtype=np.float32)
    w3 = np.asarray(weights["W3"], dtype=np.float32)
    w4 = np.asarray(weights["W4"], dtype=np.float32)
    hidden1_pre = q_batch @ w1.T
    hidden1 = _swish(hidden1_pre)
    z_new = hidden1 @ w2.T
    hidden2_pre = z_new @ w3.T
    hidden2 = _swish(hidden2_pre)
    y_new = hidden2 @ w4.T
    cache = {
        "q": q_batch,
        "hidden1_pre": hidden1_pre,
        "hidden1": hidden1,
        "z_new": z_new,
        "hidden2_pre": hidden2_pre,
        "hidden2": hidden2,
    }
    if squeeze:
        return y_new.reshape(-1), {name: value.reshape(-1) if value.shape[0] == 1 else value for name, value in cache.items()}
    return y_new, cache


def evaluate_trm_weights_on_traces(
    traces: list[dict[str, Any]],
    weights: dict[str, Any],
    decoder: dict[str, Any] | None = None,
    *,
    galaxy_order: tuple[str, ...] = DEFAULT_GALAXY_ORDER,
) -> dict[str, float]:
    usable = [
        trace
        for trace in traces
        if len(trace.get("query_embedding_512", [])) == 512 and trace.get("python_galaxies")
    ]
    if not usable:
        return {
            "trace_count": 0.0,
            "raw_avg_entropy": 0.0,
            "raw_top1_match_rate": 0.0,
            "decoder_avg_entropy": 0.0,
            "decoder_top1_match_rate": 0.0,
        }
    queries = np.asarray([trace["query_embedding_512"] for trace in usable], dtype=np.float32)
    y_new, _ = trm_forward_numpy(queries, weights)
    entropies_raw: list[float] = []
    raw_matches = 0
    entropies_decoder: list[float] = []
    decoder_matches = 0
    for index, trace in enumerate(usable):
        target_names = {str(g) for g in trace.get("python_galaxies", [])}
        raw_logits = np.asarray(y_new[index][: len(galaxy_order)], dtype=np.float32)
        raw_distribution = softmax(raw_logits)
        entropies_raw.append(float(-np.sum(raw_distribution * np.log(np.clip(raw_distribution, 1e-9, 1.0)))))
        raw_top1 = str(galaxy_order[int(np.argmax(raw_distribution))]) if raw_distribution.size else ""
        if raw_top1 and raw_top1 in target_names:
            raw_matches += 1
        if decoder is not None:
            decoded_logits = apply_galaxy_decoder(y_new[index], decoder)
            decoded_distribution = softmax(decoded_logits)
            entropies_decoder.append(
                float(-np.sum(decoded_distribution * np.log(np.clip(decoded_distribution, 1e-9, 1.0))))
            )
            decoded_top1 = str(galaxy_order[int(np.argmax(decoded_distribution))]) if decoded_distribution.size else ""
            if decoded_top1 and decoded_top1 in target_names:
                decoder_matches += 1
    count = float(len(usable))
    return {
        "trace_count": count,
        "raw_avg_entropy": float(sum(entropies_raw) / count),
        "raw_top1_match_rate": float(raw_matches / count),
        "decoder_avg_entropy": float(sum(entropies_decoder) / count) if entropies_decoder else 0.0,
        "decoder_top1_match_rate": float(decoder_matches / count) if entropies_decoder else 0.0,
    }


def apply_trm_weights_to_traces(
    traces: list[dict[str, Any]],
    weights: dict[str, Any],
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for trace in traces:
        clone = dict(trace)
        query = np.asarray(list(trace.get("query_embedding_512", [])), dtype=np.float32).reshape(-1)
        if query.size == 512:
            y_new, _ = trm_forward_numpy(query, weights)
            clone["y_new_vector_512"] = np.asarray(y_new, dtype=np.float32).reshape(-1).tolist()
        updated.append(clone)
    return updated


def _canonical_target_key(
    python_galaxies: Iterable[str],
    *,
    galaxy_order: tuple[str, ...] = DEFAULT_GALAXY_ORDER,
) -> tuple[str, ...]:
    wanted = {str(g) for g in python_galaxies}
    ordered = [name for name in galaxy_order if name in wanted]
    extras = sorted(wanted.difference(ordered))
    return tuple(ordered + extras)


def build_trace_balance_weights(
    traces: list[dict[str, Any]],
    *,
    galaxy_order: tuple[str, ...] = DEFAULT_GALAXY_ORDER,
) -> np.ndarray:
    usable = [
        trace
        for trace in traces
        if len(trace.get("query_embedding_512", [])) == 512 and trace.get("python_galaxies")
    ]
    if not usable:
        return np.zeros(0, dtype=np.float32)
    counts = Counter(
        _canonical_target_key(trace.get("python_galaxies", []), galaxy_order=galaxy_order)
        for trace in usable
    )
    raw = np.asarray(
        [
            1.0 / float(max(counts[_canonical_target_key(trace.get("python_galaxies", []), galaxy_order=galaxy_order)], 1))
            for trace in usable
        ],
        dtype=np.float32,
    )
    total = float(np.sum(raw))
    if total <= 0.0:
        return np.ones(len(usable), dtype=np.float32)
    return raw * np.float32(len(usable) / total)


def compute_galaxy_idf(
    traces: list[dict[str, Any]],
    *,
    galaxy_order: tuple[str, ...] = DEFAULT_GALAXY_ORDER,
) -> dict[str, float]:
    task_type_galaxies: dict[str, set[str]] = defaultdict(set)
    usable = [
        trace
        for trace in traces
        if trace.get("python_galaxies")
    ]
    for trace in usable:
        task_type = str(trace.get("task_type") or trace.get("benchmark") or "unknown").strip() or "unknown"
        for galaxy_name in trace.get("python_galaxies", []):
            task_type_galaxies[task_type].add(str(galaxy_name))
    task_type_count = max(len(task_type_galaxies), 1)
    idf: dict[str, float] = {}
    for galaxy_name in galaxy_order:
        doc_freq = sum(1 for galaxies in task_type_galaxies.values() if galaxy_name in galaxies)
        idf[galaxy_name] = float(np.log(float(task_type_count) / float(max(doc_freq, 1))) + 1.0)
    return idf


def summarize_trace_top1_predictions(
    traces: list[dict[str, Any]],
    decoder: dict[str, Any] | None = None,
    *,
    weights: dict[str, Any] | None = None,
    galaxy_order: tuple[str, ...] = DEFAULT_GALAXY_ORDER,
) -> dict[str, Any]:
    usable = [
        trace
        for trace in traces
        if (
            len(trace.get("query_embedding_512", [])) == 512
            or len(trace.get("y_new_vector_512", [])) == 512
        )
        and trace.get("python_galaxies")
    ]
    rows: list[dict[str, Any]] = []
    grouped: dict[str, dict[str, Any]] = {}
    for trace in usable:
        benchmark = str(trace.get("benchmark") or "unknown")
        query = np.asarray(list(trace.get("query_embedding_512", [])), dtype=np.float32).reshape(-1)
        if weights is not None and query.size == 512:
            y_vector, _ = trm_forward_numpy(query, weights)
            y_host = np.asarray(y_vector, dtype=np.float32).reshape(-1)
        else:
            y_host = np.asarray(list(trace.get("y_new_vector_512", [])), dtype=np.float32).reshape(-1)
        if y_host.size < len(galaxy_order):
            continue
        logits = apply_galaxy_decoder(y_host, decoder) if decoder is not None else y_host[: len(galaxy_order)]
        distribution = softmax(logits)
        predicted_top1 = str(galaxy_order[int(np.argmax(distribution))]) if distribution.size else ""
        drawing_index = next((idx for idx, name in enumerate(galaxy_order) if name == "Drawing"), 0)
        top3_indexes = np.argsort(distribution)[::-1][:3]
        target = [str(g) for g in trace.get("python_galaxies", [])]
        row = {
            "benchmark": benchmark,
            "task_type": str(trace.get("task_type") or ""),
            "task_id": str(trace.get("task_id") or ""),
            "predicted_top1": predicted_top1,
            "target": target,
            "match": predicted_top1 in set(target),
            "drawing_weight": float(distribution[drawing_index]) if distribution.size > drawing_index else 0.0,
            "top3": [
                {
                    "galaxy": str(galaxy_order[int(idx)]),
                    "weight": float(distribution[int(idx)]),
                }
                for idx in top3_indexes
            ],
        }
        rows.append(row)
        bucket = grouped.setdefault(
            benchmark,
            {
                "total": 0,
                "correct": 0,
                "rows": [],
                "drawing_weight_sum": 0.0,
                "drawing_weight_min": None,
                "drawing_above_0_05": 0,
            },
        )
        bucket["total"] += 1
        bucket["correct"] += int(row["match"])
        bucket["rows"].append(row)
        bucket["drawing_weight_sum"] += float(row["drawing_weight"])
        bucket["drawing_above_0_05"] += int(float(row["drawing_weight"]) > 0.05)
        current_min = bucket["drawing_weight_min"]
        bucket["drawing_weight_min"] = (
            float(row["drawing_weight"])
            if current_min is None
            else min(float(current_min), float(row["drawing_weight"]))
        )
    for bucket in grouped.values():
        total = int(bucket["total"])
        bucket["top1_match_rate"] = float(bucket["correct"] / total) if total else 0.0
        bucket["drawing_weight_avg"] = float(bucket["drawing_weight_sum"] / total) if total else 0.0
        del bucket["drawing_weight_sum"]
    return {
        "rows": rows,
        "per_benchmark": grouped,
    }


def summarize_trace_target_contributions(
    traces: list[dict[str, Any]],
    *,
    galaxy_order: tuple[str, ...] = DEFAULT_GALAXY_ORDER,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for trace in traces:
        benchmark = str(trace.get("benchmark") or "unknown")
        contribution = trace_target_distribution(
            trace.get("python_galaxies", []),
            galaxy_contribution=(
                trace.get("galaxy_contribution")
                if isinstance(trace.get("galaxy_contribution"), dict)
                else None
            ),
            galaxy_order=galaxy_order,
        )
        if contribution.size == 0 or not np.any(contribution > 0.0):
            continue
        bucket = grouped.setdefault(
            benchmark,
            {
                "count": 0,
                "sum": np.zeros(len(galaxy_order), dtype=np.float32),
                "examples": [],
            },
        )
        bucket["count"] += 1
        bucket["sum"] = bucket["sum"] + contribution
        if len(bucket["examples"]) < 3:
            top_indexes = np.argsort(contribution)[::-1][:3]
            bucket["examples"].append(
                {
                    "task_id": str(trace.get("task_id", "")).strip(),
                    "top3": [
                        {
                            "galaxy": str(galaxy_order[int(idx)]),
                            "weight": float(contribution[int(idx)]),
                        }
                        for idx in top_indexes
                        if float(contribution[int(idx)]) > 0.0
                    ],
                }
            )
    summary: dict[str, Any] = {}
    for benchmark, bucket in grouped.items():
        count = max(int(bucket["count"]), 1)
        avg = np.asarray(bucket["sum"], dtype=np.float32) / np.float32(count)
        top_indexes = np.argsort(avg)[::-1][:5]
        summary[benchmark] = {
            "count": int(bucket["count"]),
            "avg_top5": [
                {
                    "galaxy": str(galaxy_order[int(idx)]),
                    "weight": float(avg[int(idx)]),
                }
                for idx in top_indexes
                if float(avg[int(idx)]) > 0.0
            ],
            "examples": list(bucket["examples"]),
        }
    return summary


def train_trm_weights_from_traces(
    traces: list[dict[str, Any]],
    decoder: dict[str, Any],
    *,
    initial_weights: dict[str, Any] | None = None,
    epochs: int = 600,
    learning_rate: float = 1e-3,
    clip_norm: float = 1.0,
    raw_signal_weight: float = 1.0,
    arc_drawing_bonus_weight: float = 0.0,
    arc_drawing_margin: float = 0.0,
    target_blend_alpha: float = 1.0,
    galaxy_order: tuple[str, ...] = DEFAULT_GALAXY_ORDER,
) -> dict[str, Any]:
    usable = [
        trace
        for trace in traces
        if len(trace.get("query_embedding_512", [])) == 512 and trace.get("python_galaxies")
    ]
    if not usable:
        raise ValueError("No usable TRM training traces found")
    q_batch = np.asarray([trace["query_embedding_512"] for trace in usable], dtype=np.float32)
    galaxy_idf = compute_galaxy_idf(usable, galaxy_order=galaxy_order)
    target_logits = np.asarray(
        [
            trace_target_logits(
                trace.get("python_galaxies", []),
                galaxy_contribution=(
                    trace.get("galaxy_contribution")
                    if isinstance(trace.get("galaxy_contribution"), dict)
                    else None
                ),
                galaxy_idf=galaxy_idf,
                target_blend_alpha=target_blend_alpha,
                galaxy_order=galaxy_order,
            )
            for trace in usable
        ],
        dtype=np.float32,
    )
    weights = {
        name: np.asarray(value, dtype=np.float32).copy()
        for name, value in (initial_weights or initialize_trm_weight_matrices()).items()
    }
    w_galaxy = np.asarray(decoder["W_galaxy"], dtype=np.float32)
    b_galaxy = np.asarray(decoder["b_galaxy"], dtype=np.float32)
    metrics_before = evaluate_trm_weights_on_traces(usable, weights, decoder, galaxy_order=galaxy_order)
    trace_weights = build_trace_balance_weights(usable, galaxy_order=galaxy_order).reshape(-1, 1)
    drawing_index = next((idx for idx, name in enumerate(galaxy_order) if name == "Drawing"), 0)
    arc_mask = np.asarray(
        [
            str(trace.get("benchmark") or "").strip().upper() == "ARC"
            or str(trace.get("task_type") or "").strip().upper() == "SPATIAL_TASK"
            for trace in usable
        ],
        dtype=bool,
    ).reshape(-1, 1)
    loss_history: list[float] = []
    batch_scale = float(q_batch.shape[0] * target_logits.shape[1])
    raw_target = np.zeros((q_batch.shape[0], 512), dtype=np.float32)
    raw_target[:, : len(galaxy_order)] = target_logits
    raw_scale = float(q_batch.shape[0] * max(len(galaxy_order), 1))
    for _ in range(int(epochs)):
        y_new, cache = trm_forward_numpy(q_batch, weights)
        predicted_logits = y_new @ w_galaxy.T + b_galaxy
        diff = predicted_logits - target_logits
        raw_diff = y_new[:, : len(galaxy_order)] - raw_target[:, : len(galaxy_order)]
        weighted_diff = diff * trace_weights
        weighted_raw_diff = raw_diff * trace_weights
        decoder_loss = float(np.mean((diff * diff) * trace_weights))
        raw_loss = float(np.mean((raw_diff * raw_diff) * trace_weights))
        loss = decoder_loss + float(raw_signal_weight) * raw_loss
        if arc_drawing_bonus_weight > 0.0 and arc_mask.any():
            drawing_logits = predicted_logits[:, drawing_index : drawing_index + 1]
            if len(galaxy_order) > 1:
                non_drawing_logits = np.concatenate(
                    [
                        predicted_logits[:, :drawing_index],
                        predicted_logits[:, drawing_index + 1 : len(galaxy_order)],
                    ],
                    axis=1,
                )
                competing_logits = np.max(non_drawing_logits, axis=1, keepdims=True)
            else:
                competing_logits = np.zeros_like(drawing_logits)
            arc_gap = drawing_logits - competing_logits
            arc_hinge = np.maximum(np.float32(0.0), np.float32(arc_drawing_margin) - arc_gap)
            weighted_arc_hinge = arc_hinge * trace_weights * arc_mask.astype(np.float32)
            loss += float(arc_drawing_bonus_weight) * float(np.mean(weighted_arc_hinge * weighted_arc_hinge))
        loss_history.append(loss)
        d_logits = (2.0 / batch_scale) * weighted_diff
        d_y = d_logits @ w_galaxy
        if arc_drawing_bonus_weight > 0.0 and arc_mask.any():
            drawing_logits = predicted_logits[:, drawing_index : drawing_index + 1]
            if len(galaxy_order) > 1:
                non_drawing_logits = np.concatenate(
                    [
                        predicted_logits[:, :drawing_index],
                        predicted_logits[:, drawing_index + 1 : len(galaxy_order)],
                    ],
                    axis=1,
                )
                competing_argmax = np.argmax(non_drawing_logits, axis=1)
                competing_full = competing_argmax.copy()
                competing_full[competing_full >= drawing_index] += 1
                competing_logits = predicted_logits[np.arange(predicted_logits.shape[0]), competing_full].reshape(-1, 1)
            else:
                competing_full = np.zeros(predicted_logits.shape[0], dtype=np.int64)
                competing_logits = np.zeros_like(drawing_logits)
            arc_gap = drawing_logits - competing_logits
            arc_hinge = np.maximum(np.float32(0.0), np.float32(arc_drawing_margin) - arc_gap)
            arc_grad = (
                np.float32(-2.0 * arc_drawing_bonus_weight / max(batch_scale, 1.0))
                * arc_hinge
                * trace_weights
                * arc_mask.astype(np.float32)
            )
            d_logits[:, drawing_index] += arc_grad.reshape(-1)
            if len(galaxy_order) > 1:
                d_logits[np.arange(d_logits.shape[0]), competing_full] -= arc_grad.reshape(-1)
        if raw_signal_weight > 0.0:
            d_y[:, : len(galaxy_order)] += (
                np.float32(2.0 * raw_signal_weight / max(raw_scale, 1.0))
                * weighted_raw_diff.astype(np.float32, copy=False)
            )
        d_w4 = d_y.T @ cache["hidden2"]
        d_hidden2 = d_y @ weights["W4"]
        d_hidden2_pre = d_hidden2 * _swish_grad(cache["hidden2_pre"])
        d_w3 = d_hidden2_pre.T @ cache["z_new"]
        d_z = d_hidden2_pre @ weights["W3"]
        d_w2 = d_z.T @ cache["hidden1"]
        d_hidden1 = d_z @ weights["W2"]
        d_hidden1_pre = d_hidden1 * _swish_grad(cache["hidden1_pre"])
        d_w1 = d_hidden1_pre.T @ cache["q"]
        grad_norm = float(
            np.sqrt(
                np.sum(d_w1 * d_w1)
                + np.sum(d_w2 * d_w2)
                + np.sum(d_w3 * d_w3)
                + np.sum(d_w4 * d_w4)
            )
        )
        scale = 1.0
        if clip_norm > 0.0 and grad_norm > clip_norm:
            scale = float(clip_norm / max(grad_norm, 1e-9))
        weights["W1"] -= np.float32(learning_rate * scale) * d_w1.astype(np.float32, copy=False)
        weights["W2"] -= np.float32(learning_rate * scale) * d_w2.astype(np.float32, copy=False)
        weights["W3"] -= np.float32(learning_rate * scale) * d_w3.astype(np.float32, copy=False)
        weights["W4"] -= np.float32(learning_rate * scale) * d_w4.astype(np.float32, copy=False)
    metrics_after = evaluate_trm_weights_on_traces(usable, weights, decoder, galaxy_order=galaxy_order)
    return {
        "weights": weights,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "final_loss": float(loss_history[-1]) if loss_history else 0.0,
        "loss_history": loss_history[-16:],
        "raw_signal_weight": float(raw_signal_weight),
        "arc_drawing_bonus_weight": float(arc_drawing_bonus_weight),
        "arc_drawing_margin": float(arc_drawing_margin),
        "target_blend_alpha": float(target_blend_alpha),
        "trace_balance_weights": trace_weights.reshape(-1).astype(np.float32, copy=False).tolist(),
        "galaxy_idf": {key: float(value) for key, value in galaxy_idf.items()},
    }


def evaluate_decoder_on_traces(
    traces: list[dict[str, Any]],
    decoder: dict[str, Any] | None = None,
    *,
    galaxy_order: tuple[str, ...] = DEFAULT_GALAXY_ORDER,
) -> dict[str, float]:
    if not traces:
        return {"trace_count": 0.0, "avg_entropy": 0.0, "top1_match_rate": 0.0}
    entropies: list[float] = []
    top1_matches = 0
    for trace in traces:
        if decoder is None:
            logits = np.asarray(list(trace.get("y_new_vector_512", [])), dtype=np.float32)[: len(galaxy_order)]
        else:
            logits = apply_galaxy_decoder(trace.get("y_new_vector_512", []), decoder)
        distribution = softmax(logits)
        entropies.append(float(-np.sum(distribution * np.log(np.clip(distribution, 1e-9, 1.0)))))
        top1_idx = int(np.argmax(distribution)) if distribution.size else -1
        top1_name = str(galaxy_order[top1_idx]) if top1_idx >= 0 else ""
        if top1_name and top1_name in {str(g) for g in trace.get("python_galaxies", [])}:
            top1_matches += 1
    return {
        "trace_count": float(len(traces)),
        "avg_entropy": float(sum(entropies) / float(len(entropies))),
        "top1_match_rate": float(top1_matches / float(len(traces))),
    }


def fit_galaxy_decoder_from_traces(
    traces: list[dict[str, Any]],
    *,
    galaxy_idf: dict[str, float] | None = None,
    target_blend_alpha: float = 1.0,
    galaxy_order: tuple[str, ...] = DEFAULT_GALAXY_ORDER,
) -> dict[str, Any]:
    usable = [
        trace
        for trace in traces
        if len(trace.get("y_new_vector_512", [])) == 512 and trace.get("python_galaxies")
    ]
    if not usable:
        raise ValueError("No usable TRM galaxy traces found")
    x = np.asarray([trace["y_new_vector_512"] for trace in usable], dtype=np.float32)
    resolved_idf = galaxy_idf or compute_galaxy_idf(usable, galaxy_order=galaxy_order)
    targets = np.asarray(
        [
            trace_target_logits(
                trace.get("python_galaxies", []),
                galaxy_contribution=(
                    trace.get("galaxy_contribution")
                    if isinstance(trace.get("galaxy_contribution"), dict)
                    else None
                ),
                galaxy_idf=resolved_idf,
                target_blend_alpha=target_blend_alpha,
                galaxy_order=galaxy_order,
            )
            for trace in usable
        ],
        dtype=np.float32,
    )
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    coeffs, _, _, _ = np.linalg.lstsq(x_aug, targets, rcond=None)
    weights = coeffs[:-1, :].T.astype(np.float32, copy=False)
    bias = coeffs[-1, :].astype(np.float32, copy=False)
    decoder = {
        "W_galaxy": weights,
        "b_galaxy": bias,
        "galaxy_order": np.asarray(galaxy_order, dtype="<U32"),
        "galaxy_idf": {key: float(value) for key, value in resolved_idf.items()},
        "target_blend_alpha": float(target_blend_alpha),
    }
    decoder["metrics_before"] = evaluate_decoder_on_traces(usable, None, galaxy_order=galaxy_order)
    decoder["metrics_after"] = evaluate_decoder_on_traces(usable, decoder, galaxy_order=galaxy_order)
    return decoder


def save_galaxy_decoder_checkpoint(
    output_path: str | Path,
    decoder: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "W_galaxy": np.asarray(decoder["W_galaxy"], dtype=np.float32),
        "b_galaxy": np.asarray(decoder["b_galaxy"], dtype=np.float32),
        "galaxy_order": np.asarray(decoder.get("galaxy_order", DEFAULT_GALAXY_ORDER), dtype="<U32"),
        "metadata_json": np.asarray([json.dumps(metadata or {}, ensure_ascii=True)], dtype=CHECKPOINT_METADATA_DTYPE),
    }
    np.savez_compressed(path, **payload)
    return path


def load_galaxy_decoder_checkpoint(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as payload:
        decoder = {
            "W_galaxy": payload["W_galaxy"].astype(np.float32, copy=False),
            "b_galaxy": payload["b_galaxy"].astype(np.float32, copy=False),
            "galaxy_order": payload["galaxy_order"],
        }
        if "metadata_json" in payload:
            raw = payload["metadata_json"]
            if raw.size:
                raw_text = str(raw.reshape(-1)[0])
                try:
                    decoder["metadata"] = json.loads(raw_text)
                except json.JSONDecodeError:
                    decoder["metadata_error"] = "truncated_metadata_json"
                    decoder["metadata_json_raw"] = raw_text
        return decoder


def save_trm_weight_checkpoint(
    output_path: str | Path,
    weights: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: np.asarray(weights[name], dtype=np.float32)
        for name in ("W1", "W2", "W3", "W4")
    }
    if "matryoshka" in weights:
        payload["matryoshka"] = np.asarray(weights["matryoshka"], dtype=np.float32)
    payload["metadata_json"] = np.asarray([json.dumps(metadata or {}, ensure_ascii=True)], dtype=CHECKPOINT_METADATA_DTYPE)
    np.savez_compressed(path, **payload)
    return path


def load_trm_weight_checkpoint(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as payload:
        weights = {
            name: payload[name].astype(np.float32, copy=False)
            for name in ("W1", "W2", "W3", "W4")
        }
        if "matryoshka" in payload:
            weights["matryoshka"] = payload["matryoshka"].astype(np.float32, copy=False)
        if "metadata_json" in payload:
            raw = payload["metadata_json"]
            if raw.size:
                raw_text = str(raw.reshape(-1)[0])
                try:
                    weights["metadata"] = json.loads(raw_text)
                except json.JSONDecodeError:
                    weights["metadata_error"] = "truncated_metadata_json"
                    weights["metadata_json_raw"] = raw_text
        return weights
