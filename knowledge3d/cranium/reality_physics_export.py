"""Export Phase 4A physics systems into RealitySystem nodes with tier metadata."""

from __future__ import annotations

from typing import Dict

from knowledge3d.cranium.reality_nodes import RealitySystem


def export_constant_acceleration_1d(params: Dict | None = None) -> RealitySystem:
    p = params or {}
    return RealitySystem(
        node_id="system:constant_accel_1d",
        state={
            "x": float(p.get("x", 0.0)),
            "v": float(p.get("v", 1.0)),
            "a": float(p.get("a", -9.81)),
            "dt": float(p.get("dt", 0.01)),
            "F": float(p.get("F", -9.81)),
            "m": float(p.get("m", 1.0)),
        },
        behavior_rpn="v RECALL a RECALL dt RECALL * + v STORE x RECALL v RECALL dt RECALL * + x STORE",
        law_rpn="F RECALL m RECALL / a RECALL - abs 1e-6 lt",
        rpn_tier=1,
        rpn_instance=0,
        matryoshka_dim=64,
    )


def export_harmonic_oscillator_1d(params: Dict | None = None) -> RealitySystem:
    p = params or {}
    return RealitySystem(
        node_id="system:harmonic_osc_1d",
        state={
            "x": float(p.get("x", 1.0)),
            "v": float(p.get("v", 0.0)),
            "omega": float(p.get("omega", 1.0)),
            "dt": float(p.get("dt", 0.001)),
            "a": 0.0,
        },
        behavior_rpn="x RECALL omega RECALL omega RECALL * * -1 * a STORE v RECALL a RECALL dt RECALL * + v STORE x RECALL v RECALL dt RECALL * + x STORE",
        law_rpn="",
        rpn_tier=1,
        rpn_instance=1,
        matryoshka_dim=64,
    )


def export_projectile_2d(params: Dict | None = None) -> RealitySystem:
    p = params or {}
    return RealitySystem(
        node_id="system:projectile_2d",
        state={
            "x": float(p.get("x", 0.0)),
            "y": float(p.get("y", 0.0)),
            "vx": float(p.get("vx", 10.0)),
            "vy": float(p.get("vy", 20.0)),
            "g": float(p.get("g", 9.81)),
            "k": float(p.get("k", 0.1)),
            "dt": float(p.get("dt", 0.001)),
        },
        behavior_rpn="""
            vx RECALL dup * vy RECALL dup * + sqrt v_mag STORE
            k RECALL v_mag RECALL * drag STORE
            vx RECALL sign sign_vx STORE
            vy RECALL sign sign_vy STORE
            sign_vx RECALL NEG drag RECALL * ax STORE
            g RECALL NEG sign_vy RECALL NEG drag RECALL * - ay STORE
            vx RECALL ax RECALL dt RECALL * + vx STORE
            vy RECALL ay RECALL dt RECALL * + vy STORE
            x RECALL vx RECALL dt RECALL * + x STORE
            y RECALL vy RECALL dt RECALL * + y STORE
        """,
        law_rpn="",
        rpn_tier=1,
        rpn_instance=2,
        matryoshka_dim=128,
    )


def export_rigid_body_2d(params: Dict | None = None) -> RealitySystem:
    p = params or {}
    return RealitySystem(
        node_id="system:rigid_body_2d",
        state={
            "theta": float(p.get("theta", 0.0)),
            "omega": float(p.get("omega", 1.0)),
            "I": float(p.get("I", 1.0)),
            "tau": float(p.get("tau", 0.0)),
            "dt": float(p.get("dt", 0.01)),
        },
        behavior_rpn="tau RECALL I RECALL / alpha STORE omega RECALL alpha RECALL dt RECALL * + omega STORE theta RECALL omega RECALL dt RECALL * + theta STORE",
        law_rpn="",
        rpn_tier=1,
        rpn_instance=3,
        matryoshka_dim=128,
    )


def export_heat_1d(params: Dict | None = None) -> RealitySystem:
    p = params or {}
    return RealitySystem(
        node_id="system:heat_1d",
        state={
            "alpha": float(p.get("alpha", 0.1)),
            "dt": float(p.get("dt", 0.01)),
            "dx": float(p.get("dx", 1.0)),
        },
        behavior_rpn="",  # host-side stencil
        law_rpn="",
        rpn_tier=2,
        rpn_instance=12,
        matryoshka_dim=128,
    )


def export_coupled_oscillators(params: Dict | None = None) -> RealitySystem:
    p = params or {}
    return RealitySystem(
        node_id="system:coupled_oscillators",
        state={
            "x1": float(p.get("x1", 0.1)),
            "x2": float(p.get("x2", -0.1)),
            "v1": float(p.get("v1", 0.0)),
            "v2": float(p.get("v2", 0.0)),
            "k": float(p.get("k", 1.0)),
            "k_c": float(p.get("k_c", 0.5)),
            "m1": float(p.get("m1", 1.0)),
            "m2": float(p.get("m2", 1.0)),
            "dt": float(p.get("dt", 0.01)),
        },
        behavior_rpn="x1 RECALL sign x1s STORE x2 RECALL sign x2s STORE x1s RECALL x2s RECALL * mode_product STORE",
        law_rpn="",
        rpn_tier=2,
        rpn_instance=13,
        matryoshka_dim=512,
    )


def export_orbital_2d(params: Dict | None = None) -> RealitySystem:
    p = params or {}
    return RealitySystem(
        node_id="system:orbital_2d",
        state={
            "x": float(p.get("x", 1.0)),
            "y": float(p.get("y", 0.0)),
            "vx": float(p.get("vx", 0.0)),
            "vy": float(p.get("vy", 1.0)),
            "mu": float(p.get("mu", 1.0)),
            "dt": float(p.get("dt", 0.01)),
        },
        behavior_rpn="",  # heavy math in host for now
        law_rpn="",
        rpn_tier=2,
        rpn_instance=14,
        matryoshka_dim=512,
    )


def export_heat_2d(params: Dict | None = None) -> RealitySystem:
    p = params or {}
    return RealitySystem(
        node_id="system:heat_2d",
        state={
            "alpha": float(p.get("alpha", 0.1)),
            "dt": float(p.get("dt", 0.01)),
            "dx": float(p.get("dx", 1.0)),
        },
        behavior_rpn="",
        law_rpn="",
        rpn_tier=2,
        rpn_instance=15,
        matryoshka_dim=512,
    )


def export_double_pendulum_2d(params: Dict | None = None) -> RealitySystem:
    p = params or {}
    return RealitySystem(
        node_id="system:double_pendulum_2d",
        state={
            "theta1": float(p.get("theta1", 0.1)),
            "theta2": float(p.get("theta2", 0.2)),
            "omega1": float(p.get("omega1", 0.0)),
            "omega2": float(p.get("omega2", 0.0)),
            "L1": float(p.get("L1", 1.0)),
            "L2": float(p.get("L2", 1.0)),
            "m1": float(p.get("m1", 1.0)),
            "m2": float(p.get("m2", 1.0)),
            "g": float(p.get("g", 9.81)),
            "dt": float(p.get("dt", 0.01)),
        },
        behavior_rpn="",
        law_rpn="",
        rpn_tier=3,
        rpn_instance=16,
        matryoshka_dim=2048,
    )
