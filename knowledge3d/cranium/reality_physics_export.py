"""Export Phase 4A physics systems into RealitySystem nodes with tier metadata."""

from __future__ import annotations

from typing import Dict

from knowledge3d.cranium.reality_nodes import RealitySystem


def export_constant_acceleration_1d(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
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
        rpn_instance=None if auto_allocate else 0,
        matryoshka_dim=64,
    )


def export_harmonic_oscillator_1d(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
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
        rpn_instance=None if auto_allocate else 1,
        matryoshka_dim=64,
    )


def export_projectile_2d(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
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
        rpn_instance=None if auto_allocate else 2,
        matryoshka_dim=128,
    )


def export_rigid_body_2d(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
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
        rpn_instance=None if auto_allocate else 3,
        matryoshka_dim=128,
    )


def export_heat_1d(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
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
        rpn_instance=None if auto_allocate else 12,
        matryoshka_dim=128,
    )


def export_coupled_oscillators(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
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
        rpn_instance=None if auto_allocate else 13,
        matryoshka_dim=512,
    )


def export_orbital_2d(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
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
        rpn_instance=None if auto_allocate else 14,
        matryoshka_dim=512,
    )


def export_heat_2d(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
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
        rpn_instance=None if auto_allocate else 15,
        matryoshka_dim=512,
    )


def export_double_pendulum_2d(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
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
        rpn_instance=None if auto_allocate else 16,
        matryoshka_dim=2048,
    )


# ========== Phase 4B: Electromagnetism Systems ==========


def export_point_charge_2d(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
    """Export two-charge Coulomb system with ternary charge signs."""
    p = params or {}
    return RealitySystem(
        node_id="system:point_charge_2d",
        state={
            "x1": float(p.get("x1", -1.0)),
            "y1": float(p.get("y1", 0.0)),
            "x2": float(p.get("x2", 1.0)),
            "y2": float(p.get("y2", 0.0)),
            "vx1": float(p.get("vx1", 0.0)),
            "vy1": float(p.get("vy1", 0.0)),
            "vx2": float(p.get("vx2", 0.0)),
            "vy2": float(p.get("vy2", 0.0)),
            "q1": float(p.get("q1", 1e-6)),
            "q2": float(p.get("q2", 1e-6)),
            "m1": float(p.get("m1", 1.0)),
            "m2": float(p.get("m2", 1.0)),
            "k": float(p.get("k", 8.99e9)),
            "dt": float(p.get("dt", 0.001)),
        },
        behavior_rpn="""
            x2 RECALL x1 RECALL - dx STORE
            y2 RECALL y1 RECALL - dy STORE
            dx RECALL dup * dy RECALL dup * + sqrt r STORE
            k RECALL q1 RECALL * q2 RECALL * r RECALL r RECALL * / F_mag STORE
            q1 RECALL sign q1_sign STORE
            q2 RECALL sign q2_sign STORE
            q1_sign RECALL q2_sign RECALL * charge_product STORE
            F_mag RECALL dx RECALL * r RECALL / Fx STORE
            F_mag RECALL dy RECALL * r RECALL / Fy STORE
            Fx RECALL NEG m1 RECALL / ax1 STORE
            Fy RECALL NEG m1 RECALL / ay1 STORE
            Fx RECALL m2 RECALL / ax2 STORE
            Fy RECALL m2 RECALL / ay2 STORE
            vx1 RECALL ax1 RECALL dt RECALL * + vx1 STORE
            vy1 RECALL ay1 RECALL dt RECALL * + vy1 STORE
            vx2 RECALL ax2 RECALL dt RECALL * + vx2 STORE
            vy2 RECALL ay2 RECALL dt RECALL * + vy2 STORE
            x1 RECALL vx1 RECALL dt RECALL * + x1 STORE
            y1 RECALL vy1 RECALL dt RECALL * + y1 STORE
            x2 RECALL vx2 RECALL dt RECALL * + x2 STORE
            y2 RECALL vy2 RECALL dt RECALL * + y2 STORE
        """,
        law_rpn="",
        rpn_tier=1,
        rpn_instance=None if auto_allocate else 4,
        matryoshka_dim=128,
    )


def export_lc_circuit(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
    """Export LC oscillator with resonant frequency validation."""
    p = params or {}
    return RealitySystem(
        node_id="system:lc_circuit",
        state={
            "I": float(p.get("I", 1.0)),
            "V": float(p.get("V", 0.0)),
            "L": float(p.get("L", 1e-3)),
            "C": float(p.get("C", 1e-6)),
            "dt": float(p.get("dt", 1e-6)),
        },
        behavior_rpn="""
            V RECALL NEG L RECALL / dI_dt STORE
            I RECALL C RECALL / dV_dt STORE
            I RECALL dI_dt RECALL dt RECALL * + I STORE
            V RECALL dV_dt RECALL dt RECALL * + V STORE
        """,
        law_rpn="",
        rpn_tier=1,
        rpn_instance=None if auto_allocate else 5,
        matryoshka_dim=128,
    )


def export_rc_circuit(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
    """Export RC charging with exponential time constant."""
    p = params or {}
    return RealitySystem(
        node_id="system:rc_circuit",
        state={
            "V": float(p.get("V", 0.0)),
            "V_source": float(p.get("V_source", 5.0)),
            "R": float(p.get("R", 1000.0)),
            "C": float(p.get("C", 1e-6)),
            "dt": float(p.get("dt", 1e-5)),
        },
        behavior_rpn="""
            V_source RECALL V RECALL - V_diff STORE
            R RECALL C RECALL * tau STORE
            V_diff RECALL tau RECALL / dV_dt STORE
            V RECALL dV_dt RECALL dt RECALL * + V STORE
        """,
        law_rpn="V RECALL 0 ge V RECALL V_source RECALL le *",
        rpn_tier=1,
        rpn_instance=None if auto_allocate else 6,
        matryoshka_dim=64,
    )


def export_rlc_circuit(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
    """Export RLC circuit with ternary damping regime detection."""
    p = params or {}
    return RealitySystem(
        node_id="system:rlc_circuit",
        state={
            "I": float(p.get("I", 1.0)),
            "V": float(p.get("V", 0.0)),
            "R": float(p.get("R", 10.0)),
            "L": float(p.get("L", 1e-3)),
            "C": float(p.get("C", 1e-6)),
            "dt": float(p.get("dt", 1e-6)),
        },
        behavior_rpn="""
            R RECALL 0.5 * C RECALL L RECALL / sqrt * zeta STORE
            zeta RECALL 1.0 tcmp damping_regime STORE
            V RECALL NEG L RECALL / R RECALL I RECALL * L RECALL / - dI_dt STORE
            I RECALL C RECALL / dV_dt STORE
            I RECALL dI_dt RECALL dt RECALL * + I STORE
            V RECALL dV_dt RECALL dt RECALL * + V STORE
        """,
        law_rpn="",
        rpn_tier=2,
        rpn_instance=None if auto_allocate else 12,
        matryoshka_dim=512,
    )
