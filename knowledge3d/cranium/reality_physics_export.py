"""Export physics/chemistry/biology/material systems into RealitySystem nodes with tier metadata."""

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


# ========== Phase 4C: Chemistry Systems ==========


def export_water_molecule(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
    """H2O molecule with simple bond springs and energy tracking."""
    p = params or {}

    # Defaults approximate bent geometry
    x_O = float(p.get("x_O", 0.0))
    y_O = float(p.get("y_O", 0.0))
    z_O = float(p.get("z_O", 0.0))
    x_H1 = float(p.get("x_H1", 0.96))
    y_H1 = float(p.get("y_H1", 0.0))
    z_H1 = float(p.get("z_H1", 0.0))
    x_H2 = float(p.get("x_H2", -0.24))
    y_H2 = float(p.get("y_H2", 0.93))
    z_H2 = float(p.get("z_H2", 0.0))
    vx_O = float(p.get("vx_O", 0.0))
    vy_O = float(p.get("vy_O", 0.0))
    vz_O = float(p.get("vz_O", 0.0))
    vx_H1 = float(p.get("vx_H1", 0.0))
    vy_H1 = float(p.get("vy_H1", 0.0))
    vz_H1 = float(p.get("vz_H1", 0.0))
    vx_H2 = float(p.get("vx_H2", 0.0))
    vy_H2 = float(p.get("vy_H2", 0.0))
    vz_H2 = float(p.get("vz_H2", 0.0))
    dt = float(p.get("dt", 0.001))
    k_spring = float(p.get("k_spring", 450.0))

    # Compute an approximate initial energy for conservation checks
    def _bond_energy(dx: float, dy: float, dz: float) -> float:
        import math

        r = math.sqrt(dx * dx + dy * dy + dz * dz)
        dr = r - 0.96
        return 0.5 * k_spring * dr * dr

    Ek = 0.5 * (vx_O**2 + vy_O**2 + vz_O**2 + vx_H1**2 + vy_H1**2 + vz_H1**2 + vx_H2**2 + vy_H2**2 + vz_H2**2)
    Ep = _bond_energy(x_H1 - x_O, y_H1 - y_O, z_H1 - z_O) + _bond_energy(x_H2 - x_O, y_H2 - y_O, z_H2 - z_O)
    E_initial = Ek + Ep

    return RealitySystem(
        node_id="system:water_molecule",
        state={
            "x_O": x_O,
            "y_O": y_O,
            "z_O": z_O,
            "x_H1": x_H1,
            "y_H1": y_H1,
            "z_H1": z_H1,
            "x_H2": x_H2,
            "y_H2": y_H2,
            "z_H2": z_H2,
            "vx_O": vx_O,
            "vy_O": vy_O,
            "vz_O": vz_O,
            "vx_H1": vx_H1,
            "vy_H1": vy_H1,
            "vz_H1": vz_H1,
            "vx_H2": vx_H2,
            "vy_H2": vy_H2,
            "vz_H2": vz_H2,
            "bond_angle": float(p.get("bond_angle", 104.5)),
            "k_spring": k_spring,
            "E_total": E_initial,
            "E_initial": E_initial,
            "dt": dt,
        },
        behavior_rpn="",
        law_rpn="",
        rpn_tier=2,
        rpn_instance=None if auto_allocate else 7,
        matryoshka_dim=128,
    )


def export_ideal_gas(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
    """Ideal gas PV = nRT."""
    p = params or {}
    return RealitySystem(
        node_id="system:ideal_gas",
        state={
            "P": float(p.get("P", 101325.0)),
            "V": float(p.get("V", 0.001)),
            "n": float(p.get("n", 1.0)),
            "T": float(p.get("T", 300.0)),
            "R": float(p.get("R", 8.314)),
        },
        behavior_rpn="""
            n RECALL R RECALL * T RECALL * V RECALL / P STORE
        """,
        law_rpn="""
            P RECALL V RECALL * n RECALL R RECALL * T RECALL * - abs
            n RECALL R RECALL * T RECALL * / 0.01 le
        """,
        rpn_tier=1,
        rpn_instance=None if auto_allocate else 8,
        matryoshka_dim=64,
    )


def export_combustion(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
    """Methane combustion with simple activation threshold."""
    p = params or {}
    return RealitySystem(
        node_id="system:combustion_ch4",
        state={
            "n_CH4": float(p.get("n_CH4", 1.0)),
            "n_O2": float(p.get("n_O2", 2.0)),
            "n_CO2": float(p.get("n_CO2", 0.0)),
            "n_H2O": float(p.get("n_H2O", 0.0)),
            "T": float(p.get("T", 300.0)),
            "E_released": float(p.get("E_released", 0.0)),
            "reaction_rate": float(p.get("reaction_rate", 0.0)),
        },
        behavior_rpn="""
            T RECALL 1000 gt react_gate STORE

            n_O2 RECALL 0.5 * o2_needed STORE
            n_CH4 RECALL o2_needed RECALL - diff STORE
            diff RECALL abs diff_abs STORE
            n_CH4 RECALL o2_needed RECALL + diff_abs RECALL - 0.5 * limiter STORE

            limiter RECALL react_gate RECALL * 0.01 * rate STORE
            rate RECALL reaction_rate STORE

            rate RECALL n_CH4 RECALL swap - n_CH4 STORE
            rate RECALL 2 * n_O2 RECALL swap - n_O2 STORE
            rate RECALL n_CO2 RECALL + n_CO2 STORE
            rate RECALL 2 * n_H2O RECALL + n_H2O STORE

            rate RECALL 890000 * E_released RECALL + E_released STORE
        """,
        law_rpn="",
        rpn_tier=2,
        rpn_instance=None if auto_allocate else 9,
        matryoshka_dim=512,
    )


def export_co2_molecule(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
    """Linear CO2 molecule with symmetric spring bonds."""
    p = params or {}
    dt = float(p.get("dt", 0.001))
    k_spring = float(p.get("k_spring", 400.0))
    return RealitySystem(
        node_id="system:co2_molecule",
        state={
            "x_C": float(p.get("x_C", 0.0)),
            "y_C": float(p.get("y_C", 0.0)),
            "z_C": float(p.get("z_C", 0.0)),
            "x_O1": float(p.get("x_O1", -1.16)),
            "y_O1": float(p.get("y_O1", 0.0)),
            "z_O1": float(p.get("z_O1", 0.0)),
            "x_O2": float(p.get("x_O2", 1.16)),
            "y_O2": float(p.get("y_O2", 0.0)),
            "z_O2": float(p.get("z_O2", 0.0)),
            "vx_C": float(p.get("vx_C", 0.0)),
            "vy_C": float(p.get("vy_C", 0.0)),
            "vz_C": float(p.get("vz_C", 0.0)),
            "vx_O1": float(p.get("vx_O1", 0.0)),
            "vy_O1": float(p.get("vy_O1", 0.0)),
            "vz_O1": float(p.get("vz_O1", 0.0)),
            "vx_O2": float(p.get("vx_O2", 0.0)),
            "vy_O2": float(p.get("vy_O2", 0.0)),
            "vz_O2": float(p.get("vz_O2", 0.0)),
            "k_spring": k_spring,
            "dt": dt,
        },
        behavior_rpn="""
            x_O1 RECALL x_C RECALL - dx1 STORE
            y_O1 RECALL y_C RECALL - dy1 STORE
            z_O1 RECALL z_C RECALL - dz1 STORE
            dx1 RECALL dup * dy1 RECALL dup * + dz1 RECALL dup * + sqrt r1 STORE
            r1 RECALL 1.16 - dr1 STORE
            dr1 RECALL k_spring RECALL * f1 STORE
            dx1 RECALL r1 RECALL / f1 RECALL * Fx1 STORE

            x_O2 RECALL x_C RECALL - dx2 STORE
            y_O2 RECALL y_C RECALL - dy2 STORE
            z_O2 RECALL z_C RECALL - dz2 STORE
            dx2 RECALL dup * dy2 RECALL dup * + dz2 RECALL dup * + sqrt r2 STORE
            r2 RECALL 1.16 - dr2 STORE
            dr2 RECALL k_spring RECALL * f2 STORE
            dx2 RECALL r2 RECALL / f2 RECALL * Fx2 STORE

            Fx1 RECALL vx_O1 RECALL dt RECALL * + vx_O1 STORE
            Fx2 RECALL vx_O2 RECALL dt RECALL * + vx_O2 STORE
            Fx1 RECALL Fx2 RECALL + NEG vx_C RECALL dt RECALL * + vx_C STORE

            vx_O1 RECALL dt RECALL * x_O1 RECALL + x_O1 STORE
            vx_O2 RECALL dt RECALL * x_O2 RECALL + x_O2 STORE
            vx_C RECALL dt RECALL * x_C RECALL + x_C STORE
        """,
        law_rpn="dr1 RECALL abs 0.2 le dr2 RECALL abs 0.2 le *",
        rpn_tier=2,
        rpn_instance=None if auto_allocate else 10,
        matryoshka_dim=128,
    )


def export_acid_base_reaction(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
    """Strong acid-base neutralization with simple pH estimate."""
    p = params or {}
    return RealitySystem(
        node_id="system:acid_base",
        state={
            "n_HCl": float(p.get("n_HCl", 1.0)),
            "n_NaOH": float(p.get("n_NaOH", 1.0)),
            "n_NaCl": float(p.get("n_NaCl", 0.0)),
            "n_H2O": float(p.get("n_H2O", 0.0)),
            "volume": float(p.get("volume", 1.0)),
            "pH": float(p.get("pH", 7.0)),
        },
        behavior_rpn="""
            n_HCl RECALL n_NaOH RECALL - diff STORE
            diff RECALL abs diff_abs STORE
            n_HCl RECALL n_NaOH RECALL + diff_abs RECALL - 0.5 * extent STORE

            extent RECALL 0.05 * reacted STORE

            reacted RECALL n_HCl RECALL swap - n_HCl STORE
            reacted RECALL n_NaOH RECALL swap - n_NaOH STORE
            reacted RECALL n_NaCl RECALL + n_NaCl STORE
            reacted RECALL n_H2O RECALL + n_H2O STORE

            n_NaOH RECALL n_HCl RECALL - delta STORE
            delta RECALL volume RECALL / pH_delta STORE
            7.0 pH_delta RECALL + pH STORE
        """,
        law_rpn="n_HCl RECALL 0 ge n_NaOH RECALL 0 ge *",
        rpn_tier=1,
        rpn_instance=None if auto_allocate else 11,
        matryoshka_dim=64,
    )


def export_phase_transition_water(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
    """Simple water phase change (ice/liquid/vapor) with ternary phase."""
    p = params or {}
    return RealitySystem(
        node_id="system:water_phase",
        state={
            "T": float(p.get("T", 260.0)),
            "phase": float(p.get("phase", -1.0)),  # -1 ice, 0 liquid, +1 vapor
            "latent_heat": float(p.get("latent_heat", 334000.0)),
        },
        behavior_rpn="""
            T RECALL 273 tcmp phase_low STORE
            T RECALL 373 tcmp phase_high STORE
            phase_low RECALL phase_high RECALL + 0.5 * phase STORE
        """,
        law_rpn="phase RECALL abs 1 le",
        rpn_tier=1,
        rpn_instance=None if auto_allocate else 12,
        matryoshka_dim=64,
    )


# ========== Phase 4C: Biology Systems ==========


def export_simple_cell(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
    """Cell membrane diffusion (Fick's law style)."""
    p = params or {}
    C_inside = float(p.get("C_inside", 0.1))
    C_outside = float(p.get("C_outside", 1.0))
    V_inside = float(p.get("V_inside", 1e-15))
    V_outside = float(p.get("V_outside", 1e-12))
    A_membrane = float(p.get("A_membrane", 3e-10))
    permeability = float(p.get("permeability", 1e-6))
    C_total_initial = C_inside * V_inside + C_outside * V_outside

    return RealitySystem(
        node_id="system:simple_cell",
        state={
            "C_inside": C_inside,
            "C_outside": C_outside,
            "V_inside": V_inside,
            "V_outside": V_outside,
            "A_membrane": A_membrane,
            "permeability": permeability,
            "C_total_initial": C_total_initial,
        },
        behavior_rpn="""
            C_outside RECALL C_inside RECALL - dC STORE
            permeability RECALL dC RECALL * A_membrane RECALL * flux STORE

            flux RECALL V_inside RECALL / 0.05 * C_inside RECALL + C_inside STORE
            flux RECALL V_outside RECALL / 0.05 * NEG C_outside RECALL + C_outside STORE
        """,
        law_rpn="""
            C_inside RECALL V_inside RECALL * C_outside RECALL V_outside RECALL * + C_total_initial RECALL - abs 1e-3 le
        """,
        rpn_tier=1,
        rpn_instance=None if auto_allocate else 13,
        matryoshka_dim=64,
    )


def export_enzyme_kinetics(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
    """Michaelis-Menten kinetics."""
    p = params or {}
    S_initial = float(p.get("S", 10.0))
    return RealitySystem(
        node_id="system:enzyme_kinetics",
        state={
            "E": float(p.get("E", 1.0)),
            "S": S_initial,
            "ES": float(p.get("ES", 0.0)),
            "P": float(p.get("P", 0.0)),
            "Vmax": float(p.get("Vmax", 1.0)),
            "Km": float(p.get("Km", 5.0)),
            "S_initial": S_initial,
        },
        behavior_rpn="""
            S RECALL Vmax RECALL * Km RECALL S RECALL + / rate STORE
            rate RECALL 0.01 * S RECALL swap - S STORE
            rate RECALL 0.01 * P RECALL + P STORE
            rate RECALL 0.005 * ES RECALL + ES STORE
        """,
        law_rpn="",
        rpn_tier=1,
        rpn_instance=None if auto_allocate else 14,
        matryoshka_dim=64,
    )


def export_dna_replication(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
    """DNA fork progression with simple error accumulation."""
    p = params or {}
    return RealitySystem(
        node_id="system:dna_replication",
        state={
            "bases_replicated": float(p.get("bases_replicated", 0.0)),
            "polymerase_position": float(p.get("polymerase_position", 0.0)),
            "error_count": float(p.get("error_count", 0.0)),
            "template_length": float(p.get("template_length", 1000.0)),
            "error_rate": float(p.get("error_rate", 1e-4)),
            "speed": float(p.get("speed", 50.0)),  # bases per step
        },
        behavior_rpn="""
            speed RECALL polymerase_position RECALL + polymerase_position STORE
            speed RECALL bases_replicated RECALL + bases_replicated STORE
            speed RECALL error_rate RECALL * error_count RECALL + error_count STORE
        """,
        law_rpn="bases_replicated RECALL template_length RECALL le polymerase_position RECALL template_length RECALL le *",
        rpn_tier=1,
        rpn_instance=None if auto_allocate else 15,
        matryoshka_dim=64,
    )


def export_population_dynamics(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
    """Lotka-Volterra style predator-prey dynamics."""
    p = params or {}
    return RealitySystem(
        node_id="system:population_dynamics",
        state={
            "N_prey": float(p.get("N_prey", 50.0)),
            "N_predator": float(p.get("N_predator", 5.0)),
            "birth_rate": float(p.get("birth_rate", 0.1)),
            "death_rate": float(p.get("death_rate", 0.1)),
            "predation_rate": float(p.get("predation_rate", 0.01)),
            "growth_rate": float(p.get("growth_rate", 0.01)),
            "dt": float(p.get("dt", 0.1)),
        },
        behavior_rpn="""
            birth_rate RECALL N_prey RECALL * pred_prey_growth STORE
            predation_rate RECALL N_prey RECALL * N_predator RECALL * predation STORE

            pred_prey_growth RECALL predation RECALL - dt RECALL * N_prey RECALL + N_prey STORE

            growth_rate RECALL predation RECALL * growth_pred STORE
            death_rate RECALL N_predator RECALL * death_pred STORE
            growth_pred RECALL death_pred RECALL - dt RECALL * N_predator RECALL + N_predator STORE
        """,
        law_rpn="N_prey RECALL 0 ge N_predator RECALL 0 ge *",
        rpn_tier=1,
        rpn_instance=None if auto_allocate else 16,
        matryoshka_dim=128,
    )


# ========== Phase 4C: Materials Science Systems ==========


def export_crystal_lattice(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
    """FCC copper lattice thermal expansion."""
    p = params or {}
    return RealitySystem(
        node_id="system:crystal_lattice",
        state={
            "a": float(p.get("a", 3.61e-10)),
            "T": float(p.get("T", 300.0)),
            "E_cohesive": float(p.get("E_cohesive", 3.49)),
            "thermal_expansion": float(p.get("thermal_expansion", 16.5e-6)),
        },
        behavior_rpn="""
            T RECALL 300 - dT STORE
            thermal_expansion RECALL dT RECALL * 1 + 3.61e-10 * a STORE
        """,
        law_rpn="a RECALL 0 gt T RECALL 0 gt *",
        rpn_tier=1,
        rpn_instance=None if auto_allocate else 17,
        matryoshka_dim=64,
    )


def export_composite_material(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
    """Fiber-reinforced composite (rule of mixtures)."""
    p = params or {}
    return RealitySystem(
        node_id="system:composite_material",
        state={
            "stress": float(p.get("stress", 100.0)),
            "strain": float(p.get("strain", 0.0)),
            "E_fiber": float(p.get("E_fiber", 200e9)),
            "E_matrix": float(p.get("E_matrix", 3e9)),
            "volume_fraction": float(p.get("volume_fraction", 0.6)),
            "E_composite": float(p.get("E_composite", 0.0)),
        },
        behavior_rpn="""
            volume_fraction RECALL vf STORE
            1 vf RECALL - vm STORE
            vf RECALL E_fiber RECALL * vm RECALL E_matrix RECALL * + E_composite STORE
            stress RECALL E_composite RECALL / strain STORE
        """,
        law_rpn="E_composite RECALL 0 gt strain RECALL 0 ge *",
        rpn_tier=1,
        rpn_instance=None if auto_allocate else 6,
        matryoshka_dim=64,
    )


def export_metal_melting(
    params: Dict | None = None,
    auto_allocate: bool = True,
) -> RealitySystem:
    """Metal melting phase transition with ternary phase."""
    p = params or {}
    return RealitySystem(
        node_id="system:metal_melting",
        state={
            "T": float(p.get("T", 300.0)),
            "T_melting": float(p.get("T_melting", 1356.0)),
            "phase": float(p.get("phase", -1.0)),  # -1 solid, +1 liquid
            "latent_heat": float(p.get("latent_heat", 205000.0)),
        },
        behavior_rpn="""
            T RECALL T_melting RECALL tcmp phase STORE
        """,
        law_rpn="phase RECALL abs 1 le",
        rpn_tier=1,
        rpn_instance=None if auto_allocate else 7,
        matryoshka_dim=64,
    )
