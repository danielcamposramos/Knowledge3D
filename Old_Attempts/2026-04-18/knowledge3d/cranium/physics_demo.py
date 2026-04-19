"""
Minimal Physics Galaxy demo built on the existing RPN math core.

This module provides a small, concrete example of a "reality system"
implemented as an RPN program:

- 1D constant-acceleration motion:
    x_{t+1} = x_t + v_t * dt
    v_{t+1} = v_t + a * dt

The goal is to:
- Demonstrate that physical laws can be expressed as RPN programs and
  executed via the sovereign math core (`ModularRPNEngine`).
- Provide a simple, testable system that can later be wrapped in
  `reality_*` nodes and integrated with Galaxy/House as described in
  REALITY_ENABLER_SPECIFICATION.md.

This is intentionally small and self-contained: it touches only the math
core and does not yet depend on viewer or House integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from typing import Optional
import math
import numpy as np
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine


@dataclass
class ConstantAcceleration1D:
    """
    1D constant-acceleration system using RPN for state updates.

    State:
        position: current position x_t
        velocity: current velocity v_t
        acceleration: constant acceleration a
        dt: timestep size

    Update law (per step):
        v_{t+1} = v_t + a * dt
        x_{t+1} = x_t + v_{t+1} * dt
    """

    position: float
    velocity: float
    acceleration: float
    dt: float
    _engine: ModularRPNEngine | None = None

    @property
    def engine(self) -> ModularRPNEngine:
        """Lazily construct the RPN engine."""
        if self._engine is None:
            self._engine = ModularRPNEngine()
        return self._engine

    def _eval(self, expr: str) -> float:
        """Evaluate a scalar RPN expression via the math core."""
        return self.engine.evaluate(expr)

    def step(self, n_steps: int = 1) -> tuple[float, float]:
        """
        Advance the system by n_steps using RPN programs.

        Returns:
            (position, velocity) after the last step.
        """
        for _ in range(n_steps):
            # v_{t+1} = v_t + a * dt
            expr_v = f"{self.velocity} {self.acceleration} {self.dt} * +"
            new_v = self._eval(expr_v)

            # x_{t+1} = x_t + v_{t+1} * dt
            expr_x = f"{self.position} {new_v} {self.dt} * +"
            new_x = self._eval(expr_x)

            self.velocity = new_v
            self.position = new_x

        return self.position, self.velocity


@dataclass
class HarmonicOscillator1D:
    """
    1D harmonic oscillator using RPN for state updates.

    Equation:
        d²x/dt² + ω² x = 0

    Written as first-order system:
        v' = -ω² x
        x' = v

    Discrete update (Euler-like):
        v_{t+1} = v_t + (-ω² x_t) * dt
        x_{t+1} = x_t + v_{t+1} * dt
    """

    position: float
    velocity: float
    omega: float
    dt: float
    _engine: ModularRPNEngine | None = None

    @property
    def engine(self) -> ModularRPNEngine:
        if self._engine is None:
            self._engine = ModularRPNEngine()
        return self._engine

    def _eval(self, expr: str) -> float:
        return self.engine.evaluate(expr)

    def step(self, n_steps: int = 1) -> tuple[float, float]:
        """
        Advance the oscillator by n_steps.

        Returns:
            (position, velocity) after the last step.
        """
        for _ in range(n_steps):
            # Compute acceleration a = -ω² x_t in Python; RPN performs the
            # integration step where precision matters.
            a_val = -float(self.omega ** 2) * float(self.position)

            # v_{t+1} = v_t + a * dt
            expr_v = f"{self.velocity} {a_val} {self.dt} * +"
            new_v = self._eval(expr_v)

            # x_{t+1} = x_t + v_{t+1} * dt
            expr_x = f"{self.position} {new_v} {self.dt} * +"
            new_x = self._eval(expr_x)

            self.velocity = new_v
            self.position = new_x

        return self.position, self.velocity


@dataclass
class Orbital2D:
    """
    Simple 2D central-force orbit under Newtonian gravity.

    State:
        x, y   : position components
        vx, vy : velocity components
        mu     : gravitational parameter (G * M)
        dt     : timestep

    Force law:
        a = -mu * r / |r|^3,  r = (x, y)

    Discrete update (Euler-like):
        v_{t+1} = v_t + a * dt
        x_{t+1} = x_t + v_{t+1} * dt
    """

    x: float
    y: float
    vx: float
    vy: float
    mu: float
    dt: float
    _engine: ModularRPNEngine | None = None

    @property
    def engine(self) -> ModularRPNEngine:
        if self._engine is None:
            self._engine = ModularRPNEngine()
        return self._engine

    def _eval(self, expr: str) -> float:
        return self.engine.evaluate(expr)

    def step(self, n_steps: int = 1) -> tuple[float, float, float, float]:
        """
        Advance the orbit by n_steps.

        Returns:
            (x, y, vx, vy) after the last step.
        """
        for _ in range(n_steps):
            # Compute acceleration vector a = -mu * r / |r|^3 in Python.
            r2 = self.x * self.x + self.y * self.y
            r = math.sqrt(max(r2, 1e-12))
            inv_r3 = 1.0 / (r2 * r + 1e-12)
            ax = -self.mu * self.x * inv_r3
            ay = -self.mu * self.y * inv_r3

            # v_{t+1} = v_t + a * dt (component-wise via RPN)
            expr_vx = f"{self.vx} {ax} {self.dt} * +"
            expr_vy = f"{self.vy} {ay} {self.dt} * +"
            new_vx = self._eval(expr_vx)
            new_vy = self._eval(expr_vy)

            # x_{t+1} = x_t + v_{t+1} * dt (component-wise via RPN)
            expr_x = f"{self.x} {new_vx} {self.dt} * +"
            expr_y = f"{self.y} {new_vy} {self.dt} * +"
            new_x = self._eval(expr_x)
            new_y = self._eval(expr_y)

            self.vx = new_vx
            self.vy = new_vy
            self.x = new_x
            self.y = new_y

        return self.x, self.y, self.vx, self.vy


@dataclass
class Heat1D:
    """
    1D heat diffusion (finite-difference) using RPN for the integration step.

    Discrete update (explicit scheme):
        T_i^{n+1} = T_i^n + alpha * dt / dx^2 * (T_{i+1}^n - 2 T_i^n + T_{i-1}^n)

    We compute the Laplacian (neighbor stencil) in Python and delegate the
    final integration step T_i^{n+1} = T_i^n + dT_i * dt to the RPN math core.

    Boundary conditions:
        - For now, we keep T_0 and T_{N-1} fixed (Dirichlet-style).
    """

    temperature: np.ndarray  # shape [N], float32/float64
    alpha: float
    dx: float
    dt: float
    _engine: ModularRPNEngine | None = None

    @property
    def engine(self) -> ModularRPNEngine:
        if self._engine is None:
            self._engine = ModularRPNEngine()
        return self._engine

    def _eval(self, expr: str) -> float:
        return self.engine.evaluate(expr)

    def step(self, n_steps: int = 1) -> np.ndarray:
        """
        Advance the temperature field by n_steps.

        Returns:
            Updated temperature array after n_steps.
        """
        T = np.asarray(self.temperature, dtype=np.float32)
        n = T.size
        if n < 3:
            return T

        coeff = self.alpha * self.dt / max(self.dx * self.dx, 1e-12)

        for _ in range(n_steps):
            T_new = T.copy()
            for i in range(1, n - 1):
                lap = T[i + 1] - 2.0 * T[i] + T[i - 1]
                dT = coeff * lap
                expr = f"{float(T[i])} {float(dT)} 1.0 * +"
                # Here dt is already folded into coeff, so we use 1.0 in RPN.
                T_new[i] = self._eval(expr)
            T = T_new

        self.temperature = T
        return T


@dataclass
class Heat2D:
    """
    2D heat diffusion on a rectangular grid using an explicit scheme.

    Discrete update (5-point stencil):
        T_{i,j}^{n+1} = T_{i,j}^n + alpha * dt / dx^2 * (
            T_{i+1,j}^n + T_{i-1,j}^n + T_{i,j+1}^n + T_{i,j-1}^n - 4 T_{i,j}^n
        )

    Boundary conditions:
        - Fixed (Dirichlet): border cells remain unchanged.

    As with Heat1D, the stencil is assembled on host; the RPN math core
    performs the final integration step for each interior cell.
    """

    temperature: np.ndarray  # shape [H, W]
    alpha: float
    dx: float
    dt: float
    _engine: ModularRPNEngine | None = None

    @property
    def engine(self) -> ModularRPNEngine:
        if self._engine is None:
            self._engine = ModularRPNEngine()
        return self._engine

    def _eval(self, expr: str) -> float:
        return self.engine.evaluate(expr)

    def step(self, n_steps: int = 1) -> np.ndarray:
        """
        Advance the 2D field by n_steps.

        Returns:
            Updated temperature array.
        """
        T = np.asarray(self.temperature, dtype=np.float32)
        h, w = T.shape
        if h < 3 or w < 3:
            return T

        coeff = self.alpha * self.dt / max(self.dx * self.dx, 1e-12)

        for _ in range(n_steps):
            T_new = T.copy()
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    lap = (
                        T[i + 1, j]
                        + T[i - 1, j]
                        + T[i, j + 1]
                        + T[i, j - 1]
                        - 4.0 * T[i, j]
                    )
                    dT = coeff * lap
                    expr = f"{float(T[i, j])} {float(dT)} 1.0 * +"
                    T_new[i, j] = self._eval(expr)
            T = T_new

        self.temperature = T
        return T


@dataclass
class Projectile2D:
    """
    2D projectile motion with air resistance.

    State:
        x, y   : position components
        vx, vy : velocity components
        g      : gravitational acceleration (positive downward)
        k      : air resistance coefficient
        dt     : timestep

    Forces:
        F_gravity = (0, -g)
        F_drag = -k * v * |v|  (quadratic drag)

    Discrete update (Euler-like):
        a = (0, -g) - k * v * |v|
        v_{t+1} = v_t + a * dt
        x_{t+1} = x_t + v_{t+1} * dt
    """

    x: float
    y: float
    vx: float
    vy: float
    g: float
    k: float
    dt: float
    _engine: ModularRPNEngine | None = None

    @property
    def engine(self) -> ModularRPNEngine:
        if self._engine is None:
            self._engine = ModularRPNEngine()
        return self._engine

    def _eval(self, expr: str) -> float:
        return self.engine.evaluate(expr)

    def step(self, n_steps: int = 1) -> tuple[float, float, float, float]:
        """
        Advance the projectile by n_steps.

        Returns:
            (x, y, vx, vy) after the last step.
        """
        for _ in range(n_steps):
            # Compute drag force: F_drag = -k * v * |v|
            v_mag = math.sqrt(self.vx * self.vx + self.vy * self.vy)
            drag_factor = self.k * v_mag

            # ax = -k * vx * |v|
            ax = -drag_factor * self.vx

            # ay = -g - k * vy * |v|
            ay = -self.g - drag_factor * self.vy

            # v_{t+1} = v_t + a * dt (component-wise via RPN)
            expr_vx = f"{self.vx} {ax} {self.dt} * +"
            expr_vy = f"{self.vy} {ay} {self.dt} * +"
            new_vx = self._eval(expr_vx)
            new_vy = self._eval(expr_vy)

            # x_{t+1} = x_t + v_{t+1} * dt (component-wise via RPN)
            expr_x = f"{self.x} {new_vx} {self.dt} * +"
            expr_y = f"{self.y} {new_vy} {self.dt} * +"
            new_x = self._eval(expr_x)
            new_y = self._eval(expr_y)

            self.vx = new_vx
            self.vy = new_vy
            self.x = new_x
            self.y = new_y

        return self.x, self.y, self.vx, self.vy


@dataclass
class DoublePendulum2D:
    """
    Chaotic double pendulum using RPN for integration.

    State:
        theta1, theta2 : angles of pendulum 1 and 2 (from vertical)
        omega1, omega2 : angular velocities
        L1, L2         : lengths
        m1, m2         : masses
        g              : gravity
        dt             : timestep

    Equations of motion (derived from Lagrangian):
        Complex coupled second-order ODEs; we compute accelerations in Python
        and delegate integration to RPN.

    This demonstrates a chaotic system where small changes in initial
    conditions lead to drastically different trajectories.
    """

    theta1: float
    theta2: float
    omega1: float
    omega2: float
    L1: float
    L2: float
    m1: float
    m2: float
    g: float
    dt: float
    _engine: ModularRPNEngine | None = None

    @property
    def engine(self) -> ModularRPNEngine:
        if self._engine is None:
            self._engine = ModularRPNEngine()
        return self._engine

    def _eval(self, expr: str) -> float:
        return self.engine.evaluate(expr)

    def step(self, n_steps: int = 1) -> tuple[float, float, float, float]:
        """
        Advance the double pendulum by n_steps.

        Returns:
            (theta1, theta2, omega1, omega2) after the last step.
        """
        for _ in range(n_steps):
            # Compute angular accelerations using double pendulum equations
            # (simplified Lagrangian mechanics)
            delta = self.theta2 - self.theta1
            sin_delta = math.sin(delta)
            cos_delta = math.cos(delta)

            denom1 = (self.m1 + self.m2) * self.L1 - self.m2 * self.L1 * cos_delta * cos_delta
            denom2 = (self.L2 / self.L1) * denom1

            # alpha1 (angular acceleration of pendulum 1)
            alpha1 = (
                self.m2 * self.L1 * self.omega1 * self.omega1 * sin_delta * cos_delta
                + self.m2 * self.g * math.sin(self.theta2) * cos_delta
                + self.m2 * self.L2 * self.omega2 * self.omega2 * sin_delta
                - (self.m1 + self.m2) * self.g * math.sin(self.theta1)
            ) / denom1

            # alpha2 (angular acceleration of pendulum 2)
            alpha2 = (
                -self.m2 * self.L2 * self.omega2 * self.omega2 * sin_delta * cos_delta
                + (self.m1 + self.m2) * self.g * math.sin(self.theta1) * cos_delta
                - (self.m1 + self.m2) * self.L1 * self.omega1 * self.omega1 * sin_delta
                - (self.m1 + self.m2) * self.g * math.sin(self.theta2)
            ) / denom2

            # Integrate using RPN
            expr_omega1 = f"{self.omega1} {alpha1} {self.dt} * +"
            expr_omega2 = f"{self.omega2} {alpha2} {self.dt} * +"
            new_omega1 = self._eval(expr_omega1)
            new_omega2 = self._eval(expr_omega2)

            expr_theta1 = f"{self.theta1} {new_omega1} {self.dt} * +"
            expr_theta2 = f"{self.theta2} {new_omega2} {self.dt} * +"
            new_theta1 = self._eval(expr_theta1)
            new_theta2 = self._eval(expr_theta2)

            self.omega1 = new_omega1
            self.omega2 = new_omega2
            self.theta1 = new_theta1
            self.theta2 = new_theta2

        return self.theta1, self.theta2, self.omega1, self.omega2


@dataclass
class CoupledOscillators:
    """
    Two coupled harmonic oscillators connected by a spring.

    State:
        x1, x2 : displacements from equilibrium
        v1, v2 : velocities
        k      : spring constant for each oscillator
        k_c    : coupling spring constant
        m1, m2 : masses
        dt     : timestep

    Equations:
        F1 = -k * x1 - k_c * (x1 - x2)
        F2 = -k * x2 - k_c * (x2 - x1)

        a1 = F1 / m1
        a2 = F2 / m2

    Discrete update:
        v_{t+1} = v_t + a * dt
        x_{t+1} = x_t + v_{t+1} * dt
    """

    x1: float
    x2: float
    v1: float
    v2: float
    k: float
    k_c: float
    m1: float
    m2: float
    dt: float
    _engine: ModularRPNEngine | None = None

    @property
    def engine(self) -> ModularRPNEngine:
        if self._engine is None:
            self._engine = ModularRPNEngine()
        return self._engine

    def _eval(self, expr: str) -> float:
        return self.engine.evaluate(expr)

    def step(self, n_steps: int = 1) -> tuple[float, float, float, float]:
        """
        Advance the coupled oscillators by n_steps.

        Returns:
            (x1, x2, v1, v2) after the last step.
        """
        for _ in range(n_steps):
            # Compute forces
            F1 = -self.k * self.x1 - self.k_c * (self.x1 - self.x2)
            F2 = -self.k * self.x2 - self.k_c * (self.x2 - self.x1)

            # Accelerations
            a1 = F1 / max(self.m1, 1e-12)
            a2 = F2 / max(self.m2, 1e-12)

            # Integrate velocities using RPN
            expr_v1 = f"{self.v1} {a1} {self.dt} * +"
            expr_v2 = f"{self.v2} {a2} {self.dt} * +"
            new_v1 = self._eval(expr_v1)
            new_v2 = self._eval(expr_v2)

            # Integrate positions using RPN
            expr_x1 = f"{self.x1} {new_v1} {self.dt} * +"
            expr_x2 = f"{self.x2} {new_v2} {self.dt} * +"
            new_x1 = self._eval(expr_x1)
            new_x2 = self._eval(expr_x2)

            self.v1 = new_v1
            self.v2 = new_v2
            self.x1 = new_x1
            self.x2 = new_x2

        return self.x1, self.x2, self.v1, self.v2


@dataclass
class RigidBody2D:
    """
    2D rigid body rotation under external torque.

    State:
        theta  : angle (orientation)
        omega  : angular velocity
        I      : moment of inertia
        tau    : applied torque
        dt     : timestep

    Equation:
        alpha = tau / I  (angular acceleration)

    Discrete update:
        omega_{t+1} = omega_t + alpha * dt
        theta_{t+1} = theta_t + omega_{t+1} * dt
    """

    theta: float
    omega: float
    I: float
    tau: float
    dt: float
    _engine: ModularRPNEngine | None = None

    @property
    def engine(self) -> ModularRPNEngine:
        if self._engine is None:
            self._engine = ModularRPNEngine()
        return self._engine

    def _eval(self, expr: str) -> float:
        return self.engine.evaluate(expr)

    def step(self, n_steps: int = 1) -> tuple[float, float]:
        """
        Advance the rigid body by n_steps.

        Returns:
            (theta, omega) after the last step.
        """
        for _ in range(n_steps):
            # Compute angular acceleration
            alpha = self.tau / max(self.I, 1e-12)

            # omega_{t+1} = omega_t + alpha * dt
            expr_omega = f"{self.omega} {alpha} {self.dt} * +"
            new_omega = self._eval(expr_omega)

            # theta_{t+1} = theta_t + omega_{t+1} * dt
            expr_theta = f"{self.theta} {new_omega} {self.dt} * +"
            new_theta = self._eval(expr_theta)

            self.omega = new_omega
            self.theta = new_theta

        return self.theta, self.omega


class PhysicsGalaxyDemo:
    """
    Minimal persistence layer for constant-acceleration systems.

    This is a lightweight "Physics Galaxy" demo that:
    - stores 1D constant-acceleration system parameters on disk as JSON,
    - reloads them into `ConstantAcceleration1D`,
    - can advance a named system by N steps and persist the updated state.

    It does NOT attempt to be a full Reality Enabler galaxy or glTF-backed
    House integration yet; it is a stepping stone that proves we can:
    - express laws as RPN programs,
    - attach parameters as persistent state,
    - round-trip the system between disk and math cores.
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        if root is None:
            # Default under Knowledge3D.local/physics_demo (sibling of repo root)
            repo_root = Path(__file__).resolve().parents[2]
            self.root = (repo_root / ".." / "Knowledge3D.local" / "physics_demo").resolve()
        else:
            self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _state_path(self, name: str) -> Path:
        safe = "".join(c for c in name if c.isalnum() or c in ("-", "_")) or "system"
        return self.root / f"{safe}.json"

    def save_system(self, name: str, system: ConstantAcceleration1D) -> None:
        """Persist system parameters to disk."""
        path = self._state_path(name)
        payload = {
            "position": float(system.position),
            "velocity": float(system.velocity),
            "acceleration": float(system.acceleration),
            "dt": float(system.dt),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_system(self, name: str) -> ConstantAcceleration1D:
        """Load system parameters from disk into a new ConstantAcceleration1D."""
        path = self._state_path(name)
        data = json.loads(path.read_text(encoding="utf-8"))
        return ConstantAcceleration1D(
            position=float(data["position"]),
            velocity=float(data["velocity"]),
            acceleration=float(data["acceleration"]),
            dt=float(data["dt"]),
        )

    def step_system(self, name: str, n_steps: int = 1) -> tuple[float, float]:
        """
        Load a system, advance it by n_steps using RPN, and persist the result.

        Returns:
            (position, velocity) after stepping.
        """
        system = self.load_system(name)
        pos, vel = system.step(n_steps=n_steps)
        self.save_system(name, system)
        return pos, vel
