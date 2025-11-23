import numpy as np

from knowledge3d.cranium.physics_demo import (
    ConstantAcceleration1D,
    HarmonicOscillator1D,
    Orbital2D,
    Heat1D,
    PhysicsGalaxyDemo,
)


def analytic_position_velocity(x0: float, v0: float, a: float, dt: float, n_steps: int) -> tuple[float, float]:
    """Closed-form solution for constant acceleration in 1D."""
    t = n_steps * dt
    v_t = v0 + a * t
    x_t = x0 + v0 * t + 0.5 * a * t * t
    return x_t, v_t


def test_constant_acceleration_rpn_matches_analytic():
    """
    Verify that the RPN-based update matches the analytic solution
    for constant acceleration to within a small numerical tolerance.
    """
    x0 = 0.0
    v0 = 1.0
    a = -9.81
    dt = 0.01
    steps = 100  # t = 1.0 second

    system = ConstantAcceleration1D(position=x0, velocity=v0, acceleration=a, dt=dt)
    x_rpn, v_rpn = system.step(n_steps=steps)

    x_true, v_true = analytic_position_velocity(x0, v0, a, dt, steps)

    assert np.isclose(v_rpn, v_true, atol=1e-5)
    # Position uses a discrete Euler-like integrator, so allow a small
    # integration error relative to the continuous-time analytic solution.
    assert np.isclose(x_rpn, x_true, atol=5e-2)


def analytic_harmonic_oscillator(x0: float, v0: float, omega: float, dt: float, n_steps: int) -> tuple[float, float]:
    """Closed-form solution for 1D harmonic oscillator."""
    t = n_steps * dt
    w = omega
    coswt = np.cos(w * t)
    sinwt = np.sin(w * t)
    x_t = x0 * coswt + (v0 / w) * sinwt
    v_t = -w * x0 * sinwt + v0 * coswt
    return x_t, v_t


def test_harmonic_oscillator_rpn_matches_analytic():
    """
    Verify that the RPN-based harmonic oscillator matches the analytic
    solution within a reasonable integration tolerance.
    """
    x0 = 1.0
    v0 = 0.0
    omega = 1.0
    dt = 0.001
    steps = 2000  # t = 2.0 seconds

    system = HarmonicOscillator1D(position=x0, velocity=v0, omega=omega, dt=dt)
    x_rpn, v_rpn = system.step(n_steps=steps)

    x_true, v_true = analytic_harmonic_oscillator(x0, v0, omega, dt, steps)

    # With small dt the Euler-like integrator should track the analytic
    # solution reasonably well.
    assert np.isclose(v_rpn, v_true, atol=5e-3)
    assert np.isclose(x_rpn, x_true, atol=5e-3)


def test_orbital_2d_rpn_preserves_radius_and_energy_approximately():
    """
    Sanity check for a 2D orbit under central gravity:
    - Starts from approximate circular orbit.
    - After integration, radius and total energy should remain close
      to their initial values, despite using a simple integrator.
    """
    # Circular orbit approximation: mu=1, r=1, v=1
    x0, y0 = 1.0, 0.0
    vx0, vy0 = 0.0, 1.0
    mu = 1.0
    dt = 0.001
    steps = 5000  # t = 5 seconds

    def energy(x: float, y: float, vx: float, vy: float, mu_val: float) -> float:
        r = np.sqrt(x * x + y * y)
        v2 = vx * vx + vy * vy
        return 0.5 * v2 - mu_val / max(r, 1e-12)

    system = Orbital2D(x=x0, y=y0, vx=vx0, vy=vy0, mu=mu, dt=dt)
    e0 = energy(x0, y0, vx0, vy0, mu)
    x_f, y_f, vx_f, vy_f = system.step(n_steps=steps)
    e_f = energy(x_f, y_f, vx_f, vy_f, mu)

    r0 = np.sqrt(x0 * x0 + y0 * y0)
    rf = np.sqrt(x_f * x_f + y_f * y_f)

    # Radius should remain near 1.0 within a moderate tolerance.
    assert np.isclose(rf, r0, atol=0.1)
    # Total energy should be approximately conserved.
    assert np.isclose(e_f, e0, atol=0.1)


def test_heat1d_diffuses_peak_and_preserves_total_energy():
    """
    1D heat diffusion sanity check:
    - Start with a delta-like peak in the center.
    - After several steps, the peak should spread (central value decreases,
      neighbors increase) while total "heat" remains approximately constant.
    """
    N = 9
    alpha = 0.1
    dx = 1.0
    dt = 0.05
    steps = 50

    T0 = np.zeros(N, dtype=np.float32)
    center = N // 2
    T0[center] = 1.0

    system = Heat1D(temperature=T0, alpha=alpha, dx=dx, dt=dt)
    T_final = system.step(n_steps=steps)

    # Central value should have decreased; neighbors should have increased above 0.
    assert T_final[center] < 1.0
    assert T_final[center - 1] > 0.0
    assert T_final[center + 1] > 0.0

    # Total heat should be approximately conserved.
    assert np.isclose(T_final.sum(), T0.sum(), atol=1e-2)


def test_physics_galaxy_demo_roundtrip(tmp_path):
    """
    Ensure PhysicsGalaxyDemo can persist and reload a system, and that stepping
    via the demo produces the same result as stepping the in-memory system.
    """
    root = tmp_path / "physics_demo"
    galaxy = PhysicsGalaxyDemo(root=root)

    name = "test_system"
    x0 = 2.0
    v0 = -3.0
    a = 1.5
    dt = 0.02
    steps = 50

    system = ConstantAcceleration1D(position=x0, velocity=v0, acceleration=a, dt=dt)
    galaxy.save_system(name, system)

    # Step in memory
    x_mem, v_mem = system.step(n_steps=steps)

    # Step via persisted system
    x_disk, v_disk = galaxy.step_system(name, n_steps=steps)

    assert np.isclose(v_mem, v_disk, atol=1e-6)
    assert np.isclose(x_mem, x_disk, atol=1e-6)
