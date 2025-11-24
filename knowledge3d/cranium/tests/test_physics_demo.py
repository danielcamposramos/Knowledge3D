import numpy as np

from knowledge3d.cranium.physics_demo import (
    ConstantAcceleration1D,
    HarmonicOscillator1D,
    Orbital2D,
    Heat1D,
    Heat2D,
    Projectile2D,
    DoublePendulum2D,
    CoupledOscillators,
    RigidBody2D,
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


def test_heat2d_diffuses_peak_and_preserves_total_energy():
    """
    2D heat diffusion sanity check:
    - Start with a delta-like peak in the center of a 2D grid.
    - After several steps, the peak spreads while total heat remains
      approximately constant.
    """
    H, W = 9, 9
    alpha = 0.1
    dx = 1.0
    dt = 0.02
    steps = 50

    T0 = np.zeros((H, W), dtype=np.float32)
    ci, cj = H // 2, W // 2
    T0[ci, cj] = 1.0

    system = Heat2D(temperature=T0, alpha=alpha, dx=dx, dt=dt)
    T_final = system.step(n_steps=steps)

    # Central value should decrease; neighbors in 4-neighborhood should increase.
    assert T_final[ci, cj] < 1.0
    assert T_final[ci + 1, cj] > 0.0
    assert T_final[ci - 1, cj] > 0.0
    assert T_final[ci, cj + 1] > 0.0
    assert T_final[ci, cj - 1] > 0.0

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


# ============================================================================
# Phase 4A: Classical Mechanics Expansion Tests
# ============================================================================


def test_projectile2d_no_drag_matches_analytic():
    """
    Test projectile motion with zero drag against analytic solution.
    With k=0, this reduces to standard projectile motion.
    """
    # Initial conditions
    x0, y0 = 0.0, 0.0
    vx0, vy0 = 10.0, 20.0
    g = 9.81
    k = 0.0  # No drag
    dt = 0.001
    steps = 1000  # t = 1.0 second

    system = Projectile2D(x=x0, y=y0, vx=vx0, vy=vy0, g=g, k=k, dt=dt)
    x_rpn, y_rpn, vx_rpn, vy_rpn = system.step(n_steps=steps)

    # Analytic solution
    t = steps * dt
    x_true = x0 + vx0 * t
    y_true = y0 + vy0 * t - 0.5 * g * t * t
    vx_true = vx0
    vy_true = vy0 - g * t

    # Velocity should match closely (no drag)
    assert np.isclose(vx_rpn, vx_true, atol=1e-5)
    assert np.isclose(vy_rpn, vy_true, atol=1e-3)

    # Position with Euler integration
    assert np.isclose(x_rpn, x_true, atol=1e-2)
    assert np.isclose(y_rpn, y_true, atol=1e-2)


def test_projectile2d_with_drag_velocity_decreases():
    """
    Test that drag causes velocity to decrease over time.
    With positive k, total velocity magnitude should monotonically decrease.
    """
    x0, y0 = 0.0, 0.0
    vx0, vy0 = 10.0, 10.0
    g = 9.81
    k = 0.1  # Significant drag
    dt = 0.01
    steps = 100

    system = Projectile2D(x=x0, y=y0, vx=vx0, vy=vy0, g=g, k=k, dt=dt)
    v0_mag = np.sqrt(vx0 * vx0 + vy0 * vy0)

    x_f, y_f, vx_f, vy_f = system.step(n_steps=steps)
    v_f_mag = np.sqrt(vx_f * vx_f + vy_f * vy_f)

    # Velocity magnitude should decrease due to drag
    assert v_f_mag < v0_mag


def test_double_pendulum_energy_drift_bounded():
    """
    Test that double pendulum total energy drift remains bounded.
    Due to Euler integration, energy won't be perfectly conserved,
    but drift should remain within reasonable tolerance for short simulations.
    """
    # Start with small angles (near linear regime for stability)
    theta1 = 0.1
    theta2 = 0.2
    omega1 = 0.0
    omega2 = 0.0
    L1, L2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0
    g = 9.81
    dt = 0.001
    steps = 1000  # t = 1.0 second

    def total_energy(th1, th2, om1, om2, l1, l2, m1_val, m2_val, g_val):
        """Compute total energy of double pendulum."""
        # Kinetic energy
        KE1 = 0.5 * m1_val * (l1 * om1) ** 2
        KE2 = 0.5 * m2_val * (
            (l1 * om1) ** 2 + (l2 * om2) ** 2 + 2 * l1 * l2 * om1 * om2 * np.cos(th1 - th2)
        )
        KE = KE1 + KE2

        # Potential energy (taking pivot as zero)
        PE1 = -m1_val * g_val * l1 * np.cos(th1)
        PE2 = -m2_val * g_val * (l1 * np.cos(th1) + l2 * np.cos(th2))
        PE = PE1 + PE2

        return KE + PE

    system = DoublePendulum2D(
        theta1=theta1, theta2=theta2, omega1=omega1, omega2=omega2,
        L1=L1, L2=L2, m1=m1, m2=m2, g=g, dt=dt
    )

    E0 = total_energy(theta1, theta2, omega1, omega2, L1, L2, m1, m2, g)
    th1_f, th2_f, om1_f, om2_f = system.step(n_steps=steps)
    E_f = total_energy(th1_f, th2_f, om1_f, om2_f, L1, L2, m1, m2, g)

    # Energy drift should be bounded (Euler integration is not symplectic)
    # Allow 10% drift for this simple integrator over 1 second
    assert np.abs(E_f - E0) < 0.1 * np.abs(E0)


def test_double_pendulum_nonlinear_behavior():
    """
    Test that double pendulum exhibits nonlinear behavior distinct from
    uncoupled pendulums. For large initial angles, the coupled system
    should evolve differently than two independent pendulums.
    """
    # Large angles to ensure nonlinear coupling effects
    theta1 = 1.5  # ~86 degrees
    theta2 = -1.2  # ~-69 degrees
    omega1 = 0.5
    omega2 = -0.3
    L1, L2 = 1.0, 1.0
    m1, m2 = 1.0, 1.0
    g = 9.81
    dt = 0.001
    steps = 1000  # t = 1.0 second

    # Evolve coupled system
    system_coupled = DoublePendulum2D(
        theta1=theta1, theta2=theta2, omega1=omega1, omega2=omega2,
        L1=L1, L2=L2, m1=m1, m2=m2, g=g, dt=dt
    )
    th1_coupled, th2_coupled, om1_coupled, om2_coupled = system_coupled.step(n_steps=steps)

    # Evolve two independent simple pendulums (as reference)
    # For independent pendulums: alpha = -(g/L) * sin(theta)
    def evolve_simple_pendulum(th0, om0, g_val, L_val, dt_val, n_steps):
        th, om = th0, om0
        for _ in range(n_steps):
            alpha = -(g_val / L_val) * np.sin(th)
            om_new = om + alpha * dt_val
            th_new = th + om_new * dt_val
            om, th = om_new, th_new
        return th, om

    th1_indep, om1_indep = evolve_simple_pendulum(theta1, omega1, g, L1, dt, steps)
    th2_indep, om2_indep = evolve_simple_pendulum(theta2, omega2, g, L2, dt, steps)

    # Coupled system should deviate from independent evolution
    # due to coupling forces
    deviation1 = np.abs(th1_coupled - th1_indep)
    deviation2 = np.abs(th2_coupled - th2_indep)

    # At least one pendulum should show measurable deviation from
    # independent motion due to coupling (>0.01 rad ≈ 0.57 degrees)
    assert (deviation1 > 0.01 or deviation2 > 0.01)


def test_coupled_oscillators_normal_modes():
    """
    Test coupled oscillators for normal mode behavior.
    In symmetric case (k=k_c, m1=m2), system exhibits two normal modes:
    - In-phase mode: both oscillate together
    - Out-of-phase mode: oscillate oppositely
    """
    # Symmetric system
    k = 1.0
    k_c = 1.0
    m1 = 1.0
    m2 = 1.0
    dt = 0.001
    steps = 1000  # t = 1.0 second

    # Test in-phase mode: x1(0) = x2(0), v1(0) = v2(0) = 0
    # Should remain in-phase
    system_in_phase = CoupledOscillators(
        x1=1.0, x2=1.0, v1=0.0, v2=0.0, k=k, k_c=k_c, m1=m1, m2=m2, dt=dt
    )
    x1_in, x2_in, _, _ = system_in_phase.step(n_steps=steps)

    # In-phase: x1 ≈ x2 throughout
    assert np.isclose(x1_in, x2_in, atol=1e-3)

    # Test out-of-phase mode: x1(0) = -x2(0), v1(0) = v2(0) = 0
    # Should remain out-of-phase
    system_out_phase = CoupledOscillators(
        x1=1.0, x2=-1.0, v1=0.0, v2=0.0, k=k, k_c=k_c, m1=m1, m2=m2, dt=dt
    )
    x1_out, x2_out, _, _ = system_out_phase.step(n_steps=steps)

    # Out-of-phase: x1 ≈ -x2 throughout
    assert np.isclose(x1_out, -x2_out, atol=1e-2)


def test_coupled_oscillators_energy_conservation():
    """
    Test that total energy of coupled oscillators is approximately conserved.
    """
    x1 = 1.0
    x2 = -0.5
    v1 = 0.5
    v2 = -0.3
    k = 1.0
    k_c = 0.5
    m1 = 1.0
    m2 = 1.5
    dt = 0.001
    steps = 2000

    def total_energy(x1_val, x2_val, v1_val, v2_val, k_val, k_c_val, m1_val, m2_val):
        """Compute total energy of coupled oscillators."""
        KE = 0.5 * m1_val * v1_val ** 2 + 0.5 * m2_val * v2_val ** 2
        PE_springs = 0.5 * k_val * (x1_val ** 2 + x2_val ** 2)
        PE_coupling = 0.5 * k_c_val * (x1_val - x2_val) ** 2
        return KE + PE_springs + PE_coupling

    system = CoupledOscillators(x1=x1, x2=x2, v1=v1, v2=v2, k=k, k_c=k_c, m1=m1, m2=m2, dt=dt)
    E0 = total_energy(x1, x2, v1, v2, k, k_c, m1, m2)

    x1_f, x2_f, v1_f, v2_f = system.step(n_steps=steps)
    E_f = total_energy(x1_f, x2_f, v1_f, v2_f, k, k_c, m1, m2)

    # Energy should be approximately conserved
    assert np.isclose(E_f, E0, rtol=5e-3)


def test_rigid_body_constant_torque_matches_analytic():
    """
    Test rigid body rotation under constant torque against analytic solution.
    With constant torque τ:
        α = τ / I
        ω(t) = ω₀ + α * t
        θ(t) = θ₀ + ω₀ * t + 0.5 * α * t²
    """
    theta0 = 0.0
    omega0 = 0.0
    I = 2.0
    tau = 1.5
    dt = 0.001
    steps = 1000  # t = 1.0 second

    system = RigidBody2D(theta=theta0, omega=omega0, I=I, tau=tau, dt=dt)
    theta_rpn, omega_rpn = system.step(n_steps=steps)

    # Analytic solution
    t = steps * dt
    alpha = tau / I
    omega_true = omega0 + alpha * t
    theta_true = theta0 + omega0 * t + 0.5 * alpha * t * t

    # Check against analytic
    assert np.isclose(omega_rpn, omega_true, atol=1e-3)
    assert np.isclose(theta_rpn, theta_true, atol=1e-3)


def test_rigid_body_angular_momentum_conservation():
    """
    Test that with zero torque, angular momentum L = I * ω is conserved.
    """
    theta0 = 0.5
    omega0 = 2.0
    I = 3.0
    tau = 0.0  # No external torque
    dt = 0.01
    steps = 500

    system = RigidBody2D(theta=theta0, omega=omega0, I=I, tau=tau, dt=dt)
    L0 = I * omega0

    theta_f, omega_f = system.step(n_steps=steps)
    L_f = I * omega_f

    # Angular momentum should be exactly conserved with zero torque
    assert np.isclose(L_f, L0, atol=1e-6)

    # Angular velocity should remain constant
    assert np.isclose(omega_f, omega0, atol=1e-6)
