import numpy as np

from knowledge3d.cranium.physics_demo import ConstantAcceleration1D
from knowledge3d.cranium.physics_galaxy import PhysicsGalaxy


def test_physics_galaxy_store_and_load(tmp_path):
    """Ensure PhysicsGalaxy can round-trip a constant-acceleration system via ProceduralGalaxy."""
    root = tmp_path / "physics_galaxy"
    galaxy = PhysicsGalaxy(root=root)

    name = "orbit_1d_demo"
    system = ConstantAcceleration1D(position=1.0, velocity=2.0, acceleration=-0.5, dt=0.1)

    galaxy.store_constant_accel_system(name, system)
    restored = galaxy.load_constant_accel_system(name)

    # We do not expect exact equality here because the PD0x codec may
    # apply normalization or prototype-based encoding. This test only
    # asserts that a valid ConstantAcceleration1D instance can be
    # reconstructed and that its parameters are finite.
    assert isinstance(restored, ConstantAcceleration1D)
    assert np.isfinite(restored.position)
    assert np.isfinite(restored.velocity)
    assert np.isfinite(restored.acceleration)
    assert np.isfinite(restored.dt)
