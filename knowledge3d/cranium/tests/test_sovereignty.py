import sys

from knowledge3d.cranium.reality_galaxy import RealityGalaxy
from knowledge3d.cranium.reality_physics_export import (
    export_double_pendulum_2d,
    export_harmonic_oscillator_1d,
)


def test_hot_path_no_numpy():
    """Verify numpy NOT loaded during hot path."""
    if "numpy" in sys.modules:
        del sys.modules["numpy"]

    galaxy = RealityGalaxy()
    system = export_harmonic_oscillator_1d()
    galaxy.add_node(system)
    galaxy.step_system(system.node_id, n_steps=10)

    assert "numpy" not in sys.modules, "SOVEREIGNTY VIOLATION: numpy in hot path"


def test_hot_path_no_torch():
    """Verify PyTorch NOT loaded during hot path."""
    if "torch" in sys.modules:
        del sys.modules["torch"]

    galaxy = RealityGalaxy()
    system = export_double_pendulum_2d()
    galaxy.add_node(system)
    galaxy.step_system(system.node_id, n_steps=5)

    assert "torch" not in sys.modules, "SOVEREIGNTY VIOLATION: torch in hot path"


def test_gltf_export_can_use_numpy():
    """Verify export path (outside hot path) CAN use numpy."""
    from knowledge3d.cranium.reality_gltf_export import export_system_to_gltf
    from knowledge3d.cranium.reality_physics_export import export_water_molecule
    import tempfile

    system = export_water_molecule()
    with tempfile.NamedTemporaryFile(suffix=".glb") as f:
        export_system_to_gltf(system, f.name)
