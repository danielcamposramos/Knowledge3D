"""
Physics Galaxy demo built on ProceduralGalaxy + RPN math core.

This module extends the constant-acceleration demo to use the existing
ProceduralGalaxy/ProceduralCompiler infrastructure, so that a tiny
physical system state can be:

- encoded as a short embedding vector,
- compressed into a procedural program (PD0x codec),
- stored as a .ppr program under a dedicated physics root,
- reconstructed back into a live system on demand.

This is not yet a full `reality_*` galaxy backed by glTF, but it matches
the Reality Enabler design at the storage/program level and proves that
simulation state can live in the same procedural store as other stars.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from knowledge3d.cranium.procedural_galaxy import ProceduralGalaxy
from knowledge3d.cranium.physics_demo import ConstantAcceleration1D


DEFAULT_PHYSICS_GALAXY_ROOT = Path("/K3D/Knowledge3D.local/procedural_galaxy/physics")


@dataclass
class PhysicsGalaxy:
    """
    Minimal Physics Galaxy using ProceduralGalaxy as backing store.

    For now, it only handles 1D constant-acceleration systems by packing
    (position, velocity, acceleration, dt) into a 4D embedding vector.

    That vector is:
      - compressed via ProceduralCompiler (PD0x),
      - stored as a program under `root/<name>.ppr`,
      - reconstructed via `execute_program` when needed.
    """

    root: Path = DEFAULT_PHYSICS_GALAXY_ROOT

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._galaxy = ProceduralGalaxy(root=self.root)

    def store_constant_accel_system(self, name: str, system: ConstantAcceleration1D) -> None:
        """
        Store a constant-acceleration system as a 4D embedding program.

        Args:
            name: Key under which to store the system.
            system: ConstantAcceleration1D instance.
        """
        vec = np.array(
            [system.position, system.velocity, system.acceleration, system.dt],
            dtype=np.float32,
        )
        # Use ProceduralGalaxy compiler: it will choose PD04/PD02/etc as appropriate.
        program = self._galaxy.compiler.compile_embedding(vec)
        program_bytes = program.to_bytes()
        compression_ratio = float(vec.nbytes) / max(1, len(program_bytes))

        metadata = {
            "type": "constant_acceleration_1d",
            "dims": 4,
        }
        self._galaxy.store_program(name, program_bytes, compression_ratio, metadata=metadata)

    def load_constant_accel_system(self, name: str) -> ConstantAcceleration1D:
        """
        Load a constant-acceleration system from its stored program.

        Args:
            name: Key previously used with `store_constant_accel_system`.

        Returns:
            ConstantAcceleration1D reconstructed from the stored embedding.
        """
        embedding = self._galaxy.execute_program(name)
        if embedding.size < 4:
            raise ValueError("Stored embedding too small for constant-acceleration system.")
        x, v, a, dt = map(float, embedding[:4])
        return ConstantAcceleration1D(position=x, velocity=v, acceleration=a, dt=dt)

