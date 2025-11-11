"""
Math Galaxy - One of the three basic galaxies always loaded to the stage.

The Math Galaxy is a first-class galaxy that manages mathematical symbols
and semantic operations as procedural programs. It uses the existing
ProceduralGalaxy infrastructure for compression and storage.

Structure:
    /K3D/Knowledge3D.local/procedural_galaxy/math/
    ├── symbols/       # Visual + linguistic embeddings (∑.ppr, ∫.ppr, ...)
    └── operations/    # Semantic RPN programs (sum.ppr, integral.ppr, ...)

Integration:
    - Math symbols trained using EXISTING character training pipeline
    - Embeddings stored using EXISTING ProceduralCompiler (69-80:1 compression)
    - Operations executed using EXISTING AdvancedRPNEngine (Tier-3 RPN)

Usage:
    >>> from knowledge3d.cranium.math_galaxy import MathGalaxy
    >>> galaxy = MathGalaxy()
    >>>
    >>> # Store trained symbol embedding
    >>> galaxy.store_symbol('∑', embedding_128d)
    >>>
    >>> # Load symbol embedding
    >>> embedding = galaxy.load_symbol('∑')
    >>>
    >>> # Store semantic operation program
    >>> galaxy.store_operation('sum', summation_rpn_program)
    >>>
    >>> # Load operation for execution
    >>> program = galaxy.load_operation('sum')
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .procedural_compiler import ProceduralCompiler
from .procedural_galaxy import ProceduralGalaxy


DEFAULT_MATH_GALAXY_ROOT = Path("/K3D/Knowledge3D.local/procedural_galaxy/math")


class MathGalaxy:
    """
    Math Galaxy - stores mathematical symbols and operations separately.

    The Math Galaxy is one of three basic galaxies always loaded to the Stage:
    1. Language Galaxy (characters, words, meanings)
    2. Math Galaxy (mathematical symbols, semantic operations)
    3. Programs Galaxy (RPN procedural programs)

    Symbols:
        Visual + linguistic embeddings trained via existing character pipeline.
        Stored as compressed .ppr files (69-80:1 compression).

    Operations:
        Semantic RPN programs that define mathematical operations (∑, ∫, ∂, etc.)
        Stored as .ppr files containing opcode sequences.
        Executed via AdvancedRPNEngine (Tier-3 programmable RPN kernel).
    """

    def __init__(self, root: Optional[Path] = None) -> None:
        """
        Initialize Math Galaxy with separate storage for symbols and operations.

        Args:
            root: Root directory for Math Galaxy. Defaults to:
                  /K3D/Knowledge3D.local/procedural_galaxy/math/
        """
        self.root = Path(root) if root else DEFAULT_MATH_GALAXY_ROOT
        self.symbols_dir = self.root / "symbols"
        self.operations_dir = self.root / "operations"

        # Create directory structure
        self.symbols_dir.mkdir(parents=True, exist_ok=True)
        self.operations_dir.mkdir(parents=True, exist_ok=True)

        # Use ProceduralGalaxy infrastructure for storage (no reinvention)
        self.symbol_galaxy = ProceduralGalaxy(root=self.symbols_dir)
        self.operation_galaxy = ProceduralGalaxy(root=self.operations_dir)

        logging.info(
            "Math Galaxy initialized: symbols=%s operations=%s",
            self.symbols_dir,
            self.operations_dir,
        )

    # ------------------------------------------------------------------ #
    # Symbol Storage (Visual + Linguistic Embeddings)
    # ------------------------------------------------------------------ #

    def store_symbol(self, symbol: str, embedding: np.ndarray) -> None:
        """
        Store trained symbol embedding (visual + linguistic).

        Uses existing ProceduralCompiler for compression (69-80:1 ratio).

        Args:
            symbol: Mathematical symbol (e.g., '∑', '∫', '∂')
            embedding: Trained embedding vector (typically 128D float32)

        Example:
            >>> galaxy.store_symbol('∑', trained_embedding_128d)
            >>> # Stored as: math/symbols/∑.ppr (~7 bytes vs 512 bytes)
        """
        compiler = ProceduralCompiler()
        program = compiler.compile_embedding(embedding)
        program_bytes = program.to_bytes()
        compression = float(embedding.nbytes) / max(1, len(program_bytes))

        self.symbol_galaxy.store_program(symbol, program_bytes, compression)

        logging.info(
            "Math symbol stored: %s (%d bytes → %d bytes, %.1f:1)",
            symbol,
            embedding.nbytes,
            len(program_bytes),
            compression,
        )

    def load_symbol(self, symbol: str) -> np.ndarray:
        """
        Load symbol embedding from Math Galaxy.

        Decompresses .ppr file and reconstructs embedding vector.

        Args:
            symbol: Mathematical symbol (e.g., '∑', '∫', '∂')

        Returns:
            Reconstructed embedding vector (typically 128D float32)

        Raises:
            FileNotFoundError: If symbol not trained/stored

        Example:
            >>> embedding = galaxy.load_symbol('∑')
            >>> print(embedding.shape)  # (128,)
        """
        return self.symbol_galaxy.execute_program(symbol)

    def has_symbol(self, symbol: str) -> bool:
        """
        Check if symbol is trained and stored in Math Galaxy.

        Args:
            symbol: Mathematical symbol to check

        Returns:
            True if symbol embedding exists, False otherwise
        """
        program_path = self.symbols_dir / f"{self._sanitize_key(symbol)}.ppr"
        return program_path.exists()

    # ------------------------------------------------------------------ #
    # Operation Storage (Semantic RPN Programs)
    # ------------------------------------------------------------------ #

    def store_operation(self, operation_name: str, program_bytes: bytes) -> None:
        """
        Store semantic operation program (RPN opcodes).

        Operations are executable RPN programs that define mathematical semantics.
        Executed via AdvancedRPNEngine (Tier-3 programmable kernel).

        Args:
            operation_name: Operation identifier (e.g., 'sum', 'integral', 'derivative')
            program_bytes: Compiled RPN opcode sequence

        Example:
            >>> # Summation: ∑ from i=a to b of f(i)
            >>> summation_opcodes = [OP_LOOP, OP_RECALL, OP_ADD, OP_STORE, OP_BRANCH]
            >>> program_bytes = compile_opcode_program(summation_opcodes)
            >>> galaxy.store_operation('sum', program_bytes)
        """
        self.operation_galaxy.store_program(operation_name, program_bytes, compression_ratio=1.0)

        logging.info(
            "Math operation stored: %s (%d bytes)",
            operation_name,
            len(program_bytes),
        )

    def load_operation(self, operation_name: str) -> bytes:
        """
        Load semantic operation program for execution.

        Returns raw .ppr bytes containing RPN opcode sequence.
        Parse with ProceduralProgram.from_bytes() to extract opcodes.

        Args:
            operation_name: Operation identifier

        Returns:
            Program bytes (RPN opcode sequence)

        Raises:
            FileNotFoundError: If operation not defined/stored

        Example:
            >>> program_bytes = galaxy.load_operation('sum')
            >>> program = ProceduralProgram.from_bytes(program_bytes)
            >>> opcodes = parse_program_opcodes(program)
            >>> # Execute with AdvancedRPNEngine
        """
        program = self.operation_galaxy.load_program(operation_name)
        return program.to_bytes()

    def has_operation(self, operation_name: str) -> bool:
        """
        Check if operation is defined and stored in Math Galaxy.

        Args:
            operation_name: Operation identifier to check

        Returns:
            True if operation program exists, False otherwise
        """
        program_path = self.operations_dir / f"{self._sanitize_key(operation_name)}.ppr"
        return program_path.exists()

    # ------------------------------------------------------------------ #
    # Discovery & Listing
    # ------------------------------------------------------------------ #

    def list_symbols(self) -> list[str]:
        """
        List all trained math symbols in Math Galaxy.

        Returns:
            List of symbol strings (e.g., ['∑', '∫', '∂', '√', ...])

        Example:
            >>> symbols = galaxy.list_symbols()
            >>> print(f"Math Galaxy has {len(symbols)} symbols trained")
        """
        if not self.symbols_dir.exists():
            return []
        return [p.stem for p in self.symbols_dir.glob("*.ppr")]

    def list_operations(self) -> list[str]:
        """
        List all defined semantic operations in Math Galaxy.

        Returns:
            List of operation names (e.g., ['sum', 'integral', 'derivative', ...])

        Example:
            >>> operations = galaxy.list_operations()
            >>> print(f"Math Galaxy has {len(operations)} operations defined")
        """
        if not self.operations_dir.exists():
            return []
        return [p.stem for p in self.operations_dir.glob("*.ppr")]

    def get_stats(self) -> dict[str, any]:
        """
        Get Math Galaxy statistics.

        Returns:
            Dictionary with:
                - num_symbols: Number of trained symbols
                - num_operations: Number of defined operations
                - symbols_dir: Path to symbols directory
                - operations_dir: Path to operations directory
        """
        return {
            "num_symbols": len(self.list_symbols()),
            "num_operations": len(self.list_operations()),
            "symbols_dir": str(self.symbols_dir),
            "operations_dir": str(self.operations_dir),
        }

    # ------------------------------------------------------------------ #
    # Internal Helpers
    # ------------------------------------------------------------------ #

    def _sanitize_key(self, key: str) -> str:
        """Sanitize key for filesystem storage (from ProceduralGalaxy pattern)."""
        safe = "".join(c for c in key if c.isalnum() or c in ("-", "_"))
        return safe or "unknown"

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"MathGalaxy(symbols={stats['num_symbols']}, "
            f"operations={stats['num_operations']}, "
            f"root={self.root})"
        )


__all__ = ["MathGalaxy", "DEFAULT_MATH_GALAXY_ROOT"]
