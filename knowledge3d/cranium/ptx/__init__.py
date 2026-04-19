"""PTX kernel sources — post-purge stub.

Post-purge surface (2026-04-18):
    The Python binding wrappers were moved to ``Old_Attempts/2026-04-18/``
    per ``TEMP/CLAUDE_ABSOLUTE_SOVEREIGNTY_PURGE_04.18.2026.md`` §4.1:

        ptx_ops         (PTX_OPS, PTXOps)
        arc_ops         (ARC_PTX_OPS, ARCPTXOps)
        galaxy_buffer   (GalaxyGPUMemory, MeshRecord, load/save helpers)
        geometry_ops    (PTXGeometryOps)
        modality_ops    (PTXModalityOps)

    The ``.ptx`` and ``.cu`` kernel sources in this directory are
    unchanged and remain the sovereign interface. Load them through
    ``knowledge3d.cranium.sovereign.loader``:

        from knowledge3d.cranium.sovereign.loader import (
            ensure_init, load_ptx_file, get_function, launch,
        )
        ensure_init()
        mod = load_ptx_file(Path(__file__).parent / "morton_octree.ptx")
        fn  = get_function(mod, "morton_octree_build")

    The old wrapper classes were index structures on top of this surface;
    the kernels themselves are still the ground truth.
"""

__all__: list[str] = []
