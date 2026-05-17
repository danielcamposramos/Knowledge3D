"""WINE-pattern adapters for external 3D models (ingestion-only).

Post-purge surface (2026-04-18):
    Five wine-adapter modules were moved to ``Old_Attempts/2026-04-18/``
    per ``TEMP/CLAUDE_ABSOLUTE_SOVEREIGNTY_PURGE_04.18.2026.md`` §4.1:

        trellis_wine_adapter          (TRELLISWineAdapter)
        hunyuan_world_wine_adapter    (HunyuanWorldWineAdapter)
        zero_copy_bridge              (ZeroCopyBridge)
        external_model_router         (ExternalModelRouter)
        wine_adapter_factory          (WineAdapterFactory)

    These adapters were numpy-native and coupled the hot path to external
    model outputs. Per Daniel's ruling
    ``feedback_tablet_wine_still_python_orchestration.md`` (2026-04-15),
    Tablet WINE must NOT reintroduce Python orchestration. The surviving
    ``procedural_content_bridge`` stays as an ingestion-only translator
    that emits sovereign RPN tokens — nothing runtime-hot.
"""

from .procedural_content_bridge import ProceduralContentBridge  # noqa: F401

__all__ = ["ProceduralContentBridge"]
