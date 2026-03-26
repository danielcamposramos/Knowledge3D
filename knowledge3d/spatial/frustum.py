"""
Legacy import path that now routes to the sovereign frustum implementation.
"""
from knowledge3d.cranium.spatial_sovereign.frustum import (
    FrustumCuller,
    create_perspective_matrix,
    create_view_matrix,
    matmul_4x4,
    matvec_4,
)

__all__ = ["FrustumCuller", "create_perspective_matrix", "create_view_matrix", "matmul_4x4", "matvec_4"]
