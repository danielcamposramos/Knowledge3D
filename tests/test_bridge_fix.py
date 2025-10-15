#!/usr/bin/env python3
"""Quick test to verify bridge detection fix."""

import sys
sys.path.insert(0, '/workspace')

from knowledge3d.spatial.semantic_navigator import SemanticNavigator
import logging
import pytest

pytestmark = pytest.mark.skip(
    reason="SemanticNavigator uses deprecated CuPy-based spatial stack"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_bridge_detection():
    """Test that bridges are now detected with the fixed threshold."""

    logger.info("=" * 80)
    logger.info("Testing bridge detection with fixed threshold (0.7 instead of 0.85)")
    logger.info("=" * 80)

    # Initialize navigator in multi-domain mode
    navigator = SemanticNavigator(nav_mode="multi")

    # Load a knowledge graph
    # Use shapes_house.glb (4.6KB) which should be valid and small
    glb_path = "/workspace/viewer/public/shapes_house.glb"
    logger.info(f"Loading knowledge graph from: {glb_path}")
    navigator.load_house(glb_path)
    logger.info(f"Loaded {len(navigator.labels)} nodes")

    # Check multi-domain navigator stats
    if navigator.multi_domain_navigator:
        mdn = navigator.multi_domain_navigator
        logger.info(f"\n{'='*80}")
        logger.info(f"Multi-Domain Navigator Stats:")
        logger.info(f"  Total domains: {len(mdn.domains)}")
        logger.info(f"  Total bridges: {len(mdn.bridges)}")
        logger.info(f"  Bridge percentage: {100*len(mdn.bridges)/(len(navigator.labels)):.2f}%")
        logger.info(f"{'='*80}\n")

        if len(mdn.bridges) > 0:
            logger.info("✓ SUCCESS: Bridges detected! Cross-domain navigation is now possible.")
            return True
        else:
            logger.error("✗ FAILURE: Still 0 bridges. Domains remain disconnected.")
            return False
    else:
        logger.error("✗ FAILURE: Multi-domain navigator not initialized")
        return False

if __name__ == "__main__":
    try:
        success = test_bridge_detection()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
