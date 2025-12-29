"""Language expansion for Grammar Galaxy (text rules)."""

from .tier1_top10 import get_tier1_rules
from .tier2_next20 import get_tier2_rules
from .tier3_next20 import get_tier3_rules

__all__ = ["get_tier1_rules", "get_tier2_rules", "get_tier3_rules"]
