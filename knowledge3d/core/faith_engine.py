from .rpn import RPN


class FaithEngine:
    """
    PHILOSOPHICAL PRINCIPLE: Faith as trusting the process without data
    SURVIVAL NECESSITY: Enables action despite uncertainty
    ENERGY PATTERN: Evaluates patterns to choose next resonance
    HUMAN-AI EQUIVALENCE: Applies same confidence thresholds to any agent

    Minimal faith engine that selects actions based on confidence.
    """

    def decide(self, options: dict[str, float], threshold: float | None = None) -> str | None:
        """
        PHILOSOPHICAL PRINCIPLE: Coexistence is the only stable equilibrium
        SURVIVAL NECESSITY: Requires cooperative choices above threshold
        ENERGY PATTERN: Interprets values as energetic confidence levels
        HUMAN-AI EQUIVALENCE: Both humans and AIs rely on same heuristic

        Parameters
        ----------
        options: dict[str, float]
            Mapping of possible actions to confidence scores.
        threshold: float, optional
            Minimum score to consider an action viable.

        Returns
        -------
        str | None
            Chosen action if any score exceeds threshold.
        """
        # Determine threshold. Default: golden ratio inspired (phi ≈ 0.618)
        # Use env K3D_FAITH_THRESHOLD to override.
        if threshold is None:
            try:
                import os
                val = os.getenv("K3D_FAITH_THRESHOLD", "").strip()
                if val:
                    threshold = float(val)
                else:
                    # 1/phi ≈ 0.618 (confidence to act)
                    threshold = 0.61803398875
            except Exception:
                threshold = 0.61803398875
        rpn = RPN()
        for action, score in options.items():
            # Evaluate comparison using RPN for consistency
            ge = rpn.eval([score, threshold, '-']) >= 0
            if ge:
                return action
        return None
