from __future__ import annotations

from typing import List


class ThinkingTagEmbedder:
    """Sovereign stub for the thinking-tag embedder.

    The prior implementation was a torch ``nn.Module`` (Linear+Dropout+ReLU+
    Sigmoid) loaded from a ``.pth`` checkpoint. Per the 2026-04-18 absolute
    sovereignty purge, torch is banned from the hot path. The sovereign
    successor is a ternary-quantised embedding model executed through the PTX
    thinking-tag kernel using ``.trit`` weights; until that lands, this stub
    preserves the import surface so callers (``knowledge3d/bridge/live_server.py``)
    keep parsing and simply get an empty tag list.

    Spec pointer: ``TEMP/CLAUDE_ABSOLUTE_SOVEREIGNTY_PURGE_04.18.2026.md`` §5.4.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_tags: int = 100) -> None:
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_tags = int(num_tags)

    def load_state_dict(self, *_args, **_kwargs) -> None:
        # Torch weight-loading is replaced by the sovereign ternary weight
        # loader. Until that loader is wired in, accept any input and keep the
        # stub silent so boot does not abort.
        return None

    def eval(self) -> "ThinkingTagEmbedder":
        return self

    def forward(self, *_args, **_kwargs):  # pragma: no cover - sovereign successor pending
        raise NotImplementedError(
            "sovereign successor pending — see "
            "TEMP/CLAUDE_ABSOLUTE_SOVEREIGNTY_PURGE_04.18.2026.md §5.4. "
            "Replace with a PTX ternary-embedding forward pass."
        )

    def predict_thinking_tags(self, embedding: List[float], tag_names: List[str]) -> List[str]:
        # Until the sovereign PTX successor is wired, predict no tags. The
        # caller (live_server.py) treats an empty list as "no tag annotations
        # available" and continues without failing.
        return []
