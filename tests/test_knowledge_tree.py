from __future__ import annotations

from knowledge3d.cranium.bridges.mesh_bridge import MeshBridge
from knowledge3d.knowledgeverse.house_knowledge_tree import KNOWLEDGE_TREE_BRANCHES


def test_knowledge_tree_branches_have_valid_rpn() -> None:
    bridge = MeshBridge()
    for branch in KNOWLEDGE_TREE_BRANCHES:
        assert branch.meaning_class in {"branch", "leaf"}
        result = bridge.execute_rpn_program(branch.visual_rpn or "")
        assert result.mesh.vertices


def test_knowledge_tree_branches_reference_domains() -> None:
    branch_ids = {branch.star_id for branch in KNOWLEDGE_TREE_BRANCHES if branch.meaning_class == "branch"}
    assert len(branch_ids) >= 5


def test_knowledge_tree_leaves_reference_concepts() -> None:
    for branch in KNOWLEDGE_TREE_BRANCHES:
        if branch.meaning_class == "leaf":
            assert branch.taxonomy_refs
