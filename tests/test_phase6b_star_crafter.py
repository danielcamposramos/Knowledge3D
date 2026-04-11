from __future__ import annotations

from types import SimpleNamespace

from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC
from knowledge3d.ingestion.star_crafter import (
    build_default_star_crafter_program_table,
    build_foundational_star_crafter_outputs,
)


class FakeMatryoshkaEmbedder:
    def embed_stack(self, text: str) -> dict[int, list[float]]:
        seed = sum(ord(char) for char in str(text or ""))
        return {
            tier: [float(((seed + idx) % 17) + 1) for idx in range(tier)]
            for tier in (64, 128, 512, 2048)
        }


def _crafted_rows() -> list[dict]:
    return build_foundational_star_crafter_outputs(embedder=FakeMatryoshkaEmbedder())


def _rows_by_id(rows: list[dict]) -> dict[str, dict]:
    return {str(row["id"]): dict(row) for row in rows}


def _meaning_star(row: dict) -> dict:
    return dict(dict(row.get("metadata") or {}).get("meaning_star") or {})


def test_phase6b_star_crafter_counts_and_program_addrs() -> None:
    rows = _crafted_rows()
    by_id = _rows_by_id(rows)

    digit_ids = [row_id for row_id in by_id if row_id.startswith("concept_digit_")]
    operator_ids = [row_id for row_id in by_id if row_id.startswith("math_operator_")]
    program_ids = [row_id for row_id in by_id if row_id.startswith("rpn_program_")]
    word_ids = [row_id for row_id in by_id if row_id.startswith("word_digit_")]
    grammar_ids = [row_id for row_id in by_id if row_id.startswith("grammar_")]

    assert len(digit_ids) >= 10
    assert len(operator_ids) >= 5
    assert len(program_ids) >= 5
    assert len(word_ids) >= 10
    assert len(grammar_ids) >= 5

    for row_id in operator_ids:
        row = by_id[row_id]
        assert row["selection_role"] == "executor"
        assert bool(row["answer_eligible"]) is True
        assert int(row.get("meta_rule_addr") or 0) > 0

    for row_id in program_ids:
        assert int(by_id[row_id].get("meta_rule_addr") or 0) > 0


def test_phase6b_bidirectional_digit_links() -> None:
    rows = _crafted_rows()
    by_id = _rows_by_id(rows)
    digit_row = by_id["concept_digit_two"]
    digit_star = _meaning_star(digit_row)

    assert digit_star["surface_forms"]["en"]["word_ref"] == "word_digit_two_en"
    assert "char_digit_2" in digit_star["surface_forms"]["en"]["char_refs"]

    word_star = _meaning_star(by_id["word_digit_two_en"])
    char_star = _meaning_star(by_id["char_digit_2"])
    assert "concept_digit_two" in list(word_star.get("taxonomy_refs") or [])
    assert "concept_digit_two" in list(char_star.get("taxonomy_refs") or [])


def test_phase6b_operator_program_symlink_and_native_addr() -> None:
    rows = _crafted_rows()
    by_id = _rows_by_id(rows)
    add_row = by_id["math_operator_addition"]
    add_star = _meaning_star(add_row)
    program_row = by_id["rpn_program_addition"]
    program_star = _meaning_star(program_row)

    assert "rpn_program_addition" in list(add_star.get("meta_refs") or [])
    assert "math_operator_addition" in list(program_star.get("taxonomy_refs") or [])
    assert int(add_row.get("meta_rule_addr") or 0) == int(program_row.get("meta_rule_addr") or 0)
    assert int(add_row.get("program_opcode_count") or 0) == 5


def test_phase6b_content_hash_stability() -> None:
    left = _rows_by_id(_crafted_rows())
    right = _rows_by_id(_crafted_rows())

    for row_id in sorted(left):
        assert _meaning_star(left[row_id]).get("star_id") == _meaning_star(right[row_id]).get("star_id")


def test_phase6b_rows_carry_matryoshka_embedding_stack() -> None:
    rows = _crafted_rows()
    for row in rows:
        assert len(list(row.get("embedding") or [])) == 64
        assert len(list(row.get("embedding_tier_64") or [])) == 64
        assert len(list(row.get("embedding_tier_128") or [])) == 128
        assert len(list(row.get("embedding_tier_512") or [])) == 512
        assert len(list(row.get("embedding_tier_2048") or [])) == 2048
        assert len(list(row.get("embedding_max") or [])) == 2048
        assert list(row["embedding"]) == list(row["embedding_tier_64"])
        assert list(row["embedding_tier_2048"]) == list(row["embedding_max"])


def test_phase6b_program_table_layout_is_stable() -> None:
    table = build_default_star_crafter_program_table()
    assert table.size_bytes > 8
    assert table.offsets["rpn_program_addition"] > 0
    assert table.offsets["rpn_program_addition"] % 8 == 0
    assert table.lengths["rpn_program_addition"] == 5


def test_phase6b_tablet_binds_program_table_surface() -> None:
    calls: list[tuple[str, tuple]] = []

    class FakeBridge:
        def bind_query_runtime_buffers(self, **kwargs):
            calls.append(("query", tuple(sorted(kwargs.keys()))))

        def bind_galaxy_table(self, gpu_ptr, star_count, **kwargs):
            calls.append(("galaxy", (gpu_ptr, star_count)))

        def bind_program_table(self, gpu_ptr, size_bytes):
            calls.append(("program", (gpu_ptr, size_bytes)))

    fake_kv = SimpleNamespace(
        _trm_state_buffers={
            "d_q": 1,
            "d_y": 2,
            "d_z": 3,
            "d_z_new": 4,
            "d_y_new": 5,
            "d_workspace": 6,
        },
        _trm_weight_buffers={"W1": 7, "W2": 8, "W3": 9, "W4": 10},
        _matryoshka_bridge=None,
        _trm_matryoshka_weight_buffer=None,
        _sovereign_hot_path=SimpleNamespace(
            star_table=SimpleNamespace(gpu_ptr=123, star_count=25, _host_stars=[]),
            _host_stars=[],
            program_table=SimpleNamespace(gpu_ptr=456, size_bytes=128),
        ),
    )

    tablet = HeadlessTabletMPC(
        command_handler=lambda payload: payload,
        knowledgeverse=fake_kv,
        bridge=FakeBridge(),
    )

    assert tablet._bridge is not None
    assert ("program", (456, 128)) in calls
