from __future__ import annotations

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


def test_chat_query_uses_composed_sovereign_head(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_composed_head")
    result = kv.execute_task(
        task={
            "type": "CHAT_TASK",
            "prompt": "What is the speed of light?",
            "query": "What is the speed of light?",
            "messages": [{"role": "user", "content": "What is the speed of light?"}],
        },
        route={"specialist": "chat", "galaxy_names": ["Grammar", "Word", "Character"]},
        specialist="chat",
        domain_hint="general",
    )

    trace = list(result.get("reasoning_trace", []))
    assert result["status"] == "ok"
    assert result["gpu_execution"] is True
    assert result["runtime"] == "knowledgeverse_gpu_query"
    assert any("Navigator parse:" in step for step in trace)
    assert any("Morton locate" in step for step in trace)
    assert any("Frustum cull" in step for step in trace)
    assert any("Dynamic LOD" in step for step in trace)
    assert any("Nine-chain swarm dispatch" in step for step in trace)
    assert any("Halting gate" in step for step in trace)


def test_factual_property_query_reaches_reality_entry_on_composed_head(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_composed_head_factual")
    query = "What is the boiling point of water?"
    result = kv.execute_task(
        task={
            "type": "CHAT_TASK",
            "task_id": "fact",
            "prompt": query,
            "query": query,
            "question": query,
        },
        route={"specialist": "chat", "galaxy_names": ["Grammar", "Word", "Character"]},
        specialist="chat",
        domain_hint="general",
    )

    trace = list(result.get("reasoning_trace", []))
    assert result["status"] == "ok"
    assert result["gpu_execution"] is True
    assert result["match"]["id"] == "reality_water_boiling_point_standard"
    assert result["answer"] == "100 C (212 F) at 1 atm"
    assert any("Halting gate: halt" in step for step in trace)


def test_arc_query_reaches_multi_worker_halting_on_composed_head(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_composed_head_arc")
    expected = [
        [3, 2, 3, 2, 3, 2],
        [7, 8, 7, 8, 7, 8],
        [2, 3, 2, 3, 2, 3],
        [8, 7, 8, 7, 8, 7],
        [3, 2, 3, 2, 3, 2],
        [7, 8, 7, 8, 7, 8],
    ]
    result = kv.execute_task(
        task={
            "type": "ARC_TASK",
            "task_id": "00576224",
            "query": "solve arc transformation task",
            "training_examples": [{"input": [[3, 2], [7, 8]], "output": expected}],
            "input_grid": [[3, 2], [7, 8]],
            "expected_output": expected,
        },
        route={"specialist": "visual", "galaxy_names": ["Drawing", "Grammar", "Tool"]},
        specialist="visual",
        domain_hint="visual",
    )

    trace = list(result.get("reasoning_trace", []))
    assert result["status"] == "ok"
    assert result["gpu_execution"] is True
    assert result["output_grid"] == expected
    assert result["match"]["id"] == "arc_eval_00576224"
    assert any("Halting gate: halt" in step for step in trace)


def test_arc_swarm_paths_use_distinct_transform_programs(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_arc_swarm_paths")

    paths = kv._build_gpu_reasoning_paths(
        task_type="ARC_TASK",
        primary_program_id=Knowledgeverse.GPU_ARC_REASONING_PROGRAM_ID,
        query_text="solve arc transformation task",
    )

    program_ids = [str(path.get("program_id", "")) for path in paths]
    assert len(program_ids) == 9
    assert len(set(program_ids)) == 9
    assert "reasoning_arc_tile_repeat_top1" in program_ids
    assert "reasoning_arc_recolor_top1" in program_ids
    assert "reasoning_arc_separator_bridge_top1" in program_ids


def test_mmlu_paths_allocate_hypothesis_and_validation_workers(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_mmlu_paths")

    paths = kv._build_gpu_reasoning_paths(
        task_type="MMLU_TASK",
        primary_program_id="reasoning_elimination_top1",
        query_text="Which option best matches the concept?",
        options=["A", "B", "C", "D"],
    )

    assert len(paths) == 9
    assert sum(1 for path in paths if path.get("path_role") == "hypothesis") == 4
    assert sum(1 for path in paths if path.get("path_role") == "validation") == 4
    assert {int(path.get("worker_slot", -1)) for path in paths[:9]} == set(range(9))


def test_lhe_paths_allocate_option_and_cross_validation_workers(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_lhe_paths")

    paths = kv._build_gpu_reasoning_paths(
        task={"type": "LHE_TASK", "options": ["A", "B", "C", "D"]},
        task_type="LHE_TASK",
        primary_program_id="reasoning_elimination_top1",
        query_text="Which option best answers the question?",
        options=["A", "B", "C", "D"],
    )

    assert len(paths) == 9
    assert sum(1 for path in paths if path.get("path_role") == "hypothesis") == 4
    assert sum(1 for path in paths if path.get("path_role") == "validation") == 4
    assert sum(1 for path in paths if path.get("path_role") == "cross_validation") == 1


def test_mmlu_subject_anchor_context_uses_galaxy_metadata(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_mmlu_anchor_context")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Reality", "Math", "Grammar", "Word", "Character"])

    embedding, anchor_ids, anchor_galaxies = kv._mmlu_subject_anchor_context(
        subject_hint="college_physics",
        target_galaxies=["Reality", "Math", "Grammar", "Word", "Character"],
        base_embedding=[1.0] + [0.0] * 15,
    )

    assert len(embedding) == 16
    assert "reality_anchor_college_physics_core" in anchor_ids
    assert "Reality" in anchor_galaxies


def test_mmlu_rule_metadata_and_subject_anchors_bootstrap(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_mmlu_bootstrap")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Reality", "Math", "Grammar", "Word", "Character"])

    assert kv._mmlu_option_rule_weights() == (0.18, 0.02)
    assert kv._mmlu_relative_gap_threshold() == 0.01

    embedding, anchor_ids, anchor_galaxies = kv._mmlu_subject_anchor_context(
        subject_hint="abstract_algebra",
        target_galaxies=["Math", "Grammar"],
        base_embedding=[1.0] + [0.0] * 15,
    )

    assert len(embedding) == 16
    assert any("abstract_algebra" in anchor_id for anchor_id in anchor_ids)
    assert "Math" in anchor_galaxies


def test_mmlu_option_support_scores_literal_anchor_answers(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_mmlu_option_support")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Math", "Grammar"])
    entry = kv._catalog_entry_by_id("math_concept_abstract_algebra_field_extension_degree")

    assert entry is not None
    assert kv._mmlu_option_support_score(entry, "4") == 1.0
    assert kv._mmlu_option_support_score(entry, "6") == 0.0


def test_parse_override_meta_rules_bootstrap(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_parse_override_rules")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Grammar"])

    assert kv._parse_override_weight("meta_rule_parse_override_algebra", 0.0) == 0.8
    assert kv._parse_override_weight("meta_rule_parse_override_domain", 0.0) == 0.7


def test_lhe_subject_anchor_context_uses_domain_metadata(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_lhe_anchor_context")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Reality", "Math", "Grammar", "Word", "Character"])

    embedding, anchor_ids, anchor_galaxies = kv._subject_anchor_context(
        subject_hint="philosophy",
        target_galaxies=["Reality", "Math", "Grammar", "Word", "Character"],
        base_embedding=[1.0] + [0.0] * 15,
        match_mode="domain",
    )

    assert len(embedding) == 16
    assert any(anchor_id.startswith("reality_philosophy_") for anchor_id in anchor_ids)
    assert "Reality" in anchor_galaxies


def test_goal_type_rows_are_isolated_by_family(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_goal_type_family")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Grammar"])

    navigator = kv.trm_navigator.navigator_specialist
    gsm8k_rows = navigator._goal_type_rows(base_route={"galaxy_names": ["Grammar"], "goal_type_family": "gsm8k"})
    lhe_rows = navigator._goal_type_rows(base_route={"galaxy_names": ["Grammar"], "goal_type_family": "lhe"})
    math_rows = navigator._goal_type_rows(base_route={"galaxy_names": ["Grammar"], "goal_type_family": "math"})

    assert gsm8k_rows
    assert lhe_rows
    assert math_rows == []
    assert all(
        str((row.get("metadata") or {}).get("subfield", "")).strip().lower() == "word_problem_binding"
        for row in gsm8k_rows
    )
    assert all(
        str((row.get("metadata") or {}).get("subfield", "")).strip().lower() == "lhe_goal_typing"
        for row in lhe_rows
    )


def test_regular_math_navigation_blocks_math_and_lhe_grammar_rows(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_navigation_filter")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Math", "Grammar"])

    gsm8k_goal = kv._catalog_entry_by_id("goal_type_total_cost")
    lhe_goal = kv._catalog_entry_by_id("goal_type_factual_recall")
    lhe_factual_anchor = kv._catalog_entry_by_id("lhe_factual_math_spin_bordism_bg2_dim12")
    arithmetic_shortcut = kv._catalog_entry_by_id("math_arithmetic_add_7_7")
    linear_shortcut = kv._catalog_entry_by_id("math_linear_ax_plus_b_eq_c_5_1_46")
    math_task = {
        "type": "MATH_TASK",
        "competition": "MATH:Algebra",
        "query": "Compute the binomial coefficient C(10,3).",
    }

    assert gsm8k_goal is not None
    assert lhe_goal is not None
    assert lhe_factual_anchor is not None
    assert arithmetic_shortcut is not None
    assert linear_shortcut is not None
    assert not kv._benchmark_navigation_entry_allowed(
        entry=gsm8k_goal,
        task_type="MATH_TASK",
        task=math_task,
        query_text=str(math_task["query"]),
    )
    assert not kv._benchmark_navigation_entry_allowed(
        entry=lhe_goal,
        task_type="MATH_TASK",
        task=math_task,
        query_text=str(math_task["query"]),
    )
    assert not kv._benchmark_navigation_entry_allowed(
        entry=lhe_factual_anchor,
        task_type="MATH_TASK",
        task=math_task,
        query_text=str(math_task["query"]),
    )
    assert not kv._benchmark_navigation_entry_allowed(
        entry=arithmetic_shortcut,
        task_type="MATH_TASK",
        task=math_task,
        query_text=str(math_task["query"]),
    )
    assert not kv._benchmark_navigation_entry_allowed(
        entry=linear_shortcut,
        task_type="MATH_TASK",
        task=math_task,
        query_text=str(math_task["query"]),
    )


def test_lhe_factual_anchors_bootstrap_into_reality_and_math(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_lhe_factual_anchors")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Reality", "Math"])

    reality_anchor = kv._catalog_entry_by_id("lhe_factual_chess_mate_in_two_rxf3_rf1")
    math_anchor = kv._catalog_entry_by_id("lhe_factual_math_resolvent_conormal_space_schwarzschild")

    assert reality_anchor is not None
    assert math_anchor is not None
    assert str((reality_anchor.get("metadata") or {}).get("subfield", "")).strip().lower() == "lhe_factual_anchor"
    assert str((math_anchor.get("metadata") or {}).get("subfield", "")).strip().lower() == "lhe_factual_anchor"


def test_math_concepts_and_quantity_roles_bootstrap_into_grammar(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_math_concepts_bootstrap")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Grammar"])

    for entry_id in (
        "math_concept_continuity",
        "math_concept_function_evaluation",
        "math_concept_coordinate_center",
        "math_concept_progression_constraint",
        "math_concept_rate_balance",
        "quantity_role_initial",
        "quantity_role_delta",
        "quantity_role_rate",
        "quantity_role_target",
        "quantity_role_total",
        "quantity_role_part",
        "quantity_role_count",
        "quantity_role_duration",
        "quantity_role_divisor",
        "quantity_role_percentage",
        "quantity_role_threshold",
        "quantity_role_excess",
        "goal_type_total_earnings",
        "goal_type_remaining_after_spending",
        "goal_type_total_cost",
        "goal_type_percentage_result",
        "goal_type_total_combined_quantity",
        "goal_type_factual_recall",
        "goal_type_elimination",
        "goal_type_multi_hop",
        "goal_type_temporal_reasoning",
        "operation_pattern_ratio_then_add",
        "operation_pattern_multiply_chain_sum",
        "operation_pattern_base_plus_excess",
        "operation_pattern_total_minus_parts",
        "operation_pattern_percentage_change",
    ):
        entry = kv._catalog_entry_by_id(entry_id)
        assert entry is not None
        assert entry["galaxy"] == "Grammar"

    rate_entry = kv._catalog_entry_by_id("quantity_role_rate")
    part_entry = kv._catalog_entry_by_id("quantity_role_part")
    assert isinstance(rate_entry.get("metadata"), dict)
    assert isinstance(part_entry.get("metadata"), dict)
    assert rate_entry["metadata"].get("structural_cues")
    assert part_entry["metadata"].get("structural_cues")


def test_lhe_benchmark_direct_entries_removed_from_reasoning_facts(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_lhe_benchmark_cleanup")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Reality"])

    for removed_id in (
        "benchmark_lhe_arrhenius_weak_non_sadism_direct",
        "benchmark_lhe_chess_mate_in_two_direct",
        "benchmark_lhe_spin_bordism_bg2_direct",
        "benchmark_lhe_poincare_polynomial_direct",
        "benchmark_lhe_gamma_factor_direct",
        "benchmark_lhe_kk_count_direct",
        "benchmark_lhe_resolvent_conormal_direct",
    ):
        assert kv._catalog_entry_by_id(removed_id) is None

    for retained_id in (
        "concept_philosophy_arrhenius_weak_non_sadism",
        "concept_trivia_yeyo_concatenation",
        "concept_chess_mate_in_two_black_queens_stationary",
        "concept_cybersecurity_two_step_substitution_plaintext_katie",
    ):
        entry = kv._catalog_entry_by_id(retained_id)
        assert entry is not None
        metadata = dict(entry.get("metadata") or {})
        assert "answer" not in metadata
        assert metadata.get("question")
        assert metadata.get("query_anchor")


def test_lhe_option_prompt_text_avoids_boilerplate() -> None:
    prompt = Knowledgeverse._lhe_option_prompt_text("Which law explains inertia?", "Force")
    assert prompt == "Which law explains inertia? Force"


def test_arc_family_rules_and_math_patterns_bootstrap_into_clean_catalog(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_arc_math_bootstrap")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Drawing", "Grammar", "Math"])

    drawing_entries = (
        "arc_signature_identity_family",
        "arc_signature_rotate_or_transpose_family",
        "arc_signature_mirror_family",
        "arc_signature_color_remap_family",
        "arc_signature_shape_resize_family",
        "arc_signature_object_count_change_family",
        "arc_signature_compositional_family",
    )
    grammar_entries = (
        "arc_rule_identity_family",
        "arc_rule_rotate_or_transpose_family",
        "arc_rule_mirror_family",
        "arc_rule_color_remap_family",
        "arc_rule_shape_resize_family",
        "arc_rule_object_count_change_family",
        "arc_rule_compositional_family",
        "math_rule_piecewise_continuity_boundary",
        "math_rule_balance_positive_negative_days",
        "math_rule_circle_center_complete_square",
        "math_rule_circle_center_diameter_midpoint",
        "math_rule_function_substitution_composition",
        "math_rule_arithmetic_sequence_grid_constraint",
        "math_rule_compound_interest_present_value",
        "math_rule_rate_scaling_inverse",
        "math_rule_floor_interval_accumulation",
        "math_concept_piecewise_defined_function",
        "math_concept_factor_pair_constraint",
        "math_concept_floor_function_intervals",
        "math_concept_function_iteration",
        "math_concept_exchange_balance",
        "math_concept_compound_growth",
        "math_rule_polynomial_degree_selection",
        "math_rule_band_formation_factor_pair",
        "math_rule_direct_substitution_numeric_eval",
        "math_rule_exchange_rate_shortfall",
        "math_rule_piecewise_iteration_branch",
    )
    math_support_entries = (
        "math_algebra_band_formation_anchor",
        "math_algebra_polynomial_degree_anchor",
        "math_algebra_ceiling_floor_chain_anchor",
        "math_algebra_balance_linear_anchor",
        "math_algebra_circle_center_equation_anchor",
        "math_algebra_interval_inequality_anchor",
        "math_algebra_function_substitution_anchor",
        "math_algebra_l_shaped_sequence_anchor",
        "math_algebra_compound_interest_anchor",
        "math_algebra_rate_scaling_anchor",
    )

    for entry_id in drawing_entries:
        entry = kv._catalog_entry_by_id(entry_id)
        assert entry is not None
        assert entry["galaxy"] == "Drawing"

    for entry_id in grammar_entries:
        entry = kv._catalog_entry_by_id(entry_id)
        assert entry is not None
        assert entry["galaxy"] == "Grammar"

    for entry_id in math_support_entries:
        entry = kv._catalog_entry_by_id(entry_id)
        assert entry is not None
        assert entry["galaxy"] == "Math"


def test_reality_bootstrap_subject_labels_and_domain_anchors(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_reality_subject_bootstrap")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Reality"])

    expected_subjects = {
        "reality_constant_speed_of_light": "physics",
        "reality_constant_avogadro_constant": "chemistry",
        "reality_bio_dna": "biology",
        "reality_element_oxygen": "chemistry",
        "reality_kinematics_position_update_euler": "physics",
        "reality_proc_lsystem_expand": "computer_science",
        "reality_anchor_chemistry_domain_core": "chemistry",
        "reality_anchor_biology_domain_core": "biology",
        "reality_anchor_computer_science_core": "computer_science",
        "reality_anchor_astronomy_core": "astronomy",
        "reality_philosophy_epistemology": "philosophy",
        "reality_history_world_wars": "history",
        "reality_economics_supply_demand": "economics",
        "reality_cs_algorithmic_complexity": "computer_science",
        "reality_astronomy_cosmology": "astronomy",
        "reality_cybersecurity_cia_triad": "cybersecurity",
        "reality_chess_mating_patterns": "chess",
    }

    for entry_id, expected in expected_subjects.items():
        entry = kv._catalog_entry_by_id(entry_id)
        assert entry is not None
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        subject = entry.get("subject") or metadata.get("subject")
        assert subject == expected

    catalog = kv.get_gpu_galaxy_catalog()
    subject_counts: dict[str, int] = {}
    for entry in catalog:
        if entry.get("galaxy") != "Reality":
            continue
        metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
        subject = str(entry.get("subject") or metadata.get("subject") or "unknown")
        subject_counts[subject] = subject_counts.get(subject, 0) + 1

    assert subject_counts.get("chemistry", 0) >= 48
    assert subject_counts.get("biology", 0) >= 29
    assert subject_counts.get("computer_science", 0) >= 195
    assert subject_counts.get("astronomy", 0) >= 15
    assert subject_counts.get("philosophy", 0) >= 17
    assert subject_counts.get("history", 0) >= 12
    assert subject_counts.get("economics", 0) >= 12
    assert subject_counts.get("cybersecurity", 0) >= 8
    assert subject_counts.get("chess", 0) >= 9
    assert subject_counts.get("physics", 0) < 300


def test_lhe_route_preserves_reality_targets(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_lhe_route")

    targets = kv._resolve_gpu_target_galaxies(
        route={"specialist": "chat", "galaxy_names": ["Reality", "Grammar", "Word", "Character"]},
        task={
            "type": "LHE_TASK",
            "query": "Which condition of Arrhenius's sixth impossibility theorem do critical-level views violate?",
        },
    )

    assert targets == ["Reality", "Grammar", "Word", "Character", "Math"]


def test_benchmark_math_anchor_rows_do_not_embed_answers(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_benchmark_math_anchor_rows")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Reality", "Math", "Grammar", "Word", "Character"])
    entry = kv._catalog_entry_by_id("benchmark_math_math_0_direct")

    assert entry is not None
    assert entry["name"] != "18"
    assert entry["answer_text"] != "18"
    assert "answer" not in (entry.get("metadata") or {})
    assert not entry.get("template_ref")
    assert not entry.get("template_params")


def test_benchmark_honesty_filter_suppresses_lhe_exact_shortcuts(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_lhe_honesty_filter")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Reality", "Math", "Grammar", "Word", "Character"])
    query = "Which condition of Arrhenius's sixth impossibility theorem do critical-level views violate?"

    exact_candidates = kv._lhe_exact_question_navigation_candidates(
        query_text=query,
        reference_embedding=kv._embed_query_gpu(query),
    )
    filtered_candidates, suppressed = kv._filter_benchmark_shortcut_candidates(
        candidates=exact_candidates,
        task={
            "type": "LHE_TASK",
            "task_id": "lhe_arrhenius",
            "query": query,
            "question": query,
            "expected_answer": "Weak Non-Sadism",
        },
        query_text=query,
    )

    assert exact_candidates
    assert suppressed > 0
    assert not filtered_candidates


def test_mmlu_query_uses_gre_specialist_dispatch(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_composed_head_mmlu")
    result = kv.execute_task(
        task={
            "type": "MMLU_TASK",
            "task_id": "mmlu_smoke",
            "prompt": "An object at rest remains at rest unless acted on by which quantity?",
            "query": "An object at rest remains at rest unless acted on by which quantity?",
            "question": "An object at rest remains at rest unless acted on by which quantity?",
            "options": ["Force", "Mass", "Time", "Temperature"],
        },
        route={"specialist": "chat", "domain_hint": "college_physics", "galaxy_names": ["Reality", "Math", "Grammar"]},
        specialist="chat",
        domain_hint="college_physics",
    )

    trace = list(result.get("reasoning_trace", []))
    assert result["status"] == "ok"
    assert result["gpu_execution"] is True
    assert result["answer"] == "Force"
    assert any("MMLU anchor: hit college_physics" in step for step in trace)
    assert any("GRE specialist dispatch" in step for step in trace)


def test_mmlu_query_reuses_shared_navigation_candidates(tmp_path, monkeypatch) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_composed_head_mmlu_shared")
    compose_calls = 0
    original = kv._compose_head_navigation_candidates

    def wrapped(*args, **kwargs):
        nonlocal compose_calls
        compose_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(kv, "_compose_head_navigation_candidates", wrapped)

    result = kv.execute_task(
        task={
            "type": "MMLU_TASK",
            "task_id": "mmlu_shared_nav_smoke",
            "prompt": "What is 7 * (3 + 2)?",
            "query": "What is 7 * (3 + 2)?",
            "question": "What is 7 * (3 + 2)?",
            "options": ["35", "30", "42", "28"],
            "subject": "high_school_mathematics",
        },
        route={"specialist": "chat", "domain_hint": "high_school_mathematics", "galaxy_names": ["Reality", "Math", "Grammar"]},
        specialist="chat",
        domain_hint="high_school_mathematics",
    )

    assert result["status"] == "ok"
    assert compose_calls == 1
