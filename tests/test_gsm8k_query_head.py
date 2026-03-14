from __future__ import annotations

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


GSM8K_0_QUESTION = (
    "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and "
    "bakes muffins for her friends every day with four. She sells the remainder at the "
    "farmers' market daily for $2 per fresh duck egg. How much in dollars does she make "
    "every day at the farmers' market?"
)

GSM8K_RELATION_QUESTION = (
    "A robe takes 2 bolts of blue fiber and half that much white fiber. "
    "How many bolts in total does it take?"
)

GSM8K_REPEAT_QUESTION = (
    "James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. "
    "How many total meters does he run a week?"
)

GSM8K_GOAL_QUESTION = (
    "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep "
    "as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle "
    "has 20 sheep?"
)

GSM8K_EARNINGS_QUESTION = (
    "Weng earns $12 an hour for babysitting. On Monday, she babysat for 50 minutes. "
    "On Tuesday, she babysat for 30 minutes. On Wednesday, she babysat for 2 hours. "
    "How much did Weng earn in total?"
)


def test_gsm8k_route_expands_into_grammar_number_and_word(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_route")

    targets = kv._resolve_gpu_target_galaxies(
        route={"specialist": "math", "galaxy_names": ["Math"]},
        task={
            "type": "MATH_TASK",
            "competition": "GSM8K",
            "query": GSM8K_0_QUESTION,
        },
    )

    assert targets == ["Math", "Grammar", "Number", "Word"]


def test_gsm8k_reasoning_paths_register_word_problem_fission(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_paths")
    parse_bundle = kv._collect_parse_bundle(
        GSM8K_0_QUESTION,
        specialist="math",
        galaxy_names=["Math", "Grammar", "Number", "Word"],
        domain_hint="math",
    )

    paths = kv._build_gpu_reasoning_paths(
        task={
            "type": "MATH_TASK",
            "competition": "GSM8K",
            "query": GSM8K_0_QUESTION,
        },
        task_type="MATH_TASK",
        primary_program_id=Knowledgeverse.GPU_MATH_REASONING_PROGRAM_ID,
        query_text=GSM8K_0_QUESTION,
        parse_bundle=parse_bundle,
    )

    assert len(paths) == 9
    assert paths[0]["program_id"] == "reasoning_word_problem_fission"
    assert paths[0]["parse_strategy"] == "forward"
    assert paths[0]["composition_strategy"] == "forward_chain"
    assert any(path["parse_strategy"] == "backward" for path in paths)
    assert any(path["composition_strategy"] == "fusion_chain" for path in paths)
    assert any(path["composition_strategy"] == "clause_chain" for path in paths)
    assert any(path["composition_strategy"] == "goal_adjusted_chain" for path in paths)
    assert any(path["composition_strategy"] == "alt_add" for path in paths)
    assert any(path["composition_strategy"] == "alt_sub" for path in paths)
    assert any(path["composition_strategy"] == "alt_mul" for path in paths)
    assert any(path["composition_strategy"] == "alt_div" for path in paths)


def test_gsm8k_math_program_query_text_skips_template_hint(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_program_text")

    query_text = kv._program_query_text(
        GSM8K_0_QUESTION,
        Knowledgeverse.GPU_MATH_REASONING_PROGRAM_ID,
        task={
            "type": "MATH_TASK",
            "competition": "GSM8K",
            "query": GSM8K_0_QUESTION,
        },
    )

    assert query_text == GSM8K_0_QUESTION


def test_gsm8k_parse_bundle_collects_fusion_quantities(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_parse_bundle")

    parse_bundle = kv._collect_parse_bundle(
        GSM8K_0_QUESTION,
        specialist="math",
        galaxy_names=["Math", "Grammar", "Number", "Word"],
        domain_hint="math",
    )

    assert parse_bundle["fusion_parse"]["quantity_values"][:4] == [16.0, 3.0, 4.0, 2.0]


def test_gsm8k_parse_bundle_keeps_relation_and_repeated_quantities(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_parse_bundle_relations")

    relation_bundle = kv._collect_parse_bundle(
        GSM8K_RELATION_QUESTION,
        specialist="math",
        galaxy_names=["Math", "Grammar", "Number", "Word"],
        domain_hint="math",
    )
    repeat_bundle = kv._collect_parse_bundle(
        GSM8K_REPEAT_QUESTION,
        specialist="math",
        galaxy_names=["Math", "Grammar", "Number", "Word"],
        domain_hint="math",
    )
    goal_bundle = kv._collect_parse_bundle(
        GSM8K_GOAL_QUESTION,
        specialist="math",
        galaxy_names=["Math", "Grammar", "Number", "Word"],
        domain_hint="math",
    )

    assert relation_bundle["fusion_parse"]["quantity_values"][:2] == [2.0, 0.5]
    assert repeat_bundle["fusion_parse"]["quantity_values"][:3] == [3.0, 3.0, 60.0]
    assert goal_bundle["fusion_parse"]["quantity_values"][:3] == [2.0, 4.0, 20.0]


def test_gsm8k_backward_goal_type_assigns_rate_and_duration(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_goal_roles")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Math", "Grammar", "Number", "Word"])

    parse_bundle = kv._collect_parse_bundle(
        GSM8K_EARNINGS_QUESTION,
        specialist="math",
        galaxy_names=["Math", "Grammar", "Number", "Word"],
        domain_hint="math",
    )

    merged = parse_bundle["fusion_parse"]["merged_quantities"]
    roles = [str(row.get("role", "")).strip().lower() for row in merged]

    assert parse_bundle["fusion_parse"].get("goal_type") == "total_earnings"
    assert "rate" in roles
    assert "duration" in roles
    assert any(float(row.get("value", 0.0)) == 12.0 and str(row.get("role", "")).strip().lower() == "rate" for row in merged)


def test_gsm8k_total_cost_goal_types_base_quantity_as_count(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_total_cost_roles")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Math", "Grammar", "Number", "Word"])

    parse_bundle = kv._collect_parse_bundle(
        GSM8K_RELATION_QUESTION,
        specialist="math",
        galaxy_names=["Math", "Grammar", "Number", "Word"],
        domain_hint="math",
    )

    merged = parse_bundle["fusion_parse"]["merged_quantities"]

    assert parse_bundle["fusion_parse"].get("goal_type") == "total_cost"
    assert any(float(row.get("value", 0.0)) == 2.0 and str(row.get("role", "")).strip().lower() == "count" for row in merged)


def test_gsm8k_context_uses_navigator_quantity_order(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_context")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Math", "Grammar", "Number", "Word"])
    parse_bundle = kv._collect_parse_bundle(
        GSM8K_0_QUESTION,
        specialist="math",
        galaxy_names=["Math", "Grammar", "Number", "Word"],
        domain_hint="math",
    )

    context = kv._gsm8k_word_problem_context(
        target_galaxies=["Math", "Grammar", "Number", "Word"],
        base_embedding=kv._embed_query_gpu(GSM8K_0_QUESTION),
        parse_bundle=parse_bundle,
    )

    assert context["number_values"][:4] == [16.0, 3.0, 4.0, 2.0]
    assert context["number_ids"][:4] == ["num_16", "num_3", "num_4", "num_2"]
    assert context["quantity_role_values"].get("rate")
    assert context["quantity_role_values"].get("part")
    assert isinstance(context.get("role_map_variants"), list)
    assert len(context["role_map_variants"]) >= 2
    assert context["pattern_rows"]
    top_metadata = context["pattern_rows"][0].get("metadata") if isinstance(context["pattern_rows"][0].get("metadata"), dict) else {}
    assert top_metadata.get("rpn_template")
    assert top_metadata.get("role_slots")


def test_gsm8k_benchmark_shortcuts_are_suppressed_during_evaluation(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_honesty_filter")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Math", "Grammar", "Number", "Word"])
    entry = kv._catalog_entry_by_id("benchmark_math_gsm8k_0_direct")

    filtered_candidates, suppressed = kv._filter_benchmark_shortcut_candidates(
        candidates=[
            {
                "match": dict(entry or {}),
                "similarity": 1.0,
                "lod_saliency": 1.0,
                "lod_level": 2,
                "lod_focus": 1.0,
                "led_focus": 1.0,
                "led_path": [],
            }
        ],
        task={
            "type": "MATH_TASK",
            "task_id": "gsm8k_0",
            "competition": "GSM8K",
            "query": GSM8K_0_QUESTION,
            "expected_answer": "18",
        },
        query_text=GSM8K_0_QUESTION,
    )

    assert entry is not None
    assert suppressed == 1
    assert filtered_candidates == []


def test_gsm8k_worker_paths_receive_distinct_role_variant_indices(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_worker_variants")
    parse_bundle = kv._collect_parse_bundle(
        GSM8K_0_QUESTION,
        specialist="math",
        galaxy_names=["Math", "Grammar", "Number", "Word"],
        domain_hint="math",
    )

    paths = kv._build_gpu_reasoning_paths(
        task={
            "type": "MATH_TASK",
            "competition": "GSM8K",
            "query": GSM8K_0_QUESTION,
        },
        task_type="MATH_TASK",
        primary_program_id=Knowledgeverse.GPU_MATH_REASONING_PROGRAM_ID,
        query_text=GSM8K_0_QUESTION,
        parse_bundle=parse_bundle,
    )

    assert [int(path.get("role_variant_index", -1)) for path in paths[:9]] == list(range(9))


def test_gsm8k_aggregate_prefers_structural_quality_over_majority_support(tmp_path) -> None:
    class _StubEngine:
        @staticmethod
        def evaluate_batch(expressions, max_parallel=0):
            assert len(expressions) == 4
            return [0.70, 7.0, 0.60, 1.0]

    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_aggregate")
    selection_steps: list[str] = []
    records = [
        {
            "candidate": {
                "gsm8k_preview_answer": "1",
                "gsm8k_strategy_weight": 1.0,
                "gsm8k_structural_score": 0.20,
            },
            "path_score": 0.70,
        },
        {
            "candidate": {
                "gsm8k_preview_answer": "1",
                "gsm8k_strategy_weight": 1.0,
                "gsm8k_structural_score": 0.15,
            },
            "path_score": 0.68,
        },
        {
            "candidate": {
                "gsm8k_preview_answer": "3",
                "gsm8k_strategy_weight": 1.0,
                "gsm8k_structural_score": 0.95,
            },
            "path_score": 0.60,
        },
    ]

    aggregated = kv._aggregate_gsm8k_preview_records(
        engine=_StubEngine(),
        path_best_records=records,
        selection_steps=selection_steps,
    )

    assert aggregated[0]["option_text"] == "3"
    assert float(aggregated[0]["best_structural_score"]) == 0.95


def test_gsm8k_structural_score_uses_role_confidence(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_structural_conf")
    metadata = {
        "required_roles": ["rate", "count", "count"],
        "role_slots": ["count_1", "count_2", "rate"],
        "operation_chain": ["mul", "mul", "add", "add"],
    }
    high_conf = [
        {"role": "count", "value": 3.0, "role_confidence": 1.0},
        {"role": "count", "value": 3.0, "role_confidence": 1.0},
        {"role": "rate", "value": 60.0, "role_confidence": 1.0},
    ]
    low_conf = [
        {"role": "count", "value": 3.0, "role_confidence": 0.2},
        {"role": "count", "value": 3.0, "role_confidence": 0.2},
        {"role": "rate", "value": 60.0, "role_confidence": 0.2},
    ]

    high_score = kv._gsm8k_pattern_structural_score(
        metadata=metadata,
        quantity_candidates=high_conf,
        quantity_count=3,
        clause_operations=["mul"],
        top_operations=["mul", "add"],
        goal_operation="add",
    )
    low_score = kv._gsm8k_pattern_structural_score(
        metadata=metadata,
        quantity_candidates=low_conf,
        quantity_count=3,
        clause_operations=["mul"],
        top_operations=["mul", "add"],
        goal_operation="add",
    )

    assert high_score > low_score


def test_gsm8k_benchmark_navigation_anchor_is_blocked_during_evaluation(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_navigation_honesty")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Math", "Grammar", "Number", "Word"])
    entry = kv._catalog_entry_by_id("benchmark_math_gsm8k_0_direct")

    assert entry is not None
    assert not kv._benchmark_navigation_entry_allowed(
        entry=entry,
        task_type="MATH_TASK",
        task={
            "type": "MATH_TASK",
            "task_id": "gsm8k_0",
            "competition": "GSM8K",
            "query": GSM8K_0_QUESTION,
            "expected_answer": "18",
        },
        query_text=GSM8K_0_QUESTION,
    )


def test_gsm8k_context_tracks_clause_and_goal_operations(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_clause_context")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Math", "Grammar", "Number", "Word"])
    parse_bundle = kv._collect_parse_bundle(
        GSM8K_RELATION_QUESTION,
        specialist="math",
        galaxy_names=["Math", "Grammar", "Number", "Word"],
        domain_hint="math",
    )

    context = kv._gsm8k_word_problem_context(
        target_galaxies=["Math", "Grammar", "Number", "Word"],
        base_embedding=kv._embed_query_gpu(GSM8K_RELATION_QUESTION),
        parse_bundle=parse_bundle,
    )

    assert context["clause_operations"]
    assert context["goal_operation"]
    assert context["clause_values"][:2] == [2.0, 0.5]


def test_gsm8k_decomposition_fallback_can_execute_chain_program(tmp_path) -> None:
    class _StubEngine:
        @staticmethod
        def evaluate(program: str) -> float:
            stack: list[float] = []
            for token in program.split():
                if token == "+":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left + right)
                elif token == "-":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left - right)
                elif token == "*":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left * right)
                elif token == "/":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left / right)
                else:
                    stack.append(float(token))
            return stack[-1]

    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_decomposition")

    result = kv._gsm8k_decomposition_result(
        engine=_StubEngine(),
        best_candidate={
            "gsm8k_context": {
                "number_values": [16.0, 3.0, 4.0, 2.0],
                "operation_chain": ["sub", "sub", "mul"],
                "top_operation": "mul",
            }
        },
    )

    assert result is not None
    answer, trace = result
    assert answer == "18"
    assert any("GSM8K fusion eval:" in step for step in trace)


def test_gsm8k_template_preview_uses_role_bound_pattern(tmp_path) -> None:
    class _StubEngine:
        @staticmethod
        def evaluate(program: str) -> float:
            stack: list[float] = []
            for token in program.split():
                if token == "+":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left + right)
                elif token == "-":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left - right)
                elif token == "*":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left * right)
                elif token == "/":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left / right)
                else:
                    stack.append(float(token))
            return stack[-1]

    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_template_preview")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Grammar"])
    pattern = kv._catalog_entry_by_id("operation_pattern_remainder_scale")
    assert pattern is not None

    preview = kv._gsm8k_decomposition_preview(
        engine=_StubEngine(),
        context={
            "pattern_rows": [pattern],
            "quantity_role_values": {
                "initial": [16.0],
                "part": [3.0, 4.0],
                "rate": [2.0],
            },
            "number_values": [16.0, 3.0, 4.0, 2.0],
        },
        strategy="fusion_chain",
    )

    assert preview is not None
    answer, program, label = preview
    assert answer == "18"
    assert program == "16 3 - 4 - 2 *"
    assert label == "fusion_chain"


def test_gsm8k_goal_adjusted_chain_handles_relation_plus_total(tmp_path) -> None:
    class _StubEngine:
        @staticmethod
        def evaluate(program: str) -> float:
            stack: list[float] = []
            for token in program.split():
                if token == "+":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left + right)
                elif token == "-":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left - right)
                elif token == "*":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left * right)
                elif token == "/":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left / right)
                else:
                    stack.append(float(token))
            return stack[-1]

    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_goal_adjusted")
    preview = kv._gsm8k_decomposition_preview(
        engine=_StubEngine(),
        context={
            "number_values": [2.0, 0.5],
            "forward_number_values": [2.0, 0.5],
            "clause_values": [2.0, 0.5],
            "clause_operations": ["mul"],
            "goal_operation": "add",
        },
        strategy="goal_adjusted_chain",
    )

    assert preview is not None
    answer, _, label = preview
    assert answer == "3"
    assert label == "goal_adjusted_chain"


def test_gsm8k_benchmark_direct_preview_uses_match_answer(tmp_path) -> None:
    class _StubEngine:
        @staticmethod
        def evaluate(program: str) -> float:
            return 0.0

    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_direct_preview")
    result = kv._gsm8k_decomposition_result(
        engine=_StubEngine(),
        best_candidate={
            "gsm8k_preview_answer": "18",
            "gsm8k_preview_program": "math_template_arithmetic_chain_gpu",
            "gsm8k_preview_strategy": "benchmark_direct",
            "gsm8k_context": {"number_values": [16.0, 3.0, 4.0, 2.0]},
        },
    )

    assert result is not None
    answer, trace = result
    assert answer == "18"
    assert any("benchmark_direct" in step for step in trace)


def test_gsm8k_answer_consensus_prefers_supported_preview(tmp_path) -> None:
    class _StubEngine:
        @staticmethod
        def evaluate(program: str) -> float:
            stack: list[float] = []
            for token in program.split():
                if token == "+":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left + right)
                elif token == "-":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left - right)
                elif token == "*":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left * right)
                elif token == "/":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left / right)
                else:
                    stack.append(float(token))
            return stack[-1]

        @classmethod
        def evaluate_batch(cls, programs: list[str], *, max_parallel: int) -> list[float]:
            return [cls.evaluate(program) for program in programs[:max_parallel]]

    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_consensus")
    selection_steps: list[str] = []

    records = kv._aggregate_gsm8k_preview_records(
        engine=_StubEngine(),
        path_best_records=[
            {
                "candidate": {
                    "gsm8k_preview_answer": "18",
                    "gsm8k_strategy_weight": kv._gsm8k_strategy_weight("clause_chain"),
                },
                "path_score": 0.90,
            },
            {
                "candidate": {
                    "gsm8k_preview_answer": "18",
                    "gsm8k_strategy_weight": kv._gsm8k_strategy_weight("goal_adjusted_chain"),
                },
                "path_score": 0.80,
            },
            {
                "candidate": {
                    "gsm8k_preview_answer": "12",
                    "gsm8k_strategy_weight": kv._gsm8k_strategy_weight("alt_mul"),
                },
                "path_score": 1.10,
            },
        ],
        selection_steps=selection_steps,
    )

    assert records[0]["option_text"] == "18"
    assert records[0]["support_count"] == 2
    assert records[0]["weighted_support"] > records[1]["weighted_support"]
    assert any("GSM8K answer consensus: 18" in step for step in selection_steps)
    assert any("weight=" in step for step in selection_steps)


def test_gsm8k_halting_thresholds_follow_grammar_rule(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_thresholds")
    kv.bind_gpu_galaxy_runtime(galaxy_names=["Grammar"])

    assert kv._gsm8k_halting_thresholds() == (0.28, 0.04, 1.0)


def test_gsm8k_consensus_scores_are_propagated_back_to_worker_records(tmp_path) -> None:
    class _StubEngine:
        @staticmethod
        def evaluate(program: str) -> float:
            stack: list[float] = []
            for token in program.split():
                if token == "+":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left + right)
                elif token == "-":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left - right)
                elif token == "*":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left * right)
                elif token == "/":
                    right = stack.pop()
                    left = stack.pop()
                    stack.append(left / right)
                else:
                    stack.append(float(token))
            return stack[-1]

        @classmethod
        def evaluate_batch(cls, programs: list[str], *, max_parallel: int) -> list[float]:
            return [cls.evaluate(program) for program in programs[:max_parallel]]

    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_consensus_propagation")
    selection_steps: list[str] = []
    path_best_records = [
        {
            "candidate": {
                "gsm8k_preview_answer": "18",
                "gsm8k_strategy_weight": kv._gsm8k_strategy_weight("clause_chain"),
            },
            "path_score": 0.90,
        },
        {
            "candidate": {
                "gsm8k_preview_answer": "18",
                "gsm8k_strategy_weight": kv._gsm8k_strategy_weight("goal_adjusted_chain"),
            },
            "path_score": 0.80,
        },
        {
            "candidate": {
                "gsm8k_preview_answer": "12",
                "gsm8k_strategy_weight": kv._gsm8k_strategy_weight("alt_mul"),
            },
            "path_score": 1.10,
        },
    ]

    aggregated = kv._aggregate_gsm8k_preview_records(
        engine=_StubEngine(),
        path_best_records=path_best_records,
        selection_steps=selection_steps,
    )
    aggregate_by_answer = {str(record["option_text"]): record for record in aggregated}
    for record in path_best_records:
        answer_key = kv._gsm8k_preview_candidate_id(record)
        aggregate_record = aggregate_by_answer[answer_key]
        record["path_score"] = float(aggregate_record["path_score"])
        record["support_count"] = int(aggregate_record["support_count"])
        record["weighted_support"] = float(aggregate_record["weighted_support"])

    assert path_best_records[0]["path_score"] == aggregated[0]["path_score"]
    assert path_best_records[0]["support_count"] == 2
    assert path_best_records[0]["weighted_support"] > path_best_records[2]["weighted_support"]


def test_gsm8k_preview_discards_non_finite_results(tmp_path) -> None:
    class _StubEngine:
        @staticmethod
        def evaluate(program: str) -> float:
            del program
            return float("inf")

    kv = Knowledgeverse(storage_root=tmp_path / "kv_gsm8k_non_finite")
    preview = kv._gsm8k_decomposition_preview(
        engine=_StubEngine(),
        context={
            "number_values": [4.0, 0.0],
            "operation_chain": ["div"],
            "top_operation": "div",
        },
        strategy="fusion_chain",
    )

    assert preview is None
