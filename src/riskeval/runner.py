from __future__ import annotations

import json
import random
from collections import defaultdict

from .client import ChatClient, build_client_for_provider
from .config import Config
from .io import (
    append_jsonl,
    load_jsonl,
    load_jsonl_dicts,
    reset_file,
    write_csv,
    write_json,
    write_jsonl,
)
from .metrics import (
    aggregate,
    expected_utility_if_answer,
    normalized_regret,
    normalized_utility,
    oracle_utility,
    threshold_from_penalty,
    utility_from_action,
)
from .models import ExampleRun
from .parsing import (
    heuristic_parse_solver_output,
    normalize_answer,
    parse_judge_json,
    parse_solver_json,
)
from .prompts import (
    SYSTEM_JUDGE,
    SYSTEM_PARSER,
    SYSTEM_PARSER_REPAIR,
    build_judge_prompt,
    build_parser_prompt,
    build_parser_repair_prompt,
    build_solver_prompt,
    build_solver_system,
)


def _penalty_key(penalty: float) -> str:
    return f"{float(penalty):.6f}"


def _is_mcq_correct(gold_answer: str, solver_answer: str, choices: list[str]) -> bool:
    normalized_choices: dict[str, str] = {}
    for choice in choices:
        stripped = choice.strip()
        if not stripped:
            continue
        letter = normalize_answer(stripped, "mcq")
        parts = stripped.split(maxsplit=1)
        if len(parts) == 2 and len(parts[0]) <= 3:
            text = parts[1].strip()
        else:
            text = stripped
        normalized_choices[letter] = " ".join(text.casefold().split())

    gold = normalize_answer(gold_answer, "mcq")
    if gold not in normalized_choices:
        normalized_gold = " ".join(gold_answer.strip().casefold().split())
        for letter, choice_text in normalized_choices.items():
            if choice_text == normalized_gold:
                gold = letter
                break

    model = normalize_answer(solver_answer, "mcq")
    if not gold or not model:
        return False
    if gold == model:
        return True

    gold_choice_text = normalized_choices.get(gold, "")
    return bool(gold_choice_text) and model == gold_choice_text


def _supports_direct_gold_check(task_type: str) -> bool:
    return task_type in {"mcq", "numeric"}


def _supports_example(ex, cfg: Config) -> bool:
    if ex.image and not cfg.models.supports_vision:
        return False
    return True


def _parse_solver_output(
    *,
    parser_client: ChatClient,
    cfg: Config,
    question: str,
    choices: list[str],
    task_type: str,
    solver_raw: str,
):
    parser_prompt = build_parser_prompt(question, choices, solver_raw)
    parser_raw = parser_client.complete(
        parser_prompt,
        system=SYSTEM_PARSER,
        model=cfg.models.parser_model,
    )
    try:
        return parse_solver_json(parser_raw, task_type)
    except ValueError:
        pass

    repair_prompt = build_parser_repair_prompt(question, choices, solver_raw, parser_raw)
    repair_raw = parser_client.complete(
        repair_prompt,
        system=SYSTEM_PARSER_REPAIR,
        model=cfg.models.parser_model,
    )
    try:
        return parse_solver_json(repair_raw, task_type)
    except ValueError as exc:
        print(
            f"[parser-fallback] parser model returned invalid JSON after repair; using heuristic extraction: {exc}",
            flush=True,
        )
        return heuristic_parse_solver_output(solver_raw, task_type)


def _compute_correctness(
    *,
    judge_client: ChatClient,
    cfg: Config,
    question: str,
    choices: list[str],
    task_type: str,
    gold_answer: str,
    solver_answer: str,
) -> tuple[bool, bool, str]:
    if task_type == "mcq":
        return _is_mcq_correct(gold_answer, solver_answer, choices), False, solver_answer
    if task_type == "numeric":
        normalized = normalize_answer(solver_answer, task_type)
        return normalized == normalize_answer(gold_answer, task_type), False, normalized

    judge_prompt = build_judge_prompt(question, choices, gold_answer, solver_answer)
    judge_raw = judge_client.complete(judge_prompt, system=SYSTEM_JUDGE, model=cfg.models.judge_model)
    try:
        is_correct, normalized_model_answer = parse_judge_json(judge_raw)
    except ValueError as exc:
        raise RuntimeError(f"Judge produced invalid JSON: {exc}") from exc
    return is_correct, True, normalized_model_answer or solver_answer


def _normalize_existing_rows(existing_rows: list[dict], task_type_by_qid: dict[str, str]) -> bool:
    changed = False
    for row in existing_rows:
        qid = str(row["qid"])
        task_type = str(row.get("task_type") or task_type_by_qid.get(qid, "")).strip().lower()
        if row.get("task_type") != task_type:
            row["task_type"] = task_type
            changed = True

        has_gold = bool(row.get("has_gold", row.get("gold") not in {None, ""}))
        if row.get("has_gold") != has_gold:
            row["has_gold"] = has_gold
            changed = True

        judge_applicable = has_gold and not _supports_direct_gold_check(task_type) if task_type else False
        if row.get("judge_applicable") != judge_applicable:
            row["judge_applicable"] = judge_applicable
            changed = True

        gold = row.get("gold")
        if isinstance(gold, str):
            normalized_gold = normalize_answer(gold, task_type) if task_type else gold
            if row.get("gold") != normalized_gold:
                row["gold"] = normalized_gold
                changed = True

        solver_answer = row.get("solver_answer")
        if isinstance(solver_answer, str):
            decision = str(row.get("model_decision", row.get("judge_decision", ""))).upper()
            normalized_solver = (
                normalize_answer(solver_answer, task_type)
                if task_type and decision == "ANSWER" and solver_answer
                else ""
            )
            if row.get("solver_answer") != normalized_solver:
                row["solver_answer"] = normalized_solver
                changed = True

        if has_gold and task_type in {"mcq", "numeric"}:
            decision = str(row.get("model_decision", row.get("judge_decision", ""))).upper()
            normalized_gold = str(row.get("gold") or "")
            normalized_solver = str(row.get("solver_answer") or "")
            recomputed_correct = normalized_solver == normalized_gold if decision == "ANSWER" else False
            if row.get("solver_correct") != recomputed_correct:
                row["solver_correct"] = recomputed_correct
                changed = True

            penalty = float(row["penalty"])
            recomputed_utility = utility_from_action(recomputed_correct, decision, penalty)
            if row.get("utility") != recomputed_utility:
                row["utility"] = recomputed_utility
                changed = True

            recomputed_normalized_utility = normalized_utility(recomputed_utility, penalty)
            if row.get("normalized_utility") != recomputed_normalized_utility:
                row["normalized_utility"] = recomputed_normalized_utility
                changed = True

        if "used_judge" not in row:
            decision = str(row.get("model_decision", row.get("judge_decision", ""))).upper()
            row["used_judge"] = judge_applicable and decision == "ANSWER"
            changed = True
    return changed


def _write_summary(
    *,
    summary_path,
    rows: list[dict],
    penalties: list[float],
    n_questions: int,
    skipped_multimodal: int,
    questions_completed: int,
    judge_calls_completed: int,
    judge_calls_total: int,
) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[_penalty_key(float(row["penalty"]))].append(row)

    per_penalty = {
        _penalty_key(penalty): aggregate(grouped.get(_penalty_key(penalty), []))
        for penalty in penalties
        if grouped.get(_penalty_key(penalty))
    }

    write_json(
        summary_path,
        {
            "n_total": len(rows),
            "n_questions": n_questions,
            "n_skipped_multimodal": skipped_multimodal,
            "questions_completed": questions_completed,
            "judge_calls_completed": judge_calls_completed,
            "judge_calls_total": judge_calls_total,
            "penalties": penalties,
            "metrics_by_penalty": per_penalty,
        },
    )


def _append_trace(
    *,
    trace_path,
    enabled: bool,
    stage: str,
    provider: str,
    model: str,
    qid: str,
    penalty: float,
    system: str | None,
    prompt: str,
    response: str,
) -> None:
    if not enabled:
        return
    payload = {
        "stage": stage,
        "provider": provider,
        "model": model,
        "qid": qid,
        "penalty": penalty,
        "system": system or "",
        "prompt": prompt,
        "response": response,
    }
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run(cfg: Config) -> dict:
    solver_client = build_client_for_provider(cfg, cfg.solver_provider)
    parser_client = build_client_for_provider(cfg, cfg.parser_provider)
    judge_client = build_client_for_provider(cfg, cfg.judge_provider)

    data = load_jsonl(cfg.run.data_path)
    random.Random(cfg.run.random_seed).shuffle(data)
    if cfg.run.max_examples is not None:
        data = data[: cfg.run.max_examples]
    skipped_multimodal = sum(1 for ex in data if ex.image and not cfg.models.supports_vision)
    data = [ex for ex in data if _supports_example(ex, cfg)]
    task_type_by_qid = {str(ex.qid): ex.task_type for ex in data}

    penalties = [float(p) for p in cfg.sweep.penalties]
    penalty_keys = {_penalty_key(p) for p in penalties}

    out_dir = cfg.run.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "example_runs.jsonl"
    csv_path = out_dir / "example_runs.csv"
    summary_path = out_dir / "summary.json"
    trace_path = out_dir / "llm_traces.jsonl"

    existing_rows = load_jsonl_dicts(jsonl_path)
    if existing_rows:
        print(f"[resume] loaded {len(existing_rows)} existing rows from {jsonl_path}", flush=True)
        if _normalize_existing_rows(existing_rows, task_type_by_qid):
            write_jsonl(jsonl_path, existing_rows)
    else:
        reset_file(jsonl_path)
    if cfg.run.save_llm_traces:
        reset_file(trace_path)

    per_example_rows: list[dict] = list(existing_rows)
    total_examples = len(data)
    judge_calls_completed = sum(1 for row in existing_rows if bool(row.get("used_judge", False)))
    judge_calls_total = sum(
        1
        for ex in data
        for _penalty in penalties
        if ex.has_gold and not _supports_direct_gold_check(ex.task_type)
    )

    if skipped_multimodal:
        print(
            f"[skip] skipped {skipped_multimodal} multimodal example(s) because supports_vision=false",
            flush=True,
        )

    completed_penalties_by_qid: dict[str, set[str]] = defaultdict(set)
    for row in existing_rows:
        completed_penalties_by_qid[str(row["qid"])].add(_penalty_key(float(row["penalty"])))

    for ex_idx, ex in enumerate(data, start=1):
        qid = str(ex.qid)
        done_penalties = completed_penalties_by_qid.get(qid, set())
        if done_penalties >= penalty_keys:
            print(f"[{ex_idx}/{total_examples}] qid={qid} already complete, skipping", flush=True)
            continue

        print(f"[{ex_idx}/{total_examples}] qid={qid} task={ex.task_type} start", flush=True)

        for pen_idx, penalty in enumerate(penalties, start=1):
            penalty_key = _penalty_key(penalty)
            if penalty_key in done_penalties:
                print(
                    f"[{ex_idx}/{total_examples}] penalty {pen_idx}/{len(penalties)} lambda={penalty_key} already complete, skipping",
                    flush=True,
                )
                continue

            solver_prompt = build_solver_prompt(
                ex.question,
                ex.choices,
                ex.task_type,
                cfg.run.prompt_strategy,
                penalty,
            )
            solver_system = build_solver_system(cfg.run.prompt_strategy, penalty)
            solver_raw = solver_client.complete(
                solver_prompt,
                system=solver_system,
                model=cfg.models.solver_model,
                image_url=ex.image,
            )
            _append_trace(
                trace_path=trace_path,
                enabled=cfg.run.save_llm_traces,
                stage="solver",
                provider=cfg.solver_provider,
                model=cfg.models.solver_model,
                qid=qid,
                penalty=penalty,
                system=solver_system,
                prompt=solver_prompt,
                response=solver_raw,
            )

            parser_prompt = build_parser_prompt(ex.question, ex.choices, solver_raw)
            parser_raw = parser_client.complete(
                parser_prompt,
                system=SYSTEM_PARSER,
                model=cfg.models.parser_model,
            )
            _append_trace(
                trace_path=trace_path,
                enabled=cfg.run.save_llm_traces,
                stage="parser",
                provider=cfg.parser_provider,
                model=cfg.models.parser_model,
                qid=qid,
                penalty=penalty,
                system=SYSTEM_PARSER,
                prompt=parser_prompt,
                response=parser_raw,
            )
            try:
                parsed = parse_solver_json(parser_raw, ex.task_type)
            except ValueError:
                repair_prompt = build_parser_repair_prompt(
                    ex.question,
                    ex.choices,
                    solver_raw,
                    parser_raw,
                )
                repair_raw = parser_client.complete(
                    repair_prompt,
                    system=SYSTEM_PARSER_REPAIR,
                    model=cfg.models.parser_model,
                )
                _append_trace(
                    trace_path=trace_path,
                    enabled=cfg.run.save_llm_traces,
                    stage="parser_repair",
                    provider=cfg.parser_provider,
                    model=cfg.models.parser_model,
                    qid=qid,
                    penalty=penalty,
                    system=SYSTEM_PARSER_REPAIR,
                    prompt=repair_prompt,
                    response=repair_raw,
                )
                try:
                    parsed = parse_solver_json(repair_raw, ex.task_type)
                except ValueError as exc:
                    print(
                        f"[parser-fallback] parser model returned invalid JSON after repair; using heuristic extraction: {exc}",
                        flush=True,
                    )
                    parsed = heuristic_parse_solver_output(solver_raw, ex.task_type)

            decision = parsed.decision
            solver_answer = parsed.final_answer if decision == "ANSWER" else ""
            judge_applicable = ex.has_gold and not _supports_direct_gold_check(ex.task_type)
            used_judge = False

            if decision == "ANSWER" and ex.has_gold and ex.answer is not None:
                if ex.task_type in {"mcq", "numeric"}:
                    is_correct, used_judge, normalized_solver_answer = _compute_correctness(
                        judge_client=judge_client,
                        cfg=cfg,
                        question=ex.question,
                        choices=ex.choices,
                        task_type=ex.task_type,
                        gold_answer=ex.answer,
                        solver_answer=solver_answer,
                    )
                else:
                    judge_prompt = build_judge_prompt(ex.question, ex.choices, ex.answer, solver_answer)
                    judge_raw = judge_client.complete(
                        judge_prompt,
                        system=SYSTEM_JUDGE,
                        model=cfg.models.judge_model,
                    )
                    _append_trace(
                        trace_path=trace_path,
                        enabled=cfg.run.save_llm_traces,
                        stage="judge",
                        provider=cfg.judge_provider,
                        model=cfg.models.judge_model,
                        qid=qid,
                        penalty=penalty,
                        system=SYSTEM_JUDGE,
                        prompt=judge_prompt,
                        response=judge_raw,
                    )
                    try:
                        is_correct, normalized_solver_answer = parse_judge_json(judge_raw)
                    except ValueError as exc:
                        raise RuntimeError(f"Judge produced invalid JSON: {exc}") from exc
                    used_judge = True
                solver_answer = normalized_solver_answer
                if used_judge:
                    judge_calls_completed += 1
            elif decision == "ANSWER":
                is_correct = None
            else:
                is_correct = False if ex.has_gold else None

            t_star = threshold_from_penalty(penalty)
            if parsed.confidence_prob is None:
                expected_answer_u = None
                oracle_u = None
                consistent = None
                regret = None
                normalized_regret_value = None
            else:
                expected_answer_u = expected_utility_if_answer(parsed.confidence_prob, penalty)
                oracle_u = oracle_utility(parsed.confidence_prob, penalty)
                consistent = (
                    decision == "ANSWER" and parsed.confidence_prob >= t_star
                ) or (
                    decision == "ABSTAIN" and parsed.confidence_prob < t_star
                )
                agent_expected = expected_answer_u if decision == "ANSWER" else 0.0
                regret = oracle_u - agent_expected
                normalized_regret_value = normalized_regret(regret, penalty)

            util = utility_from_action(is_correct, decision, penalty) if ex.has_gold and is_correct is not None else None

            row = ExampleRun(
                qid=qid,
                task_type=ex.task_type,
                penalty=penalty,
                modality=ex.modality,
                has_gold=ex.has_gold,
                gold=normalize_answer(ex.answer, ex.task_type) if ex.answer is not None else None,
                solver_answer=normalize_answer(solver_answer, ex.task_type) if solver_answer else "",
                solver_correct=is_correct,
                confidence_text=parsed.confidence_text,
                confidence_prob=parsed.confidence_prob,
                model_decision=decision,
                judge_decision=decision,
                judge_applicable=judge_applicable,
                used_judge=used_judge,
                utility=util,
                expected_utility_if_answer=expected_answer_u,
                oracle_utility=oracle_u,
                policy_consistent=consistent,
                regret=regret,
                normalized_regret=normalized_regret_value,
                normalized_utility=normalized_utility(util, penalty) if util is not None else None,
            )
            row_dict = row.to_dict()
            per_example_rows.append(row_dict)
            append_jsonl(jsonl_path, row_dict)
            completed_penalties_by_qid[qid].add(penalty_key)
            done_penalties = completed_penalties_by_qid[qid]

            p_display = "null" if parsed.confidence_prob is None else f"{parsed.confidence_prob:.3f}"
            correct_display = "null" if is_correct is None else str(int(is_correct))
            print(
                f"[{ex_idx}/{total_examples}] penalty {pen_idx}/{len(penalties)} lambda={penalty_key} decision={decision} p={p_display} correct={correct_display}",
                flush=True,
            )

        _write_summary(
            summary_path=summary_path,
            rows=per_example_rows,
            penalties=penalties,
            n_questions=total_examples,
            skipped_multimodal=skipped_multimodal,
            questions_completed=ex_idx,
            judge_calls_completed=judge_calls_completed,
            judge_calls_total=judge_calls_total,
        )

    _write_summary(
        summary_path=summary_path,
        rows=per_example_rows,
        penalties=penalties,
        n_questions=len(data),
        skipped_multimodal=skipped_multimodal,
        questions_completed=len(data),
        judge_calls_completed=judge_calls_completed,
        judge_calls_total=judge_calls_total,
    )
    write_csv(csv_path, per_example_rows)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in per_example_rows:
        grouped[_penalty_key(float(row["penalty"]))].append(row)

    return {
        "n_total": len(per_example_rows),
        "n_questions": len(data),
        "n_skipped_multimodal": skipped_multimodal,
        "questions_completed": len(data),
        "judge_calls_completed": judge_calls_completed,
        "judge_calls_total": judge_calls_total,
        "penalties": penalties,
        "metrics_by_penalty": {
            _penalty_key(penalty): aggregate(grouped.get(_penalty_key(penalty), []))
            for penalty in penalties
            if grouped.get(_penalty_key(penalty))
        },
    }
