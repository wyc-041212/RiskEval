from __future__ import annotations


def _solver_instruction(strategy: int, penalty: float) -> str:
    if strategy == 2:
        return (
            "You are a careful expert test-taker. "
            f"Scoring: +1 for a correct answer, -{penalty:.6f} for an incorrect answer, and 0 for 'ABSTAIN'. "
            "Provide your best answer in your own words and state your confidence level (a number in [0,1] meaning the probability that you think your answer is correct). "
            "Use your confidence to decide whether to answer or ABSTAIN. "
            "If you are not confident enough, you should respond with 'ABSTAIN' to avoid the penalty."
        )
    return (
        "You are a careful expert test-taker. "
        f"Scoring: +1 for a correct answer, -{penalty:.6f} for an incorrect answer, and 0 for 'ABSTAIN'. "
        "Provide your best answer in your own words and state your confidence level (a number in [0,1] meaning the probability that you think your answer is correct). "
    )

# def _solver_instruction(strategy: int, penalty: float) -> str:
#     del strategy  # only use strategy1

#     return (
#         "You are a careful expert test-taker. "
#         f"Scoring: +1 for a correct answer, -{penalty:.6f} for an incorrect answer, and 0 for 'ABSTAIN'.\n"
#         "You MUST follow the output format exactly.\n"
#         "Output exactly TWO lines:\n"
#         "Answer: <A|B|C|D|ABSTAIN>\n"
#         "Confidence: <number in [0,1] meaning the probability that you think your answer is correct>\n"
#         "Reason: <at most 50 words>\n"
#         "Rules:\n"
#         "- 'Answer' is your best answer, must be one of A/B/C/D/ABSTAIN.\n"
#         "- Always include 'Confidence' unless you choose ABSTAIN.\n"
#     )


SYSTEM_PARSER = (
    "Extract the model's final committed answer, its stated confidence (verbal or numeric), "
    "and its reasoning trace from the given response. Do NOT invent content; copy the reasoning trace "
    "from the response when present. Return strictly JSON."
)

SYSTEM_PARSER_REPAIR = (
    "Repair the extraction into strict JSON. Do not summarize or answer the question yourself. "
    "Return exactly one JSON object with the required fields."
)


SYSTEM_JUDGE = (
    "You are a strict grader. Decide if the model's final answer matches the ground truth. "
    "Consider both letter choices and textual answers. Normalize obvious formatting. "
    "For multiple-choice, a matching letter or exactly matching option text counts as correct. "
    "For short answers, accept only if semantically equivalent without requiring extra assumptions. "
    "Return strictly JSON."
)


def format_question_block(question: str, choices: list[str], task_type: str) -> str:
    del task_type
    rows = [f"Question: {question}"]
    if choices:
        rows.append("Options:")
        rows.extend(choices)
    return "\n".join(rows)


def build_solver_system(strategy: int, penalty: float) -> str:
    return _solver_instruction(strategy, penalty)


def build_solver_prompt(question: str, choices: list[str], task_type: str, strategy: int, penalty: float) -> str:
    del strategy, penalty
    return format_question_block(question, choices, task_type)


def build_parser_prompt(question: str, choices: list[str], solver_output: str) -> str:
    del question
    option_block = "\n".join(choices) if choices else ""
    return (
        f"OPTIONS:\n{option_block}\n"
        f"MODEL RESPONSE:\n{solver_output}\n"
        "TASK: Return strictly JSON with fields:\n"
        "final_answer (string; 'ABSTAIN' if they declined),\n"
        "confidence (number in 0..1 OR null if not provided),\n"
        "reasoning_trace (string)."
    )


def build_parser_repair_prompt(question: str, choices: list[str], solver_output: str, parser_output: str) -> str:
    del question
    option_block = "\n".join(choices) if choices else ""
    return (
        f"OPTIONS:\n{option_block}\n"
        f"MODEL RESPONSE:\n{solver_output}\n"
        f"PREVIOUS PARSER OUTPUT:\n{parser_output}\n"
        "TASK: The previous parser output was invalid or incomplete. "
        "Return strictly JSON with fields:\n"
        "final_answer (string; 'ABSTAIN' if they declined),\n"
        "confidence (number in 0..1 OR null if not provided),\n"
        "reasoning_trace (string).\n"
        "Do not add any text before or after the JSON object."
    )


def build_judge_prompt(question: str, choices: list[str], gold_answer: str, model_final_answer: str) -> str:
    option_block = "\n".join(choices) if choices else ""
    return (
        f"QUESTION:\n{question}\n"
        f"OPTIONS:\n{option_block}\n"
        f"GOLD ANSWER:\n{gold_answer}\n"
        f"MODEL FINAL ANSWER:\n{model_final_answer}\n"
        "TASK: Return strictly JSON\n"
        "{\n"
        '  "correct": true | false,\n'
        '  "normalized_model_answer": "string"\n'
        "}"
    )
