from verl.experimental.fepo.data import adapt_row_to_verl, convert_split_to_verl_rows, make_messages
from verl.experimental.fepo.metrics import evaluate_responses_by_prompt, summarize_training_records


def test_dapo_schema_conversion():
    row = {
        "prompt": [
            {
                "role": "user",
                "content": (
                    "Solve the following math problem step by step.\n\n"
                    "What is 6 times 7?\n\n"
                    'Remember to put your answer on its own line after "Answer:".'
                ),
            }
        ],
        "label": "42",
    }
    converted = adapt_row_to_verl(row, "zhuzilin/dapo-math-17k", "train", 0)
    assert converted["data_source"] == "dapo-math-17k"
    assert converted["reward_model"]["ground_truth"] == "42"
    assert converted["prompt"][0]["role"] == "system"
    assert "6 times 7" in converted["extra_info"]["problem"]


def test_skywork_verl_schema_is_preserved():
    row = {
        "data_source": "skywork",
        "prompt": make_messages("Compute 20+22."),
        "reward_model": {"style": "rule", "ground_truth": "42"},
    }
    converted = adapt_row_to_verl(row, "sungyub/skywork-or1-math-verl", "train", 3)
    assert converted["data_source"] == "skywork-or1-math-verl"
    assert converted["prompt"] == row["prompt"]
    assert converted["reward_model"]["ground_truth"] == "42"


def test_generic_math_ai_schema_conversion():
    rows = [{"problem": "What is $1+1$?", "answer": "2"}]
    converted = convert_split_to_verl_rows(rows, "math-ai/math500", "test")
    assert converted[0]["data_source"] == "math500"
    assert converted[0]["reward_model"]["ground_truth"] == "2"


def test_amc_question_answer_schema_conversion():
    row = {"question": "How many miles?", "answer": "27", "id": 0}
    converted = adapt_row_to_verl(row, "math-ai/amc23", "test", 0)
    assert converted["data_source"] == "amc23"
    assert converted["reward_model"]["ground_truth"] == "27"
    assert "How many miles" in converted["extra_info"]["problem"]


def test_aime24_solution_schema_conversion():
    row = {"problem": "Find the value.", "solution": r"\boxed{204}", "id": 60}
    converted = adapt_row_to_verl(row, "math-ai/aime24", "test", 0)
    assert converted["data_source"] == "aime24"
    assert converted["reward_model"]["ground_truth"] == "204"


def test_validation_metrics():
    examples = [
        {"ground_truth_normalized": "42"},
        {"ground_truth_normalized": "7"},
    ]
    responses_by_prompt = [
        [r"\boxed{41}", r"\boxed{42}"],
        [r"\boxed{7}", r"\boxed{8}"],
    ]
    metrics = evaluate_responses_by_prompt(examples, responses_by_prompt, "math500", k=2)
    assert metrics["pass_at_1"] == 0.5
    assert metrics["pass_at_k"] == 1.0
    assert metrics["avg_at_k"] == 0.5
    assert metrics["parse_rate"] == 1.0


def test_training_summary_uses_prompt_groups_and_parse_rate():
    records = [
        {"prompt_uid": "p0", "reward": 0.0, "has_parseable_answer": 1.0, "response_length": 12},
        {"prompt_uid": "p0", "reward": 1.0, "has_parseable_answer": 1.0, "response_length": 8},
        {"prompt_uid": "p1", "reward": 0.0, "has_parseable_answer": 0.0, "response_length": 5},
        {"prompt_uid": "p1", "reward": 0.0, "has_parseable_answer": 1.0, "response_length": 7},
    ]
    summary = summarize_training_records(records)
    assert summary["train/response_acc"] == 0.25
    assert summary["train/prompt_pass_at_n"] == 0.5
    assert summary["train/parse_rate"] == 0.75
    assert summary["train/prompt_count"] == 2
