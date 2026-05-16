from verl.experimental.fepo.data import extract_dapo_problem
from verl.experimental.fepo.math_parser import (
    compute_math_reward,
    normalize_answer_text,
    normalize_ground_truth_answer,
    normalize_model_answer,
)


def test_answer_normalization_and_extraction():
    assert normalize_answer_text("1,234") == "1234"
    assert normalize_answer_text(r"\frac{1}{2}") == "0.5"
    assert normalize_ground_truth_answer("Natalia did it. #### 72", "legacy_hash") == "72"
    assert normalize_ground_truth_answer("34", "dapo-math-17k") == "34"
    assert normalize_ground_truth_answer("-3", "dapo-math-17k") == "-3"
    assert normalize_ground_truth_answer(["$f(x)=\\frac{1}{x}$"], "olympiadbench") == "f(x)=\\frac{1}{x}"
    assert normalize_model_answer(r"We get \boxed{204}.") == "204"
    assert normalize_model_answer("Reasoning... #### 18") == "18"
    assert normalize_model_answer("The answer is 27.") == "27"


def test_reward_correctness_and_parse_flags():
    correct = compute_math_reward(r"We get \boxed{72}.", "72", dataset_kind="math500")
    assert correct.reward == 1.0
    assert correct.anti_reward == -1.0
    assert correct.is_correct is True
    assert correct.has_parseable_answer is True

    wrong = compute_math_reward(r"We get \boxed{71}.", "72", dataset_kind="math500")
    assert wrong.reward == 0.0
    assert wrong.anti_reward == 1.0
    assert wrong.is_correct is False

    unparseable = compute_math_reward("   ", "72", dataset_kind="math500")
    assert unparseable.reward == 0.0
    assert unparseable.has_parseable_answer is False


def test_dapo_prompt_extraction():
    sample_prompt = [
        {
            "role": "user",
            "content": (
                "Solve the following math problem step by step. The last line of your response "
                "should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\n"
                "In triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$. Find $AB+AC$.\n\n"
                'Remember to put your answer on its own line after "Answer:".'
            ),
        }
    ]
    problem = extract_dapo_problem(sample_prompt)
    assert problem.startswith("In triangle $ABC$")
    assert "Remember to put your answer" not in problem
