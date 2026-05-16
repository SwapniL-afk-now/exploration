import torch

from verl.experimental.fepo.core_algos import (
    attach_fepo_dr_grpo_group_advantages,
    compute_group_advantages,
    token_failure_sft_loss,
    token_solver_fepo_loss,
)


def test_dr_grpo_sign_flipped_advantages():
    result = compute_group_advantages([1.0, 0.0, 0.0], num_generations=3)
    expected_solver = torch.tensor([2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0])
    assert torch.allclose(result.solver_advantages, expected_solver, atol=1e-6)
    assert torch.allclose(result.failure_advantages, -2.0 * result.solver_advantages, atol=1e-6)
    assert abs(float(result.solver_advantages.mean().item())) < 1e-6


def test_attach_advantages_adds_token_masks():
    items = [
        {"reward": 1.0, "completion_length": 4},
        {"reward": 0.0, "completion_length": 4},
        {"reward": 0.0, "completion_length": 4},
    ]
    attach_fepo_dr_grpo_group_advantages(items)
    assert items[0]["failed_token_mask"].sum().item() == 0
    assert items[1]["failed_token_mask"].sum().item() == 4
    assert torch.allclose(items[1]["failure_token_advantages"], -2.0 * items[1]["token_advantages"])

    all_wrong = [{"reward": 0.0, "completion_length": 3}, {"reward": 0.0, "completion_length": 3}]
    attach_fepo_dr_grpo_group_advantages(all_wrong)
    assert all(item["grpo_advantage"] == 0.0 for item in all_wrong)
    assert all(item["failure_advantage"] == 0.0 for item in all_wrong)


def test_token_solver_fepo_loss_matches_notebook_formula():
    current = torch.zeros(4, dtype=torch.float32, requires_grad=True)
    old = torch.zeros(4, dtype=torch.float32)
    ref = torch.tensor([0.0, 0.25, -0.25, 0.0], dtype=torch.float32)
    failure = torch.tensor([0.0, -0.5, 0.3, 0.0], dtype=torch.float32)
    adv = torch.tensor([0.5, -0.5, 0.25, 0.0], dtype=torch.float32)
    failed = torch.tensor([False, True, True, False])
    mask = torch.tensor([True, True, True, False])

    loss, metrics = token_solver_fepo_loss(
        current,
        old,
        ref,
        failure,
        adv,
        failed,
        clip_eps=0.2,
        reference_beta=0.03,
        escape_alpha=0.01,
        loss_mask=mask,
    )

    valid = mask.bool()
    failed_valid = failed.bool() & valid
    log_ratio = current.detach()[valid] - old[valid]
    expected_pg = -(torch.exp(log_ratio) * adv[valid]).mean()
    failure_minus_current = failure[failed_valid] - current.detach()[failed_valid]
    expected_failure_kl = (torch.exp(failure_minus_current) - failure_minus_current - 1.0).mean()
    ref_minus_current = ref[failed_valid] - current.detach()[failed_valid]
    expected_reference_kl = (torch.exp(ref_minus_current) - ref_minus_current - 1.0).mean()
    expected_loss = expected_pg + 0.03 * expected_reference_kl - 0.01 * expected_failure_kl

    assert torch.allclose(loss.detach(), expected_loss, atol=1e-6)
    assert metrics["failed_token_count"] == 2.0
    assert metrics["loss_token_count"] == 3.0
    loss.backward()
    assert current.grad is not None


def test_solver_loss_edge_groups_are_finite():
    for rewards in ([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]):
        items = [{"reward": reward, "completion_length": 2} for reward in rewards]
        attach_fepo_dr_grpo_group_advantages(items)
        for item in items:
            current = torch.zeros(2, dtype=torch.float32, requires_grad=True)
            zeros = torch.zeros(2, dtype=torch.float32)
            loss, metrics = token_solver_fepo_loss(
                current,
                zeros,
                zeros,
                zeros,
                item["token_advantages"],
                item["failed_token_mask"],
                clip_eps=0.2,
                reference_beta=0.03,
                escape_alpha=0.01,
                loss_mask=item["token_loss_mask"],
            )
            assert torch.isfinite(loss)
            assert metrics["loss_token_count"] == 2.0
            loss.backward()
            assert current.grad is not None


def test_solver_loss_handles_no_parse_empty_completion():
    current = torch.zeros(0, dtype=torch.float32, requires_grad=True)
    empty = torch.zeros(0, dtype=torch.float32)
    failed = torch.zeros(0, dtype=torch.bool)
    loss, metrics = token_solver_fepo_loss(
        current,
        empty,
        empty,
        empty,
        empty,
        failed,
        clip_eps=0.2,
        reference_beta=0.03,
        escape_alpha=0.01,
        loss_mask=failed,
    )
    assert loss.item() == 0.0
    assert metrics["loss_token_count"] == 0.0
    loss.backward()
    assert current.grad is not None


def test_failure_sft_only_updates_failed_responses():
    current = torch.tensor([-1.0, -2.0, -3.0], requires_grad=True)
    mask = torch.tensor([True, False, True])
    inactive_loss, inactive_metrics = token_failure_sft_loss(current, is_failed=False, loss_mask=mask)
    assert inactive_loss.item() == 0.0
    assert inactive_metrics["failure_is_sft_active"] == 0.0

    loss, metrics = token_failure_sft_loss(current, is_failed=True, loss_mask=mask)
    assert torch.allclose(loss, torch.tensor(2.0))
    assert metrics["failure_sft_token_count"] == 2.0
    assert metrics["failure_is_sft_active"] == 1.0
