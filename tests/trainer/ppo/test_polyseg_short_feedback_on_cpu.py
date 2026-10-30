"""Short rollouts must retain their actual macro axis in actor microbatches."""

import pytest
import torch

from verl.trainer.ppo.polyseg import compute_segment_policy_loss


def _loss(log_probs, ids, feedback, response_mask=None, configured_m=16):
    return compute_segment_policy_loss(
        old_log_prob=torch.zeros_like(log_probs),
        log_prob=log_probs,
        token_advantages=torch.ones_like(log_probs),
        response_mask=torch.ones_like(ids, dtype=torch.bool) if response_mask is None else response_mask,
        segment_ids_tensor=ids,
        segment_feedback_mask=feedback,
        num_segments=configured_m,
    )[0]


@pytest.mark.parametrize("actual_m", [1, 2, 8, 13])
def test_short_rollout_uses_actual_feedback_width(actual_m):
    # One action token per segment occurs when the response is shorter than m.
    current = torch.full((1, actual_m), 0.05, requires_grad=True)
    ids = torch.arange(actual_m).unsqueeze(0)
    loss = _loss(current, ids, torch.ones(1, actual_m, dtype=torch.bool))
    expected = -current.exp().mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        torch.autograd.grad(loss, current)[0],
        -current.detach().exp() / actual_m,
    )


def test_native_microbatch_and_padded_batch_feedback_have_identical_gradient():
    # The full batch may have 16 segments elsewhere; this microbatch has 3.
    ids = torch.tensor([[0, 0, 1, 2, 2, -1]])
    action = ids >= 0
    current = torch.tensor([[0.03, 0.02, 0.04, -0.02, -0.01, 10.0]], requires_grad=True)
    native = torch.tensor([[True, False, True]])
    padded = torch.nn.functional.pad(native, (0, 13), value=False)
    loss_native = _loss(current, ids, native, action)
    loss_padded = _loss(current, ids, padded, action)
    # Unqueried segment 1 and padding must not affect either gradient.
    expected = -(current[0, :2].sum().exp() + current[0, 3:5].sum().exp()) / 2
    torch.testing.assert_close(loss_native, expected)
    torch.testing.assert_close(loss_padded, expected)
    grad_expected = torch.autograd.grad(expected, current, retain_graph=True)[0]
    torch.testing.assert_close(torch.autograd.grad(loss_native, current)[0], grad_expected)
    torch.testing.assert_close(torch.autograd.grad(loss_padded, current)[0], grad_expected)
    assert grad_expected[0, 2] == grad_expected[0, 5] == 0


def test_feedback_must_cover_every_observed_segment_id():
    with pytest.raises(ValueError, match="segment_feedback_mask"):
        _loss(torch.zeros(1, 3), torch.tensor([[0, 1, 2]]), torch.ones(1, 2, dtype=torch.bool))


def test_feedback_batch_axis_must_match_actor_batch():
    with pytest.raises(ValueError, match="segment_feedback_mask"):
        _loss(torch.zeros(2, 3), torch.tensor([[0, 1, 2], [0, 1, 2]]), torch.ones(1, 3, dtype=torch.bool))


def test_no_feedback_keeps_a_differentiable_zero_loss():
    current = torch.zeros(1, 3, requires_grad=True)
    loss = _loss(current, torch.tensor([[0, 1, 2]]), torch.zeros(1, 3, dtype=torch.bool))
    assert loss.item() == 0
    torch.testing.assert_close(torch.autograd.grad(loss, current)[0], torch.zeros_like(current))
