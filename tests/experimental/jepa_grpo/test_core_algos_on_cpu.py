"""CPU unit tests for verl.experimental.jepa_grpo.core_algos.

Focus: SIGReg must produce a real anti-collapse gradient when fed the
L2-normalized (unit-sphere) embeddings the loss functions actually pass it.

Regression context: SIGReg was being computed on unit-sphere embeddings while
comparing their random 1-D projections against a *unit-variance* Gaussian. A
random projection of a unit vector in d dims has variance ~1/d, so the empirical
characteristic function stayed pinned at ~1 across the whole frequency grid for
every arrangement on the sphere — the statistic was a near-constant ~0.21 with
no usable gradient, so it could not oppose representation collapse (observed in
W&B run ta5p11ro: sigreg_loss frozen to 4 d.p., pos/neg triplet scores
converging in lockstep, hard_neg_ew_variance ~2e-5). The fix standardizes inside
sigreg_loss to zero-mean / unit per-dim variance first.
"""

import torch
import torch.nn.functional as F

from verl.experimental.jepa_grpo.core_algos import sigreg_loss


def _isotropic_sphere(n, d, seed):
    """n L2-normalized embeddings drawn from an isotropic Gaussian (well spread)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return F.normalize(x, dim=-1)


def _collapsed_cone(n, d, seed, jitter=1e-2):
    """n L2-normalized embeddings packed into a narrow cone around one direction."""
    g = torch.Generator().manual_seed(seed)
    center = F.normalize(torch.randn(d, generator=g), dim=-1)
    x = center.unsqueeze(0) + jitter * torch.randn(n, d, generator=g)
    return F.normalize(x, dim=-1)


def test_sigreg_penalizes_collapse_more_than_isotropy():
    """A collapsed cone must score strictly higher than a well-spread isotropic set."""
    d = 256
    iso = _isotropic_sphere(512, d, seed=0)
    cone = _collapsed_cone(512, d, seed=0)

    # Same random projection directions for a fair comparison (seed torch global RNG).
    torch.manual_seed(1234)
    loss_iso = sigreg_loss(iso, M=1024)
    torch.manual_seed(1234)
    loss_cone = sigreg_loss(cone, M=1024)

    assert loss_cone > loss_iso, (loss_cone.item(), loss_iso.item())
    # The gap must be meaningful, not float noise — pre-fix these were ~equal.
    assert (loss_cone - loss_iso) > 0.05, (loss_cone.item(), loss_iso.item())


def test_sigreg_has_nonvanishing_gradient_on_collapse():
    """The collapsed batch must receive a non-trivial gradient (it was ~0 pre-fix)."""
    d = 256
    cone = _collapsed_cone(512, d, seed=7).requires_grad_(True)
    torch.manual_seed(99)
    loss = sigreg_loss(cone, M=1024)
    loss.backward()

    assert cone.grad is not None
    grad_norm = cone.grad.norm().item()
    # On unit-sphere input pre-fix this was numerically ~0; require clearly nonzero.
    assert grad_norm > 1e-3, grad_norm


def test_sigreg_gradient_responds_to_collapse_severity():
    """Tighter collapse (smaller jitter) should not produce a *smaller* gradient.

    Sanity check that the statistic tracks collapse severity rather than being a
    flat constant: a tighter cone is more degenerate and must be penalized at
    least as hard as a looser one.
    """
    d = 256
    tight = _collapsed_cone(512, d, seed=3, jitter=1e-3)
    loose = _collapsed_cone(512, d, seed=3, jitter=1e-1)

    torch.manual_seed(2024)
    loss_tight = sigreg_loss(tight, M=1024)
    torch.manual_seed(2024)
    loss_loose = sigreg_loss(loose, M=1024)

    assert loss_tight >= loss_loose, (loss_tight.item(), loss_loose.item())


def test_sigreg_is_finite_on_total_collapse():
    """Exact collapse (all points identical, per-dim std=0) must stay finite via eps."""
    d = 128
    x = F.normalize(torch.ones(64, d), dim=-1)  # every row identical
    torch.manual_seed(5)
    loss = sigreg_loss(x, M=512)
    assert torch.isfinite(loss), loss.item()
    assert loss.item() > 0.0
