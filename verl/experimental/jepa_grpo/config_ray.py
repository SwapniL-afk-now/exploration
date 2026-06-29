# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Config dataclass for the Ray-based JEPA-GRPO trainer."""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf


@dataclass
class JEPARayConfig:
    """JEPA hyperparameters for the Ray-based distributed trainer.

    Loss:  L_total = L_DrGRPO(CoT) + alpha * L_LeJEPA(enc_q_cot, enc_a_code)
    The LeJEPA loss aligns CoT prompt embeddings with correct Code view
    embeddings on the unit sphere (align_loss) while regularising the joint
    distribution toward N(0, I_d) via SIGReg (Epps-Pulley test).
    """

    enable: bool = True
    # Split of actor_rollout_ref.rollout.n completions per prompt between the
    # CoT-framed and Code-framed system prompts. Both subsets contribute to
    # the GRPO policy-gradient update (the old behavior generated a SEPARATE,
    # full rollout_n-sized Code batch on top of the CoT one, whose reward was
    # used only for JEPA pairing — half the rollout compute never reached the
    # policy gradient). n_cot + n_code must equal rollout_n; see validate().
    n_cot: int = 4
    n_code: int = 4
    alpha: float = 0.1
    # Linearly ramp the EFFECTIVE alpha used in jepa_update from 0 -> `alpha`
    # over this many JEPA-update steps (0 = disabled, full `alpha` from step
    # 0 — prior behavior). Pre-clip grad_norm is observed to be elevated
    # (~5) at the very start of training before the predictor/encoder
    # geometry settles; gradient clipping already bounds the actual optimizer
    # step, but warming up alpha additionally reduces how much weight that
    # noisy early direction gets, independent of clip_grad.
    alpha_warmup_steps: int = 0
    ema_decay: float = 0.99
    # Gradient-clip max-norm used by jepa_update()'s own optimizer_step() call
    # (worker.py), separate from the actor's PPO optimizer_step() clip_grad
    # (typically 1.0). jepa_update shares the actor's optimizer/parameters but
    # runs as its own backward+step, so without its own (tighter) ceiling, an
    # occasional JEPA grad-norm spike injects a full-magnitude, uncoordinated
    # update onto the same LoRA weights the PPO step is trying to keep stable.
    max_grad_norm: float = 0.5
    embed_micro_batch_size: int = 16
    # Minimum number of valid (cot_correct AND code_correct) pairs to skip step
    min_valid_pairs: int = 2
    # SIGReg hyper-params (shared with core_algos.py defaults)
    n_projections: int = 1024
    t_min: float = -5.0
    t_max: float = 5.0
    epps_pulley_s: float = 1.0
    # Code view system prompt override; if empty uses default math CoT prompt
    code_system_prompt: str = (
        "You are a Python programming expert. "
        "Solve the following math problem by writing a complete, executable Python program "
        "that prints the answer. Do not include any natural language explanation outside comments."
    )
    # JEPA objective. The ray/worker path supports a single value:
    #   "jepa-tcr-dual" — Dual-target, self-consistent Teacher-Correct Representation
    #       alignment. SEPARATE CoT and Code student anchors, each with [PRED]; each
    #       view's pred is pulled toward its OWN precomputed teacher-correct target
    #       (align_cot / align_code; offline teacher solutions encoded by a frozen
    #       student-size reference into the 1536-d student space — no projector), plus
    #       a self-consistency pull of pred_CoT toward the stop-grad Code_S boundary
    #       read, with SIGReg over [pred_cot, pred_code] for anti-collapse. Also folds
    #       a per-view β·ŝ teacher-alignment term into token_level_rewards before the
    #       GRPO advantage (set tcr_reward_beta=0 to disable). Requires teacher_cache_path
    #       AND code_teacher_cache_path. See core_algos.llm_jepa_tcr_dual_loss.
    # (The separate JEPAGRPOTrainer entrypoint in trainer.py uses its own EMA
    # "lejepa" loss and does not read this field.)
    loss_type: str = "jepa-tcr-dual"
    # -- jepa-tcr-dual teacher caches --
    # Path to the offline teacher-target cache produced by
    # examples/jepa_grpo_trainer/precompute_teacher_targets.py: a torch.save dict
    # {dataset_index (int): float16 tensor (n_i, d)} of L2-normalized teacher-correct
    # target embeddings in student space. Empty => tcr mode cannot run.
    teacher_cache_path: str = ""
    # -- jepa-tcr-dual only --
    # Second offline cache (same {index: (n_i,d)} format) of CODE-view teacher targets
    # produced by precompute_teacher_targets.py --view code (a coder model's verified
    # Python solutions encoded by the SAME frozen student-size reference, so both caches
    # share the 1536-d student space). Empty => jepa-tcr-dual cannot run.
    code_teacher_cache_path: str = ""
    # Weight of the self-consistency align term (pred_CoT -> stop-grad Code_S boundary
    # read) inside the (1-lambda) slot, alongside the two teacher pulls. No new lambda.
    self_consist_w: float = 1.0
    # Max teacher targets kept/used per question (the offline pass caps at this; the
    # builder cycles anchors over whatever is cached).
    n_targets_per_q: int = 4
    # Anchor->target matching when a question has multiple cached targets (both
    # resolved at batch-build time so the worker only ever sees one target per
    # anchor):
    #   "cycle"  — anchor j uses target [j % n_u] (default; deterministic)
    #   "random" — anchor j uses a uniformly random cached target
    tcr_match: str = "cycle"
    # Which student rollouts become JEPA anchors (jepa-tcr-loss only). All anchors,
    # correct or wrong, use the identical [x, y_S, [PRED]xk] -> sg(z_T^+) format; the
    # [PRED] token predicts the teacher-correct latent from the student's response
    # context (refinement for correct, latent correction for wrong). Reward-stratified,
    # prompt-averaged so wrong-anchor counts never implicitly weight the loss.
    #   "correct" — only rew>0 rollouts (default; reproduces today's selection)
    #   "all"     — every rollout (correct + wrong)
    #   "wrong"   — only rew<=0 rollouts (analysis ablation)
    jepa_anchor_set: str = "correct"
    # -- jepa-tcr-dual reward shaping (idea #2) --
    # Uses the SAME teacher_cache_path / n_targets_per_q as jepa-tcr-loss, but applies
    # the alignment as an additive advantage term β·ŝ_i (NO differentiable loss / no
    # backward). ŝ_i is the per-rollout score s_i = max_k <p_i, z_k> standardized within
    # its (uid, is_correct) reward stratum, so the term is global-shift invariant and
    # never flips a correct-vs-wrong ordering. See JEPA_TCR_LOSS.md / the plan.
    tcr_reward_beta: float = 0.5          # shaping strength (standardized score scale)
    tcr_reward_sigma_floor: float = 0.1   # min within-stratum std (noise guard)
    # -- auto-disable the JEPA auxiliary signal once teacher-alignment plateaus --
    # When the tracked alignment metric stops improving by `auto_off_min_delta` for
    # `auto_off_patience` consecutive steps (after `auto_off_warmup_steps`), the JEPA
    # signal is LATCHED OFF for the rest of training: the differentiable jepa_update
    # backward AND the reward-shaping advantage term are both skipped (whichever the
    # loss_type uses). GRPO is unaffected. Frees the JEPA forward/shaping cost once the
    # representation objective has converged. Off by default (opt-in).
    auto_off_enable: bool = False
    # Metric key to watch (higher = better alignment). Empty => auto by loss_type:
    # reward-shaping modes use "shaping/s_mean_correct", differentiable modes "jepa/cos_cot".
    auto_off_metric: str = ""
    auto_off_patience: int = 10           # consecutive non-improving steps before turning off
    auto_off_min_delta: float = 0.002     # min metric increase counted as an improvement
    auto_off_warmup_steps: int = 20       # do not evaluate the plateau before this many steps
    # Minimum cosine similarity each arm must reach before its plateau counter starts.
    # An arm below this threshold never accumulates stall counts even after warmup,
    # so the plateau latch cannot fire until alignment is genuinely established.
    # Applies per-arm to cos_cot / cos_code / cos_self (and to the global metric when
    # auto_off_metric is set). Set to 0.0 to disable (original behaviour).
    auto_off_min_cos: float = 0.0
    # Number of tied-weight predictor tokens (paper §3.1). k=0 -> Pred(x) = x
    # (identity), so for a real predictive separation set predictor_k > 0.
    predictor_k: int = 0
    # Token id used for the appended predictor tokens. Resolved programmatically by
    # ray_trainer.py (which holds the tokenizer) before jepa_init; -1 means unset/unused.
    predictor_token_id: int = -1
    # SIGReg lambda for the jepa-tcr-dual loss: L = (1-λ)·(align_cot + align_code +
    # w·L_self) + λ·SIGReg([pred_cot, pred_code]). (Name retained for config back-compat.)
    triplet_sigreg_lambda: float = 0.05

    @classmethod
    def from_config(cls, config: DictConfig | dict | None) -> "JEPARayConfig":
        if not config:
            return cls()
        merged = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.create(config))
        return OmegaConf.to_object(merged)

    def validate(self, rollout_n: int) -> None:
        """Cross-check the cot/code split against actor_rollout_ref.rollout.n.

        Called explicitly by JEPARayPPOTrainer.__init__ (this dataclass has no
        visibility into the sibling `actor_rollout_ref.rollout.n` Hydra node on
        its own). Fails fast at trainer construction, before any Ray workers
        spin up, instead of surfacing as a shape mismatch deep inside fit().
        """
        if not self.enable:
            return
        if self.loss_type != "jepa-tcr-dual":
            raise ValueError(
                f"jepa.loss_type must be 'jepa-tcr-dual' (the only supported ray/worker "
                f"objective); got {self.loss_type!r}"
            )
        if not self.teacher_cache_path:
            raise ValueError(
                "jepa.loss_type='jepa-tcr-dual' requires jepa.teacher_cache_path "
                "(the CoT-view teacher cache from precompute_teacher_targets.py --view cot)"
            )
        if not self.code_teacher_cache_path:
            raise ValueError(
                "jepa.loss_type='jepa-tcr-dual' requires jepa.code_teacher_cache_path "
                "(the Code-view cache from precompute_teacher_targets.py --view code)"
            )
        if self.tcr_match not in ("cycle", "random"):
            raise ValueError(f"jepa.tcr_match must be 'cycle' or 'random'; got {self.tcr_match!r}")
        if self.jepa_anchor_set not in ("correct", "all", "wrong"):
            raise ValueError(
                f"jepa.jepa_anchor_set must be 'correct', 'all' or 'wrong'; got {self.jepa_anchor_set!r}"
            )
        if self.tcr_reward_beta < 0:
            raise ValueError(f"jepa.tcr_reward_beta must be >= 0; got {self.tcr_reward_beta}")
        if self.tcr_reward_sigma_floor <= 0:
            raise ValueError(f"jepa.tcr_reward_sigma_floor must be > 0; got {self.tcr_reward_sigma_floor}")
        if self.n_cot < 0 or self.n_code < 0:
            raise ValueError(f"jepa.n_cot ({self.n_cot}) and jepa.n_code ({self.n_code}) must be >= 0")
        if self.n_code <= 0:
            raise ValueError(
                "jepa.loss_type='jepa-tcr-dual' requires jepa.n_code > 0 "
                "(the Code view needs code-framed rollouts)"
            )
        if self.n_cot + self.n_code != rollout_n:
            raise ValueError(
                f"jepa.n_cot ({self.n_cot}) + jepa.n_code ({self.n_code}) must equal "
                f"actor_rollout_ref.rollout.n ({rollout_n})"
            )


def jepa_enabled(config: DictConfig | dict | None) -> bool:
    jepa_cfg = config.get("jepa", {}) if config is not None else {}
    return bool(jepa_cfg and jepa_cfg.get("enable", False))
