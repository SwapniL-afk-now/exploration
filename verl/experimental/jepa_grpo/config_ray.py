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
    # LeJEPA / SIGReg hyper-params (shared with core_algos.py defaults)
    sigreg_lambda: float = 0.1
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
    # JEPA objective. The ray/worker path supports two values:
    #   "jepa-separation-loss" — a predictor-space triplet: p^c = Pred(Enc(correct
    #       CoT)) is pulled toward the stop-gradiented correct-code target e^c
    #       (align) and pushed off the stop-gradiented clean-wrong target e^w via a
    #       1:1 margin hinge (separation_margin), + SIGReg over [p^c, e^c, e^w]
    #       (see core_algos.llm_jepa_separation_loss).
    #   "jepa-clreg-loss" — the v3 CLReg objective (jepa_separation_loss.md): the
    #       per-anchor margin hinge is replaced by a per-group FULL cross product of
    #       every correct anchor p_i against EVERY wrong joint embedding e^w_k in the
    #       same GRPO group, scored with a DPO log-sigmoid at temperature
    #       separation_tau (or InfoNCE when separation_mode="info"). Same stop-grad
    #       rule and code-only SIGReg pool (see core_algos.llm_jepa_clreg_loss).
    # (The separate JEPAGRPOTrainer entrypoint in trainer.py uses its own EMA
    # "lejepa" loss and does not read this field.)
    loss_type: str = "jepa-separation-loss"
    # Number of tied-weight predictor tokens (paper §3.1). k=0 -> Pred(x) = x
    # (identity), so for a real predictive separation set predictor_k > 0.
    predictor_k: int = 0
    # Token id used for the appended predictor tokens. Resolved programmatically by
    # ray_trainer.py (which holds the tokenizer) before jepa_init; -1 means unset/unused.
    predictor_token_id: int = -1
    # SIGReg lambda for the separation loss. Kept separate from `sigreg_lambda` so it
    # has its own default. (Name retained for config back-compat.)
    triplet_sigreg_lambda: float = 0.05
    # -- jepa-separation-loss hinge --
    # Cosine-distance margin: L_sep = (1/T) Σ relu(separation_margin - (1 - <p^c,e^w>)).
    # Keep small so it does not fight the prompt structure.
    separation_margin: float = 0.1
    # Weight of L_sep inside the (1-lambda) slot. Default 1.0 = UNWEIGHTED:
    # the negative-side separation term is not down-weighted.
    # L = (1-lambda)*(L_align + separation_w*L_sep) + lambda*L_SIGReg.
    separation_w: float = 1.0
    # -- jepa-clreg-loss (v3) only --
    # Temperature for the CLReg contrastive term (replaces the hinge's
    # separation_margin, which is unused in clreg mode). Doc default 0.5; sweep
    # {0.1, 0.3, 0.5, 0.7, 0.9}.
    separation_tau: float = 0.5
    # CLReg negative form: "dpo" (averaged-pairwise log-sigmoid, the doc default)
    # or "info" (InfoNCE alternative). Ignored unless loss_type=jepa-clreg-loss.
    separation_mode: str = "dpo"

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
        if self.enable and self.loss_type not in ("jepa-separation-loss", "jepa-clreg-loss"):
            raise ValueError(
                f"jepa.loss_type must be 'jepa-separation-loss' or 'jepa-clreg-loss' "
                f"(the supported ray/worker objectives); got {self.loss_type!r}"
            )
        if self.enable and self.loss_type == "jepa-clreg-loss" and self.separation_mode not in ("dpo", "info"):
            raise ValueError(
                f"jepa.separation_mode must be 'dpo' or 'info' for jepa-clreg-loss; "
                f"got {self.separation_mode!r}"
            )
        if self.n_cot < 0 or self.n_code < 0:
            raise ValueError(f"jepa.n_cot ({self.n_cot}) and jepa.n_code ({self.n_code}) must be >= 0")
        if self.n_cot + self.n_code != rollout_n:
            raise ValueError(
                f"jepa.n_cot ({self.n_cot}) + jepa.n_code ({self.n_code}) must equal "
                f"actor_rollout_ref.rollout.n ({rollout_n})"
            )
        if self.enable and self.n_code == 0:
            raise ValueError(
                "jepa.enable=True requires jepa.n_code > 0 (no code-framed rollouts to build JEPA pairs from)"
            )


def jepa_enabled(config: DictConfig | dict | None) -> bool:
    jepa_cfg = config.get("jepa", {}) if config is not None else {}
    return bool(jepa_cfg and jepa_cfg.get("enable", False))
