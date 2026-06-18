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
    alpha: float = 0.1
    ema_decay: float = 0.99
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
    # Which JEPA objective to use. "lejepa" (default) is the existing squared-Euclidean
    # align + SIGReg loss; "llm-jepa-loss" switches to the LLM-JEPA paper's cosine-distance
    # prediction loss (arXiv:2509.14252) + SIGReg; "jepa-triplet-loss" extends
    # "llm-jepa-loss" with a hard-negative triplet term against a "clean wrong" code
    # rollout (see core_algos.llm_jepa_triplet_loss). Mutually exclusive — exactly one
    # is used.
    loss_type: str = "lejepa"
    # Number of tied-weight predictor tokens (paper §3.1). k=0 -> Pred(x) = x (identity),
    # matching current behavior. Only used when loss_type in {"llm-jepa-loss", "jepa-triplet-loss"}.
    predictor_k: int = 0
    # Token id used for the appended predictor tokens. Resolved programmatically by
    # ray_trainer.py (which holds the tokenizer) before jepa_init; -1 means unset/unused.
    predictor_token_id: int = -1
    # -- jepa-triplet-loss only --
    # Hinge margin for the triplet term: max(0, triplet_margin - <p^c,e^c> + <p^c,e^w>).
    triplet_margin: float = 0.1
    # Weight of the triplet term relative to L_align inside the (1-lambda) slot of
    # L_total: (1-lambda)*(L_align + triplet_w*L_tri) + lambda*L_SIGReg.
    triplet_w: float = 0.3
    # Dedicated SIGReg lambda for this mode. Deliberately separate from `sigreg_lambda`
    # (default 0.1, used by "lejepa"/"llm-jepa-loss") so picking "jepa-triplet-loss"
    # doesn't silently inherit the other modes' default.
    triplet_sigreg_lambda: float = 0.05

    @classmethod
    def from_config(cls, config: DictConfig | dict | None) -> "JEPARayConfig":
        if not config:
            return cls()
        merged = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.create(config))
        return OmegaConf.to_object(merged)


def jepa_enabled(config: DictConfig | dict | None) -> bool:
    jepa_cfg = config.get("jepa", {}) if config is not None else {}
    return bool(jepa_cfg and jepa_cfg.get("enable", False))
