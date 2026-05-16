# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import hydra
from omegaconf import OmegaConf

from verl.experimental.fepo.trainer import FEPOSingleGPUConfig, FEPOSingleGPUTrainer


@hydra.main(config_path="config", config_name="fepo_trainer", version_base=None)
def main(config):
    resolved = OmegaConf.to_container(config, resolve=True)
    trainer = FEPOSingleGPUTrainer(FEPOSingleGPUConfig(**resolved))
    try:
        trainer.setup()
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
