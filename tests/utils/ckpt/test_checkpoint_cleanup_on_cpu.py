# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import os
import shutil
import tempfile

import pytest


class TestCheckpointCleanupLogic:
    """Tests for checkpoint cleanup methods in BaseCheckpointManager."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @pytest.fixture
    def manager(self, monkeypatch):
        """Create a minimal BaseCheckpointManager for testing."""
        import torch.distributed

        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)

        from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager

        class MockModel:
            pass

        class MockOptimizer:
            pass

        return BaseCheckpointManager(
            model=MockModel(),
            optimizer=MockOptimizer(),
            lr_scheduler=None,
            processing_class=None,
            checkpoint_config=None,
        )

    def _create_checkpoint_dir(self, step: int) -> str:
        """Create a mock checkpoint directory."""
        path = os.path.join(self.test_dir, f"global_step_{step}")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "checkpoint.txt"), "w") as f:
            f.write(f"step={step}")
        return path

    def test_max_ckpt_1_preserves_existing_before_save(self, manager):
        """
        Regression test: max_ckpt_to_keep=1 must NOT delete existing checkpoint before save.
        """
        ckpt_100 = self._create_checkpoint_dir(100)
        manager.previous_saved_paths = {"default": [ckpt_100]}

        manager.ensure_checkpoint_capacity(max_ckpt_to_keep=1)

        assert os.path.exists(ckpt_100), "Bug: checkpoint deleted before save!"
        assert manager.previous_saved_paths["default"] == [ckpt_100]

    def test_max_ckpt_1_deletes_old_after_save(self, manager):
        """After save succeeds, old checkpoint should be deleted."""
        ckpt_100 = self._create_checkpoint_dir(100)
        manager.previous_saved_paths = {"default": [ckpt_100]}

        ckpt_200 = self._create_checkpoint_dir(200)
        manager.register_checkpoint(ckpt_200, max_ckpt_to_keep=1)

        assert not os.path.exists(ckpt_100)
        assert os.path.exists(ckpt_200)
        assert manager.previous_saved_paths["default"] == [ckpt_200]

    def test_max_ckpt_2_keeps_one_before_save(self, manager):
        """With max_ckpt_to_keep=2, pre-save cleanup keeps 1 checkpoint."""
        ckpt_100 = self._create_checkpoint_dir(100)
        ckpt_200 = self._create_checkpoint_dir(200)
        manager.previous_saved_paths = {"default": [ckpt_100, ckpt_200]}

        manager.ensure_checkpoint_capacity(max_ckpt_to_keep=2)

        assert not os.path.exists(ckpt_100)
        assert os.path.exists(ckpt_200)
        assert len(manager.previous_saved_paths["default"]) == 1

    def test_max_ckpt_0_keeps_all(self, manager):
        """max_ckpt_to_keep=0 means unlimited - no deletions."""
        ckpt_100 = self._create_checkpoint_dir(100)
        ckpt_200 = self._create_checkpoint_dir(200)
        manager.previous_saved_paths = {"default": [ckpt_100, ckpt_200]}

        manager.ensure_checkpoint_capacity(max_ckpt_to_keep=0)
        ckpt_300 = self._create_checkpoint_dir(300)
        manager.register_checkpoint(ckpt_300, max_ckpt_to_keep=0)

        assert os.path.exists(ckpt_100)
        assert os.path.exists(ckpt_200)
        assert os.path.exists(ckpt_300)
        assert len(manager.previous_saved_paths["default"]) == 3

    def test_full_save_cycle_max_ckpt_1(self, manager):
        """Simulate multiple save cycles with max_ckpt_to_keep=1."""
        # First save
        manager.ensure_checkpoint_capacity(1)
        ckpt_100 = self._create_checkpoint_dir(100)
        manager.register_checkpoint(ckpt_100, 1)
        assert manager.previous_saved_paths["default"] == [ckpt_100]

        # Second save - existing checkpoint must survive pre-save
        manager.ensure_checkpoint_capacity(1)
        assert os.path.exists(ckpt_100), "Bug: checkpoint deleted before save!"

        ckpt_200 = self._create_checkpoint_dir(200)
        manager.register_checkpoint(ckpt_200, 1)
        assert not os.path.exists(ckpt_100)
        assert manager.previous_saved_paths["default"] == [ckpt_200]

        # Third save
        manager.ensure_checkpoint_capacity(1)
        assert os.path.exists(ckpt_200), "Bug: checkpoint deleted before save!"

        ckpt_300 = self._create_checkpoint_dir(300)
        manager.register_checkpoint(ckpt_300, 1)
        assert not os.path.exists(ckpt_200)
        assert manager.previous_saved_paths["default"] == [ckpt_300]

    def test_default_tag_backward_compatible(self, manager):
        """Omitting `tag` entirely must behave exactly like the old single-list design."""
        manager.ensure_checkpoint_capacity(1)
        ckpt_100 = self._create_checkpoint_dir(100)
        manager.register_checkpoint(ckpt_100, 1)
        assert manager.previous_saved_paths == {"default": [ckpt_100]}

        manager.ensure_checkpoint_capacity(1)
        ckpt_200 = self._create_checkpoint_dir(200)
        manager.register_checkpoint(ckpt_200, 1)
        assert not os.path.exists(ckpt_100)
        assert manager.previous_saved_paths == {"default": [ckpt_200]}

    def test_tags_are_independent_fifos(self, manager):
        """
        Regression test for the "best checkpoint deleted by periodic rotation" bug:
        a checkpoint saved under a different tag must survive rotation of the
        "default" tag's lineage, and vice versa.
        """
        ckpt_default_1 = self._create_checkpoint_dir(100)
        manager.register_checkpoint(ckpt_default_1, max_ckpt_to_keep=1, tag="default")

        ckpt_best = self._create_checkpoint_dir(101)
        manager.register_checkpoint(ckpt_best, max_ckpt_to_keep=1, tag="best")

        # Best checkpoint must not have evicted the default-tag checkpoint (different lineage).
        assert os.path.exists(ckpt_default_1)
        assert os.path.exists(ckpt_best)

        ckpt_default_2 = self._create_checkpoint_dir(120)
        manager.ensure_checkpoint_capacity(max_ckpt_to_keep=1, tag="default")
        manager.register_checkpoint(ckpt_default_2, max_ckpt_to_keep=1, tag="default")

        # The new "default" save should evict the OLD "default" checkpoint...
        assert not os.path.exists(ckpt_default_1)
        assert os.path.exists(ckpt_default_2)
        # ...but must never touch the "best" checkpoint, which lives in its own lineage.
        assert os.path.exists(ckpt_best)
        assert manager.previous_saved_paths["best"] == [ckpt_best]
        assert manager.previous_saved_paths["default"] == [ckpt_default_2]

    def test_constant_path_overwrite_is_not_self_evicted(self, manager):
        """
        Regression test for the "best checkpoint deleted in the same step it is
        saved" bug. The best checkpoint always writes to the SAME path
        (".../best/actor", no global_step), and is re-registered every time
        accuracy improves. Re-registering an identical path must be treated as an
        in-place overwrite, NOT a new FIFO entry — otherwise rotation immediately
        evicts "the oldest", which is the directory that was just (re)written,
        leaving the tracked path pointing at a deleted directory.
        """
        best_path = os.path.join(self.test_dir, "best", "actor")
        os.makedirs(best_path, exist_ok=True)
        with open(os.path.join(best_path, "model.txt"), "w") as f:
            f.write("step=0")

        # First best save (e.g. step 0).
        manager.ensure_checkpoint_capacity(max_ckpt_to_keep=1, tag="best")
        manager.register_checkpoint(best_path, max_ckpt_to_keep=1, tag="best")
        assert os.path.exists(best_path)

        # Second best save to the SAME path (e.g. step 10, accuracy improved). The
        # FSDP save overwrites best_path in place; re-register must not delete it.
        with open(os.path.join(best_path, "model.txt"), "w") as f:
            f.write("step=10")
        manager.ensure_checkpoint_capacity(max_ckpt_to_keep=1, tag="best")
        manager.register_checkpoint(best_path, max_ckpt_to_keep=1, tag="best")

        assert os.path.exists(best_path), "Bug: best checkpoint deleted in the step it was saved!"
        assert os.path.exists(os.path.join(best_path, "model.txt"))
        assert manager.previous_saved_paths["best"] == [best_path]
