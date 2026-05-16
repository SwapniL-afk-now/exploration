import torch

from verl.experimental.fepo.checkpoint import OptimizerStepCheckpointManager


def test_optimizer_step_checkpoint_roundtrip(tmp_path):
    manager = OptimizerStepCheckpointManager(tmp_path, keep_last=1)
    model_state = {"solver": {"w": torch.tensor([1.0, 2.0])}}
    optimizer_state = {"solver": {"step": 2}}
    saved = manager.save(
        global_step=2,
        payload={"cursor": 10},
        model_state=model_state,
        optimizer_state=optimizer_state,
        dataloader_state={"cursor": 10},
    )
    assert saved.path.exists()
    assert manager.latest_checkpoint_dir() == saved.path

    state = manager.load()
    assert state["global_step"] == 2
    assert state["payload"]["cursor"] == 10
    assert torch.equal(state["model_state"]["solver"]["w"], torch.tensor([1.0, 2.0]))

    manager.save(global_step=3, payload={}, model_state={}, optimizer_state={}, dataloader_state={})
    assert not saved.path.exists()
