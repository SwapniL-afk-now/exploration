import wandb

ENTITY = "ismamnurswapnil-bangladesh-university-of-engineering-and"
PROJECT = "verl_drgrpo_deepscaler"
RUN_IDS_IN_ORDER = ["wsnycgb6", "5t8m5eu6"]  # steps 1-80, then 81+
MERGED_NAME = "deepseek_r1_distill_qwen_1_5b_drgrpo_fsdp-20260616_154614-merged"

api = wandb.Api()

merged = wandb.init(entity=ENTITY, project=PROJECT, name=MERGED_NAME, job_type="merged")

seen_steps = set()
for run_id in RUN_IDS_IN_ORDER:
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    history = run.scan_history()
    rows = sorted(history, key=lambda r: r.get("_step", 0))
    for row in rows:
        step = row.get("_step")
        if step is None or step in seen_steps:
            continue
        seen_steps.add(step)
        log_row = {k: v for k, v in row.items() if not k.startswith("_")}
        merged.log(log_row, step=step)
    print(f"merged {len(rows)} rows from {run_id}")

merged.finish()
print(f"Merged run: {merged.url}")
