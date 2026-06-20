#!/usr/bin/env python3
"""Live per-step grad_norm comparison: current run vs previous wandb runs.

Usage: python scripts/gradnorm_compare.py [current_run_log]
Default current log = /tmp/jepa_nowarmup_run.log
"""
import re, os, sys, glob, time

CUR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/jepa_nowarmup_run.log"

# (label, path-to-output.log, note)
PREV = [
    ("dtoh8rej(wu?)", "wandb/run-20260619_194835-dtoh8rej/files/output.log", "step300, old code"),
    ("m3xfn18u",      "wandb/run-20260618_090507-m3xfn18u/files/output.log", "step413"),
    ("ji9xn9k4(wu10)","wandb/run-20260620_121038-ji9xn9k4/files/output.log", "warmup=10"),
]

STEP_RE  = re.compile(r'step:(\d+) ')
ACT_RE   = re.compile(r'actor/grad_norm:([0-9.eE+-]+)')
JEPA_RE  = re.compile(r'jepa/grad_norm:([0-9.eE+-]+)')

def parse(path):
    d = {}
    if not path or not os.path.exists(path):
        return d
    for line in open(path, errors="ignore"):
        m = STEP_RE.search(line)
        if not m:
            continue
        a, j = ACT_RE.search(line), JEPA_RE.search(line)
        if a and j:
            d[int(m.group(1))] = (float(a.group(1)), float(j.group(1)))
    return d

def fmt(v):
    return f"{v[0]:.4f}/{v[1]:.4f}" if v else "    -      "

def render():
    cur = parse(CUR)
    prev = [(lab, parse(p), note) for lab, p, note in PREV]
    os.system("clear")
    print("\033[1mPer-step  actor_gn / jepa_gn   (current run = warmup=0)\033[0m")
    print(f"current log: {CUR}   updated {time.strftime('%H:%M:%S')}\n")
    cols = ["NOW(wu0)"] + [lab for lab, _, _ in prev]
    print("step | " + " | ".join(f"{c:^13}" for c in cols))
    print("-" * (7 + 16 * len(cols)))
    maxstep = max([max(cur) if cur else 0] + [max(d) if d else 0 for _, d, _ in prev])
    show = sorted(set(list(range(1, min(maxstep, 15) + 1)) + list(cur.keys())))
    for s in show:
        cells = [fmt(cur.get(s))] + [fmt(d.get(s)) for _, d, _ in prev]
        flag = "  <= current" if s in cur else ""
        print(f"{s:>4} | " + " | ".join(f"{c:^13}" for c in cells) + flag)
    print(f"\ncurrent run reached step {max(cur) if cur else 0}")

if __name__ == "__main__":
    loop = "--once" not in sys.argv
    while True:
        render()
        if not loop:
            break
        time.sleep(15)
