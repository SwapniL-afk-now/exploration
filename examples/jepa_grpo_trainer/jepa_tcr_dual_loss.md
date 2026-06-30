# Total objective — `loss_type=jepa-tcr-dual` + DAPO

Two **separate backward passes / optimizer steps** on the shared LoRA weights
(`update_actor`, then `jepa_update` scaled by `alpha`), not one fused graph:

```
L_total = L_DAPO(on shaped reward)  +  alpha * L_JEPA          (alpha = 0.005)
```

## 1. DAPO term (CoT + Code rollouts)

Clip-higher + dual-clip, token-mean, no KL:

```
L_DAPO = -E_t[ min( r_t * A_t , clip(r_t, 1-eps_low, 1+eps_high) * A_t )  (dual-clipped at c) ]
         eps_low = 0.2,  eps_high = 0.28,  c = 10
```

`A_t` = group-relative GRPO advantage, **std-normalized**.

**JEPA reward shaping enters HERE**, before the advantage (ray_trainer Step 2.5).
Per row, at its last response token:

```
r_tilde = r + beta * s_hat          beta = 0.5
```

`s_hat` = within-stratum standardized teacher-alignment score (per view, sigma floor 0.1).
`A_t` is then computed over `r_tilde`, so beta-shaping is standardized per group with the raw reward.

## 2. JEPA term — jepa-tcr-dual (core_algos.py:355)

```
L_JEPA = (1 - lambda) * ( c_cot*A_cot + c_code*A_code + c_self*w*L_self )  +  lambda * L_SIGReg
         lambda = 0.3,  w = 1.0
```

with `<.>` = cosine, `sg` = stop-grad:

- `A_cot   = strat-mean( 1 - <pred_cot,  sg(z_teacher_cot)> )`   CoT student -> cached CoT teacher target
- `A_code  = strat-mean( 1 - <pred_code, sg(z_teacher_code)> )`  Code student -> cached Code teacher target
- `L_self  = mean( 1 - <pred_cot, sg(b)> )`   CoT pred -> stop-grad Code_S **boundary read** of paired Code rollout
- `L_SIGReg` = anti-collapse (Epps-Pulley over pooled [pred_cot; pred_code]), M = 4096 directions

Both teacher targets are frozen cached constants (stop-grad); self-target also stop-grad
(only `pred_cot` moves toward it).

## Auto-off coefficients (c_cot, c_code, c_self)

Each `c` in {0, 1} = per-arm plateau latch (core_algos.py:352-354). A latched arm gives
**zero gradient to its align term AND zero reward shaping for its view**, but its preds
still feed SIGReg. As arms plateau, the objective collapses progressively:

```
all on  ->  some off  ->  all 3 off  =>  L_JEPA skipped entirely (incl. SIGReg)
                                      =>  L_total = L_DAPO   (beta-shaping off too)
```

Once `cos_cot`, `cos_code`, `cos_self` have all plateaued, training continues as
**pure DAPO** for the rest of the run.
