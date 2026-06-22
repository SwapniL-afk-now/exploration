# jepa-separation-loss v3 — merged objective (CLReg $L_s$ folded into the original total loss)

This supersedes `jepa_separation_loss.md`. It keeps that file's total objective and $L_{\text{align}}$/$L_{\text{SIGReg}}$ verbatim, and replaces the heuristic margin-hinge $L_s$ with the CLReg-grounded contrastive term from `jepa_separation_loss_v2_clreg.md`. It also fixes one architectural inconsistency that falls out of doing the splice properly (see "Stop-gradient fix" below) — this is the actual bug being corrected, not just a cosmetic merge.

## Total objective (unchanged from v1)

$$
L_{\text{total}} = L_{\text{GRPO}}(\text{CoT} + \text{Code}) \;+\; \alpha \, L_{\text{sep}}
$$

$$
L_{\text{sep}} = (1 - \lambda)\,\big(L_{\text{align}} + w_{\text{sep}} \, L_{\text{s}}\big) \;+\; \lambda \, L_{\text{SIGReg}}
$$

## $L_{\text{align}}$ (unchanged)

$$
L_{\text{align}} = \frac{1}{B}\sum_{i=1}^{B} \Big(1 - \big\langle \operatorname{Pred}(\operatorname{Enc}(\text{CoT}_i)),\; \operatorname{sg}\!\big(e^{c}_i\big) \big\rangle \Big)
$$

$e^c_i = \operatorname{Enc}(\text{Code}_i^{\text{correct}})$, $B$ = number of correct rollouts in the batch. $\operatorname{sg}(\cdot)$ = stop-gradient.

## $L_s$ — replaced (this is the fix)

The v1 hinge is gone. In its place, per GRPO group $g$ (one prompt, with correct rollouts $\mathcal C_g$ and wrong rollouts $\mathcal W_g$):

$$
p_i = \operatorname{Pred}(\operatorname{Enc}(\text{CoT}_i)), \quad i \in \mathcal C_g
$$

$$
e_k^{w} = \operatorname{sg}\Big(\operatorname{Enc}\big(\text{CoT}_k \oplus \text{Code}_k\big)\Big), \quad k \in \mathcal W_g
$$

$$
L_{\text{s}} = \frac{1}{|\mathcal C_g|}\sum_{i \in \mathcal C_g} \frac{1}{|\mathcal W_g|}\sum_{k \in \mathcal W_g} \left[-2\tau_{\text{sep}}\log\sigma\!\left(\frac{s(p_i,\, \operatorname{sg}(e_i^c)) - s(p_i,\, e_k^{w})}{\tau_{\text{sep}}}\right)\right]
$$

with $s(\cdot,\cdot)$ cosine similarity on L2-normalized vectors, batch loss = mean of the per-group $L_s$ over all groups in the batch. $L_s = 0$ for any group with $\mathcal W_g = \emptyset$.

This is the averaged-pairwise-DPO form from v2 (the form that keeps CLReg's Prop. 3.2/Cor. 3.3 exact per pair). The InfoNCE alternative from v2 is available as a config flag (`mode="info"`) but is not the default — see v2 doc for the grounding trade-off.

### Combination rule — stated explicitly

$L_s$ is a **full cross product**, not a matched 1:1 pairing: every correct anchor $p_i$, $i \in \mathcal C_g$, is compared against **every** wrong joint embedding $e_k^w$, $k \in \mathcal W_g$, in that same group — $|\mathcal C_g| \times |\mathcal W_g|$ pairs total per group. There is no notion of "this correct rollout pairs with that wrong rollout"; a correct rollout has no inherent correspondence to any one wrong rollout, so every correct anchor is pushed away from the *entire* wrong set, not from one designated counterpart. The double sum and double average in the $L_s$ equation above is exactly this: inner sum/average over all $k$ for a fixed $i$, outer sum/average over all $i$.

### Stop-gradient — stated explicitly, per tensor

| Tensor | Definition | Gradient? |
|---|---|---|
| $p_i = \operatorname{Pred}(\operatorname{Enc}(\text{CoT}_i))$, $i \in \mathcal C_g$ | predictor output on **correct CoT only** | **Yes — the only tensor with gradient in $L_s$ or $L_\text{align}$** |
| $e_i^c = \operatorname{sg}(\operatorname{Enc}(\text{Code}_i))$, $i \in \mathcal C_g$ | correct rollout's code embedding (the positive) | **No — always frozen**, in both $L_\text{align}$ and $L_s$ |
| $e_k^w = \operatorname{sg}(\operatorname{Enc}(\text{CoT}_k \oplus \text{Code}_k))$, $k \in \mathcal W_g$ | wrong rollout's **joint** CoT+Code embedding (the negative) | **No — always frozen** |

Only the correct rollouts' CoT ever feeds the trainable predictor pathway. Wrong rollouts contribute **only** as frozen negatives via their joint embedding — there is no $p_k$ computed for wrong rollouts at all, and no path by which a wrong rollout's CoT or code ever receives gradient from $L_\text{sep}$. This is the single invariant that fixes v1's bug (v1 had $e^c$ receiving gradient inside $L_s$ specifically, inconsistent with its frozen role in $L_\text{align}$); after this fix, $e^c$'s gradient status is identical in every term that uses it.

**What changed vs. v1's $L_s$, concretely:**
- Negative target: was $e^w_i = \operatorname{Enc}(\text{Code}^w_i)$, a single fixed wrong-code embedding per correct anchor. Now it's $e^w_k$, the *joint* (CoT+Code) embedding of *every* wrong rollout in the group — a variable-size set, not a fixed pair.
- Loss form: was a margin hinge with a hard zero region. Now a log-sigmoid (DPO-style) term with non-vanishing gradient everywhere.
- Hyperparameter: $m_{\text{sep}}$ (margin) is gone, replaced by $\tau_{\text{sep}}$ (temperature).

## $L_{\text{SIGReg}}$ — unchanged definition, **one open decision flagged**

v1 defined the isotropy pool as $[\,p^c, e^c, e^w\,]$ where $e^w$ was the single code-only wrong embedding. Now that $L_s$ uses joint (CoT+Code) wrong embeddings instead of code-only ones, there's a real choice:

- **(a) Keep the pool code-only**: pool stays $[\,p^c, e^c, e^w_{\text{code-only}}\,]$, computed exactly as in v1, fully decoupled from whatever $L_s$ uses. Simplest, no behavior change to this term.
- **(b) Switch the pool to match $L_s$'s negatives**: pool becomes $[\,p^c, e^c, \{e^w_k\}_{k\in\mathcal W_g}\}]$, i.e. the joint embeddings, so the anti-collapse term sees the same representation space $L_s$ is actually shaping.

**Recommendation: (a).** $L_{\text{SIGReg}}$'s job is to stop the *code* embedding space from collapsing; the joint CoT+Code embedding is a different representation (different encoder call, different semantic content) introduced specifically for $L_s$'s negative set, and folding it into the isotropy pool conflates two regularization targets. Keep $L_{\text{SIGReg}}$ exactly as in v1, computed only from code-only embeddings. Flagging this explicitly rather than silently picking — confirm before implementing if (b) was actually intended.

## Stop-gradient rule — fixed, now consistent everywhere

This is the actual bug the merge surfaces. **v1's stop-gradient rule was inconsistent across its two terms:** in $L_{\text{align}}$, $e^c$ was frozen (correct, target embedding shouldn't move just because a predictor is chasing it). But in v1's $L_s$, $e^c$ was the one *receiving* gradient (only $e^w$ was frozen) — i.e. the same tensor ($e^c$) was a frozen target in one loss term and a trainable variable in the other, in the same backward pass. That's not just inelegant, it means the code encoder was being pulled by $L_s$ for reasons unrelated to matching its own CoT, which fights the "two separately-weighted terms" design goal of $L_\text{align}$ vs. $L_s$ described in v2.

**Fixed rule (matches v2's architecture constraint):**
- $e^c$ is **always** stop-gradiented, in every term that touches it ($L_{\text{align}}$ and $L_s$ alike). The code encoder pathway receives **zero gradient** from $L_{\text{sep}}$. It is a frozen reference, full stop.
- $e^w_k$ (joint CoT+Code embedding of wrong rollouts) is likewise always stop-gradiented.
- The **only** tensor in $L_{\text{sep}}$ that carries gradient back into the model is $p_i = \operatorname{Pred}(\operatorname{Enc}(\text{CoT}_i))$, for $i$ ranging over correct rollouts. Everything else in $L_{\text{sep}}$ is a frozen target.
- $L_{\text{GRPO}}$ still trains the policy (CoT + Code generation) normally through the RL loss — this stop-gradient rule only governs the auxiliary representation-alignment term $L_{\text{sep}}$, not the main RL objective.

## Symbols (consolidated)

| Symbol | Meaning |
|---|---|
| $\langle\cdot,\cdot\rangle$, $s(\cdot,\cdot)$ | cosine similarity, L2-normalized embeddings |
| $\operatorname{Enc}$ | shared embedding encoder (CoT and Code both pass through it) |
| $\operatorname{Pred}$ | predictor head, JEPA-style, maps CoT embedding → predicted code embedding |
| $\operatorname{sg}(\cdot)$ | stop-gradient |
| $\mathcal C_g, \mathcal W_g$ | correct / wrong rollout indices within GRPO group $g$ |
| $e^c_i$ | $\operatorname{Enc}(\text{Code}_i^{\text{correct}})$, frozen, reused by $L_{\text{align}}$ and $L_s$ |
| $e^w_k$ | $\operatorname{sg}(\operatorname{Enc}(\text{CoT}_k \oplus \text{Code}_k))$, joint embedding of wrong rollout $k$, frozen |
| $p_i$ | $\operatorname{Pred}(\operatorname{Enc}(\text{CoT}_i))$, the only trainable quantity in $L_{\text{sep}}$ |
| $\tau_{\text{sep}}$ | temperature for $L_s$ (replaces v1's margin $m_{\text{sep}}$) |
| $\alpha, \lambda, w_{\text{sep}}$ | unchanged top-level weighting hyperparameters |

## Hyperparameters — starting point

$$
\alpha = 0.005,\quad \lambda = 0.5,\quad w_{\text{sep}} = 1.0,\quad \tau_{\text{sep}} = 0.5,\quad \text{mode} = \texttt{dpo}
$$

Re-tune $w_{\text{sep}}$ once live (loss scale moved from a bounded hinge to an unbounded log-sigmoid). Sweep $\tau_{\text{sep}} \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$.