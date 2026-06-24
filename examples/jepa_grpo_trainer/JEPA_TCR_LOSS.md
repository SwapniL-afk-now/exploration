# Teacher‑Correct Representation Alignment for Reasoning Policies

## Abstract

We augment reinforcement learning of a reasoning policy with an auxiliary
representation‑alignment objective. Alongside the usual reward‑driven policy
gradient, the model is trained so that the internal representation it forms while
solving a problem is drawn toward the representation of a *verified‑correct*
solution to the same problem. Correct solutions are produced once, offline, by a
teacher and re‑encoded into the policy's own representation space; at training
time no teacher is present and the alignment target is a fixed vector. We refer
to this objective as **Teacher‑Correct Representation (TCR) alignment**. This
document describes the methodology.

---

## 1. Motivation

A policy trained purely from a scalar reward receives no information about *what
a correct internal state looks like* — only about whether its emitted text
happened to score. We hypothesise that supervising the policy's latent
representation, in addition to its behaviour, accelerates the acquisition of
correct reasoning. The supervision signal is the latent of a known‑correct
solution. Crucially, the target is expressed in the policy's *own* representation
space, so the alignment is a within‑space attraction that requires neither a
learned projection nor a cross‑dimensional similarity.

The method therefore imposes two simultaneous objectives on the policy:

* a **behavioural** objective — the reinforcement‑learning policy gradient, which
  rewards text that solves the task; and
* a **representational** objective — TCR alignment, which shapes the latent
  formed at the point of prediction to resemble that of a correct solution.

The behavioural objective is the **Dr.GRPO** clipped surrogate, optionally
augmented by a **KL penalty** anchoring the policy to a frozen reference, both
detailed in Section 4.7.

---

## 2. Overview of the Method

The method has two stages. The first is performed once, offline; the second runs
throughout policy optimisation.

**Stage I — Target construction (offline).** A teacher model generates several
candidate solutions per training problem. Each candidate is verified by the same
correctness checker later used to compute reward, and only verified‑correct
candidates are retained. Each retained solution is then encoded by a frozen
reference encoder whose representation space coincides with the policy's, and the
resulting unit‑norm vectors are stored as the alignment targets for that problem.

**Stage II — Joint optimisation (online).** During reinforcement learning, the
policy's representation for a rollout on a given problem is pulled toward one of
that problem's precomputed correct‑solution targets, while an anti‑collapse
regulariser prevents the representation from degenerating. This term is optimised
jointly with the policy‑gradient loss.

The remainder of this document specifies each stage.

---

## 3. Stage I: Constructing Correct‑Solution Targets

### 3.1 Candidate generation

For every training problem the teacher model produces a set of `K` candidate
solutions by stochastic sampling (temperature `0.8`, nucleus `0.95`, with
`K = 8` in our runs). We call this the set of **all generated responses**. It
constitutes a candidate pool only; the majority of these responses are discarded
in the next step and never influence training. The teacher is a separate
generation model selected for solution quality; we did **not** use a 3B teacher
in our runs.

### 3.2 Verification and selection

Each of the `K` candidates is evaluated by the *same* correctness verifier used
to assign reward during reinforcement learning. Only candidates judged correct
are kept, up to a cap of `N` per problem (`N = 4` in our runs). We call this
retained subset the **correct responses**. A problem for which the teacher
produces no correct candidate yields no target and is simply absent from the
supervision signal; the method never fabricates a target. This verification gate
guarantees that every alignment target corresponds to a demonstrably correct
solution.

### 3.3 Encoding into the policy's representation space

Each retained correct solution, concatenated with its problem statement, is
passed through a **frozen reference encoder of the same size and architecture as
the policy** — in our runs the `Qwen2.5‑Math‑1.5B‑Instruct` model. We take the
hidden state of the final layer at the last real token, and L2‑normalise it to
obtain a unit vector `z⁺` on the representation hypersphere. All target
embeddings are produced by this 1.5B reference encoder; **no 3B model is used to
generate embeddings**.

This encoding procedure is identical to the one applied online to the policy's
own hidden states (Section 4.2). Because the reference encoder shares the
policy's hidden dimension (1536), the targets live in exactly the policy's
representation space. Two consequences follow, both deliberate:

1. no projection head is required to relate predictions and targets, and
2. similarity is an ordinary within‑space cosine, never a cross‑dimensional one.

For each problem we obtain a small set of unit target vectors `{z⁺}` — one per
retained correct solution — which are stored for use in Stage II. The teacher
contributes nothing further: only its *text*, re‑encoded by the reference model,
survives as a constant target.

---

## 4. Stage II: Joint Optimisation

### 4.1 Anchors

At each optimisation step the policy generates a group of rollouts per problem.
Any rollout on a problem that has stored targets is eligible to serve as an
**anchor**. An anchor is presented to the policy in the form
`[problem, policy_solution, ⟨predict⟩]`: the problem, the policy's own response,
and a dedicated prediction token at which the aligned representation is read.

Which rollouts are used as anchors is a design choice:

* **correct anchors** — only rollouts that themselves earned reward; the policy
  is taught to make a correct trajectory's latent resemble the correct target;
* **wrong anchors** — only rollouts that failed; from the context of a *failed*
  trajectory the prediction token must still recover the correct latent, a form
  of *latent correction*, without forcing the failed text's own state to appear
  correct;
* **both** — the union of the two.

The correctness of the *anchor* (a property of the policy's rollout) is
independent of the correctness of the *target*, which is guaranteed by Stage I.

### 4.2 Prediction and pairing

Each anchor is encoded by the policy with the same final‑layer, last‑token,
L2‑normalised read‑out used to build the targets, producing a unit prediction
`p`. Each anchor is paired with one target vector `z⁺` drawn from its problem's
stored set (cyclically or at random across the available targets).

### 4.3 Alignment loss

The per‑anchor loss is the cosine distance between the prediction and the
**stop‑gradient** target,

```
ℓ_i = 1 − ⟨ p_i , sg(z⁺_i) ⟩ .
```

The stop‑gradient is essential: the target is a fixed correct‑solution latent, so
gradient flows only through the policy's prediction. The policy moves toward the
correct latent; the target never moves toward the policy.

### 4.4 Reward‑stratified, problem‑level aggregation

Anchors must not be aggregated by a flat mean, because a problem that happens to
yield many eligible rollouts would then dominate the objective. We instead
aggregate at the level of problems and stratify by reward. For a problem `x` with
correct anchors `C_x` and wrong anchors `W_x`,

```
L(x) = ½ · mean_{i∈C_x} ℓ_i  +  ½ · mean_{i∈W_x} ℓ_i ,
```

when both subsets are present, and the mean over whichever subset is non‑empty
otherwise. The alignment loss is then the mean of `L(x)` over the problems
present in the batch. The fixed ½/½ weighting defines a *reward‑stratified anchor
distribution* — it equalises the influence of correct and wrong contexts — and is
not a tunable per‑class loss weight.

### 4.5 Anti‑collapse regularisation

Minimising the alignment term alone admits a trivial solution in which the
encoder maps every input to a single point. We prevent this with a SIGReg
regulariser computed over the policy predictions alone, since the predictions are
the only quantity carrying gradient. Because the predictions are already
unit‑normalised, the regulariser is applied with a global‑radius rescaling
without which it would be inert on the hypersphere; this rescaling is
load‑bearing. Notably, TCR contains **no explicit repulsion or separation term**
between predictions and incorrect references — the verified‑correct target
replaces the positive reference, and SIGReg alone supplies anti‑collapse.

### 4.6 Combined objective

The representation objective is a convex combination of alignment and
regularisation,

```
L_TCR = (1 − λ) · L_align  +  λ · L_SIGReg ,
```

with `λ` balancing the two (`λ = 0.5` by default). How this representation loss is
combined with the behavioural objective is specified in Section 4.8.

### 4.7 Behavioural objective: the full Dr.GRPO loss and the KL term

**Group‑relative advantages.** For each problem the policy samples a group of
`G` rollouts and each rollout `i` receives a scalar reward `R_i`. The advantage
is the reward centred on the group mean, **without** division by the group
standard deviation — the Dr.GRPO variant, which removes the length/variance bias
introduced by standard GRPO normalisation:

```
Â_i = R_i − (1/G) Σ_{j=1}^{G} R_j .
```

The advantage is broadcast to every completion token of rollout `i`.

**Clipped policy‑gradient surrogate.** Let
`r_{i,t} = π_θ(a_{i,t} | s_{i,t}) / π_old(a_{i,t} | s_{i,t})` be the per‑token
probability ratio between the current policy and the rollout‑time policy. The
Dr.GRPO loss is the PPO‑style clipped surrogate, averaged over all valid
completion tokens `𝒯`:

```
L_PG(θ) = − (1/|𝒯|) Σ_{(i,t)∈𝒯}
            min( r_{i,t} · Â_i ,  clip(r_{i,t}, 1−ε, 1+ε) · Â_i ) ,
```

with clip range `ε = 0.2`. For numerical stability the log‑ratio
`log r_{i,t}` is additionally clamped to `[−8, 8]` before exponentiation.

**KL penalty.** A KL penalty toward a frozen reference policy `π_ref` is
*supported* as an optional behavioural regulariser — a low‑variance KL estimator
added to the actor loss (not folded into the reward):

```
L_actor(θ) = L_PG(θ) + β · KL( π_θ ‖ π_ref ) .
```

It is **disabled in the run documented here** (`β = 0`, and no KL is added to the
reward), so the behavioural objective reduces to `L_PG` alone.

### 4.8 The full training objective

The two objectives are **optimised by two separate backward/optimiser steps on
the same parameters**, not by a single summed scalar. Each policy update applies
the Dr.GRPO actor loss; a distinct JEPA update applies the representation loss
scaled by a coefficient `α` (optionally warmed up linearly from `0` over the
first few updates):

```
Policy update:          minimise   L_actor(θ) = L_PG(θ)  (+ β·KL,  β = 0 here)
Representation update:  minimise   α · L_TCR(θ) = α · [ (1−λ)·L_align + λ·L_SIGReg ]
```

Conceptually the model is trained against

```
L_total = L_PG  (+ β·KL,  disabled here)  +  α · L_TCR ,
```

with the two terms applied as decoupled gradient steps in practice. Setting
`α = 0` recovers pure Dr.GRPO; the KL term can be re‑enabled by setting `β > 0`.

---

## 5. Properties and Rationale

| Design decision | Rationale |
|---|---|
| Teacher invoked only offline | The training loop never loads a teacher; targets are deterministic and cheap to reuse. |
| Retain only verified‑correct responses | Every alignment target corresponds to a provably correct solution. |
| Re‑encode targets with a policy‑size reference encoder | Targets share the policy's 1536‑d space → no projection head, no cross‑dimensional similarity. |
| Stop‑gradient on the target | The policy moves toward the correct latent, not the reverse; targets are constants. |
| Problem‑level, reward‑stratified aggregation | Prevents prolific problems from dominating; equalises correct and wrong contexts. |
| SIGReg over predictions only | Supplies anti‑collapse without a separate repulsion target; global‑radius rescaling retained. |
| Wrong‑anchor mode | Enables latent correction: failed trajectories learn to point at the correct latent. |

---

## 6. Summary

TCR alignment converts a stronger model's *verified‑correct* solutions into fixed
targets inside the policy's own representation space, then trains the policy so
that its prediction‑point latent is attracted to those targets while an
anti‑collapse regulariser preserves representational diversity. Correctness is
enforced twice — once when selecting teacher responses and again, implicitly,
through the reward used in joint optimisation — and the teacher's cost is paid
exactly once, before training begins.
