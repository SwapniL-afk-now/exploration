# Detailed Plan: GRPO + JEPA-TCR with Correct and Optional Wrong Response Anchors

## 1. Goal

The goal is to extend the current JEPA-TCR loss so that it can optionally use both correct and wrong student responses as JEPA anchors, while keeping the method minimal and avoiding extra loss-weight hyperparameters for correct and wrong responses.

The main training objective remains:

[
\mathcal{L}_{total}
===================

\mathcal{L}*{GRPO}
+
\lambda*{JEPA}\mathcal{L}*{JEPA}
+
\beta \mathcal{L}*{KL}
]

There should be only one global JEPA coefficient:

[
\lambda_{JEPA}
]

No separate `correct_jepa_weight` or `wrong_jepa_weight` should be added.

---

## 2. Current Correct-Only Design

The current implementation does this:

Student side:

[
[x, y_S^+, \texttt{[PRED]}]
]

where:

* (x) = question/prompt
* (y_S^+) = correct student-generated response
* `[PRED]` = predictor token

The student embedding is extracted from the last `[PRED]` hidden state:

[
\hat{z}_S
=========

h_\theta([x,y_S^+,\texttt{[PRED]}])_{\texttt{[PRED]}}
]

Target side:

[
[x, y_T^+]
]

where:

* (y_T^+) = cached teacher-generated correct response
* target is encoded offline by the frozen 1.5B reference model

The target embedding is extracted from the last real token:

[
z_T^+
=====

h_{\text{ref}}([x,y_T^+])_{\text{last real token}}
]

Then the JEPA loss is:

[
\mathcal{L}_{JEPA}
==================

1-
\cos(\hat{z}_S,\text{sg}(z_T^+))
]

Currently, only correct student responses participate in this loss.

---

## 3. New Extended Design

The new design should keep the current correct-only behavior as the default, but add an optional switch:

```python
include_wrong_jepa: bool = False
```

When:

```python
include_wrong_jepa = False
```

JEPA anchors are:

[
\mathcal{A}
===========

{y_{S,i}: r_i=1}
]

This is the current correct-only behavior.

When:

```python
include_wrong_jepa = True
```

JEPA anchors are:

[
\mathcal{A}
===========

{y_{S,i}: r_i=1}
\cup
{y_{S,i}: r_i=0}
]

So both correct and wrong student responses participate in the JEPA branch.

However, every selected anchor uses the same prediction format:

[
[x, y_{S,i}, \texttt{[PRED]}]
\rightarrow
z_T^+
]

This means wrong responses are not directly forced to have correct hidden states. Instead, the `[PRED]` token learns to predict the latent representation of the teacher-correct solution from the student response context.

---

## 4. Important Conceptual Rule

Do not directly align:

[
h_\theta([x,y_S^-])_{\text{last real token}}
\rightarrow
z_T^+
]

This is risky because it forces the hidden state of a wrong textual trajectory to look like the hidden state of a correct solution.

Instead, always align:

[
h_\theta([x,y_S^-,\texttt{[PRED]}])_{\texttt{[PRED]}}
\rightarrow
z_T^+
]

This means:

> Given the student’s response, whether correct or wrong, the predictor token should predict the latent representation of the cached teacher-correct solution.

For correct responses, this refines successful reasoning.

For wrong responses, this becomes a latent correction-prediction signal.

---

## 5. Offline Target Cache

The offline target cache does not need to change for this extension.

For each question (x), the teacher model generates one or more correct responses:

[
y_T^+
]

These responses are verified and stored.

Then the frozen student-sized reference model encodes:

[
[x, y_T^+]
]

The cached target is:

[
z_T^+
=====

\text{L2Norm}
\left(
h_{\text{ref}}([x,y_T^+])_{\text{last real token}}
\right)
]

This target remains detached during training.

Do not append `[PRED]` to the target side for the main method.

Target side stays:

[
[x, y_T^+]
]

not:

[
[x, y_T^+, \texttt{[PRED]}]
]

---

## 6. Student Anchor Construction

During GRPO training, the student generates a group of responses for each question:

[
y_{S,1}, y_{S,2}, ..., y_{S,G}
]

The verifier gives rewards:

[
r_i \in {0,1}
]

The batch builder should select anchors like this:

### Default behavior

```python
if include_wrong_jepa is False:
    anchors = responses where reward > 0
```

### Extended behavior

```python
if include_wrong_jepa is True:
    anchors = responses where reward > 0 or reward <= 0
```

In both cases, a response can only become a JEPA anchor if its prompt/question has a cached teacher target.

If a prompt does not have a cached teacher target, skip it for the JEPA branch.

GRPO itself remains unchanged and still uses all rollouts normally.

---

## 7. Student Input Format

For every selected anchor, use the same format:

[
[x, y_{S,i}, \texttt{[PRED]}^k]
]

where (k =) `predictor_k`.

For correct response:

[
[x, y_S^+, \texttt{[PRED]}^k]
]

For wrong response:

[
[x, y_S^-, \texttt{[PRED]}^k]
]

The embedding should be extracted from the final `[PRED]` token:

[
\hat{z}_{S,i}
=============

\text{L2Norm}
\left(
h_\theta([x,y_{S,i},\texttt{[PRED]}^k])_{\text{last [PRED]}}
\right)
]

If `predictor_k=1`, this is simply the hidden state of the single appended `[PRED]`.

---

## 8. JEPA Loss

There should be one unified JEPA loss over all selected anchors:

[
\mathcal{L}_{JEPA}
==================

\frac{1}{|\mathcal{A}|}
\sum_{i \in \mathcal{A}}
\left[
1 -
\cos
\left(
\hat{z}_{S,i},
\text{sg}(z_T^+(x_i))
\right)
\right]
]

where:

* (\mathcal{A}) = selected JEPA anchors
* if `include_wrong_jepa=False`, (\mathcal{A}) contains only correct responses
* if `include_wrong_jepa=True`, (\mathcal{A}) contains both correct and wrong responses
* (z_T^+(x_i)) = cached teacher-correct target for that question
* target is detached

No separate correct/wrong weighting should be used.

Do not implement:

[
\lambda_c \mathcal{L}*{correct}
+
\lambda_w \mathcal{L}*{wrong}
]

Instead, use one averaged loss:

[
\mathcal{L}_{JEPA}
==================

\text{mean over all selected anchors}
]

Then the global training objective uses the existing JEPA coefficient:

[
\mathcal{L}_{total}
===================

\mathcal{L}*{GRPO}
+
\lambda*{JEPA}\mathcal{L}*{JEPA}
+
\beta \mathcal{L}*{KL}
]

---

## 9. Empty Anchor Handling

If no JEPA anchors exist in the batch:

[
|\mathcal{A}| = 0
]

then:

[
\mathcal{L}_{JEPA}=0
]

This must be implemented safely to avoid NaNs.

Possible cases:

1. No correct responses and `include_wrong_jepa=False`
2. No cached teacher targets for the prompts
3. Empty batch after filtering
4. All selected responses are invalid/truncated

In all these cases, skip JEPA and continue with GRPO/KL.

---

## 10. Metrics

Even though the training objective should use one unified JEPA loss, the logs should still separate correct and wrong statistics for analysis.

Add or preserve these metrics:

```python
jepa/num_anchors_total
jepa/num_anchors_correct
jepa/num_anchors_wrong

jepa/loss_total
jepa/loss_correct_monitor
jepa/loss_wrong_monitor

jepa/cos_total
jepa/cos_correct
jepa/cos_wrong
```

Important:

* `loss_correct_monitor` and `loss_wrong_monitor` are for logging only.
* They should not be separately weighted in the training objective.
* `loss_total` is the actual optimized JEPA loss.

---

## 11. SIGReg Anti-Collapse Pool

If SIGReg is used, the SIGReg pool should contain the trainable student-side predicted embeddings that participate in the JEPA loss.

So:

[
\mathcal{P}_{SIG}
=================

{\hat{z}_{S,i}: i \in \mathcal{A}}
]

If `include_wrong_jepa=False`:

[
\mathcal{P}_{SIG}
=================

{\hat{z}_{S,i}: r_i=1}
]

If `include_wrong_jepa=True`:

[
\mathcal{P}_{SIG}
=================

{\hat{z}_{S,i}: r_i=1 \text{ or } r_i=0}
]

Do not include frozen teacher/reference targets in the SIGReg pool.

Teacher targets are detached constants and should not be regularized by SIGReg.

---

## 12. Gradient Flow

Student side:

[
\hat{z}_{S,i}
]

must receive gradients.

Target side:

[
z_T^+
]

must be detached.

Correct behavior:

```python
loss = 1 - cosine(pred_embedding, teacher_target.detach())
```

Wrong behavior:

```python
loss = 1 - cosine(pred_embedding.detach(), teacher_target)
```

or allowing gradients into the cached teacher target.

The target cache should remain frozen.

---

## 13. Backward Compatibility

The default setting must reproduce the current behavior:

```python
include_wrong_jepa = False
```

This means:

* only correct responses are selected as JEPA anchors
* wrong responses are ignored by JEPA
* GRPO still uses all rollouts
* existing runs should remain valid
* no extra correct/wrong loss weights exist

Only when:

```python
include_wrong_jepa = True
```

wrong responses are added to the JEPA anchor set.

---

## 14. Files Likely to Modify

Based on the current Claude Code inspection, the relevant files are likely:

```text
verl/experimental/jepa_grpo/config_ray.py
verl/experimental/jepa_grpo/ray_trainer.py
verl/experimental/jepa_grpo/worker.py
verl/experimental/jepa_grpo/core_algos.py
```

Likely modifications:

### `config_ray.py`

Add:

```python
include_wrong_jepa: bool = False
```

Do not add:

```python
correct_jepa_weight
wrong_jepa_weight
```

### `ray_trainer.py`

Modify the TCR batch builder.

Current logic likely selects:

```python
correct_cot = [i for i in cot_by_uid.get(u, []) if rew[i] > 0]
```

New logic should be:

```python
if cfg.include_wrong_jepa:
    selected_cot = [i for i in cot_by_uid.get(u, []) if rew[i] > 0 or rew[i] <= 0]
else:
    selected_cot = [i for i in cot_by_uid.get(u, []) if rew[i] > 0]
```

But implement carefully, avoiding invalid responses and prompts without teacher targets.

Also store anchor labels for logging:

```python
anchor_is_correct = rew[i] > 0
```

### `worker.py`

Make sure `joint_predictor_k = [cfg.predictor_k] * n_anchors`.

Do not create a different path for wrong responses.

Correct and wrong anchors both use:

```text
[question, student_response, [PRED]×k]
```

The student embedding path should remain exactly the same.

### `core_algos.py`

Modify the TCR loss to compute one unified JEPA loss over all anchors.

Also compute correct/wrong monitor losses separately if labels are available.

The actual loss should be:

```python
loss_jepa = (1.0 - cosine(pred_text, teacher_target.detach())).mean()
```

not:

```python
loss_jepa = correct_weight * loss_correct + wrong_weight * loss_wrong
```

---

## 15. Testing Plan

### Test 1: Default behavior

Run with:

```python
include_wrong_jepa = False
```

Expected:

```text
jepa/num_anchors_wrong = 0
jepa/num_anchors_correct > 0 if correct rollouts exist
```

The loss should match current correct-only behavior.

### Test 2: Wrong branch enabled

Run with:

```python
include_wrong_jepa = True
```

Expected:

```text
jepa/num_anchors_wrong > 0 when wrong rollouts exist
jepa/num_anchors_correct >= 0
jepa/num_anchors_total = correct + wrong
```

### Test 3: No correct responses

With `include_wrong_jepa=False`:

```text
loss_jepa = 0
```

With `include_wrong_jepa=True`:

```text
wrong anchors should still participate if cached teacher targets exist
```

### Test 4: No cached teacher target

Any response whose question does not have a cached teacher target should be skipped from JEPA.

### Test 5: Gradient check

Confirm:

```text
student [PRED] embedding has grad
teacher target has no grad
```

### Test 6: `[PRED]` usage

Confirm wrong anchors also use:

```text
[question, wrong_student_response, [PRED]]
```

and not:

```text
[question, wrong_student_response]
```

---

## 16. Experimental Comparison

Run these experiments:

### Baseline 1

```text
GRPO only
```

### Baseline 2

```text
GRPO + correct-only JEPA-TCR
include_wrong_jepa=False
```

### Main extension

```text
GRPO + all-anchor JEPA-TCR
include_wrong_jepa=True
```

### Optional ablation

```text
GRPO + wrong-only JEPA-TCR
```

This is only for analysis, not the main method. It can help reveal whether wrong anchors are useful or harmful.

---

## 17. Main Claim

The clean claim is:

> We extend JEPA-TCR from correct-only latent refinement to optional all-response latent prediction. Correct rollouts refine successful reasoning representations, while wrong rollouts learn to predict the teacher-correct latent target through a `[PRED]` token, without directly forcing wrong-response hidden states to look correct.

The method remains minimal because it introduces only one switch:

```python
include_wrong_jepa
```

and keeps the existing global JEPA coefficient.

---

## 18. Final Intended Behavior

When wrong JEPA is disabled:

```text
GRPO uses all responses.
JEPA uses correct responses only.
```

When wrong JEPA is enabled:

```text
GRPO uses all responses.
JEPA uses correct and wrong responses.
All JEPA anchors use [PRED].
All anchors predict the cached teacher-correct target.
There are no separate correct/wrong loss weights.
```

Final loss:

[
\boxed{
\mathcal{L}_{total}
===================

\mathcal{L}*{GRPO}
+
\lambda*{JEPA}
\cdot
\frac{1}{|\mathcal{A}|}
\sum_{i \in \mathcal{A}}
\left[
1 -
\cos
\left(
h_\theta([x,y_{S,i},\texttt{[PRED]}])*{\texttt{[PRED]}},
\text{sg}(z_T^+(x_i))
\right)
\right]
+
\beta \mathcal{L}*{KL}
}
]

where:

[
\mathcal{A}
===========

\begin{cases}
{i:r_i=1}, & \text{if include_wrong_jepa=False} \
{i:r_i=1}\cup{i:r_i=0}, & \text{if include_wrong_jepa=True}
\end{cases}
]
