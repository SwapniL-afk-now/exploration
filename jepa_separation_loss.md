# jepa-separation-loss — total objective

$$
L_{\text{total}} = L_{\text{GRPO}}(\text{CoT} + \text{Code}) \;+\; \alpha \, L_{\text{sep}}
$$

$$
L_{\text{sep}} = (1 - \lambda)\,\big(L_{\text{align}} + w_{\text{sep}} \, L_{\text{s}}\big) \;+\; \lambda \, L_{\text{SIGReg}}
$$

where

$$
L_{\text{align}} = \frac{1}{B}\sum_{i=1}^{B} \Big(1 - \big\langle \operatorname{Pred}(\operatorname{Enc}(\text{CoT}_i)),\; \operatorname{Enc}(\text{Code}^{c}_i) \big\rangle \Big)
$$

$$
L_{\text{s}} = \frac{1}{T}\sum_{i=1}^{T} \operatorname{relu}\!\Big( m_{\text{sep}} - \big(1 - \langle e^{c}_i,\; e^{w}_i \rangle\big) \Big)
$$

**Symbols:** $\langle\cdot,\cdot\rangle$ cosine similarity (L2-normalized embeddings); $e^{c}, e^{w}$ correct/wrong code embeddings; $B$ pairs, $T \le B$ triplet-eligible; $L_{\text{SIGReg}}$ isotropy (anti-collapse) regularizer over the pool $[\,p^{c}, e^{c}, e^{w}\,]$.

**This run:** $\alpha = 0.005,\; \lambda = 0.5,\; w_{\text{sep}} = 1.0,\; m_{\text{sep}} = 0.1$.
