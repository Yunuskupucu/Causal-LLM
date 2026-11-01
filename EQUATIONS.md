# Denklem Haritası (Rapor ↔ Kod)

Kılavuz 4.3-3 gereği her kod bloğunun rapordaki hangi denkleme karşılık
geldiği aşağıda listelenmiştir. Kod yorumlarındaki "Rapor Denk. (k)"
etiketleri bu tabloya göredir.

| No. | Denklem | Kod konumu |
|----|---------|------------|
| (1) | Zincir kuralı: p(x_{1:T}) = ∏_t p(x_t \| x_{<t}) | `exp05_chain_rule.py` (paralel = sıralı testi) |
| (2) | Kayıp: L(θ) = −(1/N) Σ_t log p_θ(x_{t+1} \| x_{≤t}) | `src/model.py::sequence_nll` |
| (3) | PPL = exp(L), bpc = L / ln 2 | `exp02_train.py`, `exp05_chain_rule.py` |
| (4) | Gömme: h⁰_t = E[x_t] + P[t] | `src/model.py::MiniGPT.forward` |
| (5) | Q = XW_Q, K = XW_K, V = XW_V | `src/model.py::CausalSelfAttention.__init__` |
| (6) | S = QKᵀ / √d_k | `CausalSelfAttention.forward` (`scores_raw`) |
| (7) | Maske: M_ij = 0 (j ≤ i), −∞ (j > i) | `src/model.py::causal_mask` |
| (8) | A = softmax(S + M), out = AV | `CausalSelfAttention.forward` (`attn`) |
| (9) | MHA: concat(head_1..h) W_O | `CausalSelfAttention.forward` (`w_o`) |
| (10) | Ön-LN artık blok | `src/model.py::Block` |
| (11) | Logit: z = h^L W_Eᵀ (bağlı ağırlık) | `MiniGPT.lm_head` |
| (12) | Sıcaklık: p_i(T) = exp(z_i/T) / Σ_j exp(z_j/T) | `src/sampling.py::softmax_temperature` |
| (13) | ∂p_i/∂z_j = (1/T) p_i (δ_ij − p_j) | `sampling.py::softmax_jacobian_analytic`, `exp04` |
| (14) | ∂L/∂z = (1/T)(p_T − y) | `exp04_softmax_temperature.py::part_b` |
| (15) | Entropi: H(p) = −Σ p_i log p_i | `src/sampling.py::entropy` |
| (16) | dH/dT = Var_{p_T}(z) / T³ | `sampling.py::entropy_temperature_derivative_analytic`, `exp04` |
| (17) | Top-k: V_k = en büyük k; yeniden normalize | `src/sampling.py::top_k_filter` |
| (18) | Top-p: en küçük V_p, Σ_{V_p} p ≥ p | `src/sampling.py::top_p_filter` |
| (19) | Nedensellik: ∂z_i/∂e_j = 0 (j > i) | `exp01_mask_verification.py::jacobian_matrix` |
| (20) | n-gram add-α: (c(ctx,x)+α)/(c(ctx)+α\|V\|) | `src/ngram.py::NGramLM.log_prob` |
