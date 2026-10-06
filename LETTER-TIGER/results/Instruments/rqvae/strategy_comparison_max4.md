# Vanilla RQ-VAE Strategy Comparison Report: LETTER-TIGER (Instruments)

- **Dataset**: `Instruments`
- **Model**: `LETTER-TIGER`
- **Tokenizer**: `rqvae`
- **Training Phase**: `Phase 1 & Phase 1.5 (All)`
- **Catalog Items**: 9,922
- **Total Interactions**: 206,153
- **Fixed ID Depth**: 4 tokens
- **Minimum Allowable Depth**: 1 token

### 1. Recommendation Performance

| Strategy | Phase | Mean Length | Weighted Length | Collisions | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | - | 4.00 | 4.00 | 25 | - | - | - | - | - | pending |
| **fixed_L10** | - | 10.00 | 10.00 | - | - | - | - | - | - | pending |
| **shortest_unique** | 1 | 2.57 | 2.72 | 25 | - | - | - | - | - | pending |
| **popularity** | 1 | 3.66 | 3.20 | 25 | - | - | - | - | - | pending |
| **residual** | 1 | - | - | - | - | - | - | - | - | pending |

### 2. Semantic ID Evaluation & Compression

| Strategy | Phase | Catalog Mean | Traffic W-Len | Token Savings | L=1 | L=2 | L=3 | L=4 | Collisions | Spearman ρ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | - | 4.000 | 4.000 | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 25 | - |
| **fixed_L10** | - | 10.000 | 10.000 | -150.0% | - | - | - | - | - | - |
| **shortest_unique** | 1 | 2.572 | 2.724 | 31.9% | 0.2% | 48.1% | 46.1% | 5.6% | 25 | - |
| **popularity** | 1 | 3.663 | 3.197 | 20.1% | 0.0% | 3.5% | 26.6% | 69.9% | 25 | 1.0000 |
| **residual** | 1 | - | - | - | - | - | - | - | - | - |

### 3. Rate-Distortion & Pareto Frontier Analysis

> Benchmarking variable-length token efficiency against fixed-length baselines of equivalent token budgets.
> - **Floor Baseline ($L_{\text{floor}}$)**: Highest evaluated fixed depth $\le$ Mean Length.
> - **Ceiling Baseline ($L_{\text{ceil}}$)**: Lowest evaluated fixed depth $\ge$ Mean Length.
> - **Interpolated ($L_{\text{interp}}$)**: Expected fixed metric at matched average token length: $M_{\text{interp}} = (1-\alpha) M_{\text{floor}} + \alpha M_{\text{ceil}}$.
> - **Pareto Classification**:
>   - **★ Dominant**: Variable model achieves equal or higher recommendation quality using strictly fewer tokens than $L_{\text{ceil}}$ ($M \ge M_{\text{ceil}}$ with $\bar{L} \le L_{\text{ceil}}$).
>   - **▲ Efficient**: Variable model outperforms the interpolated fixed baseline at matched token budget ($\Delta_{\text{interp}} > 0$).
>   - **≈ Parity**: Within 2% of matched-budget baseline.
>   - **▼ Trade-off**: Standard compression trade-off.

#### Variable-Length Rate-Distortion Frontier & Pareto Evaluation

| Strategy | Phase | Mean Length | Token Savings | Hit@10 | NDCG@10 | Floor L | Ceil L | Interp NDCG@10 | Δ vs Interp | Δ vs L_max | Pareto Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **shortest_unique** | 1 | 2.57 | 31.9% | - | - | - | - | - | - | - | (pending) |
| **popularity** | 1 | 3.66 | 20.1% | - | - | - | - | - | - | - | (pending) |
| **residual** | 1 | - | - | - | - | - | - | - | - | - | (pending) |

### 4. Pairwise Exact Semantic ID Agreement Rate (%)

> Percentage of items in catalog that receive an identical Semantic ID across strategies.

| Strategy | fixed | shortest_unique | popularity |
| :--- | :---: | :---: | :---: |
| **fixed** | 100.00% | 5.64% | 69.85% |
| **shortest_unique** | 5.64% | 100.00% | 25.77% |
| **popularity** | 69.85% | 25.77% | 100.00% |

### 5. Pairwise Mean Absolute Length Difference (Tokens)

> Average token length divergence per item (|L_A - L_B|). Lower value indicates closer length profiles.

| Strategy | fixed | shortest_unique | popularity |
| :--- | :---: | :---: | :---: |
| **fixed** | 0.000 | 1.428 | 0.337 |
| **shortest_unique** | 1.428 | 0.000 | 1.091 |
| **popularity** | 0.337 | 1.091 | 0.000 |
