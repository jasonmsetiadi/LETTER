# Vanilla RQ-VAE Strategy Comparison Report: LETTER-TIGER (Instruments)

- **Dataset**: `Instruments`
- **Model**: `LETTER-TIGER`
- **Tokenizer**: `rqvae`
- **Training Phase**: `Phase 1 & Phase 1.5 (All)`
- **Fixed ID Depth**: 10 tokens
- **Minimum Allowable Depth**: 1 token

### 1. Recommendation Performance

| Strategy | Phase | Mean Length | Weighted Length | Collisions | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed_L4** | - | 4.00 | 4.00 | - | - | - | - | - | - | pending |
| **fixed** | - | 10.00 | 10.00 | - | - | - | - | - | - | pending |
| **shortest_unique** | 1 | - | - | - | - | - | - | - | - | pending |
| **popularity** | 1 | - | - | - | - | - | - | - | - | pending |
| **residual** | 1 | - | - | - | - | - | - | - | - | pending |

### 2. Semantic ID Evaluation & Compression

| Strategy | Phase | Catalog Mean | Traffic W-Len | Token Savings | L=1 | L=2 | L=3 | L=4 | L=5 | L=6 | L=7 | L=8 | L=9 | L=10 | Collisions | Spearman ρ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed_L4** | - | 4.000 | 4.000 | 60.0% | - | - | - | - | - | - | - | - | - | - | - | - |
| **fixed** | - | 10.000 | 10.000 | 0.0% | - | - | - | - | - | - | - | - | - | - | - | - |
| **shortest_unique** | 1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| **popularity** | 1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| **residual** | 1 | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |

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
| **shortest_unique** | 1 | - | - | - | - | - | - | - | - | - | (pending) |
| **popularity** | 1 | - | - | - | - | - | - | - | - | - | (pending) |
| **residual** | 1 | - | - | - | - | - | - | - | - | - | (pending) |
