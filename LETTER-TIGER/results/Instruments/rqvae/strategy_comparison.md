# Vanilla RQ-VAE Strategy Comparison Report: LETTER-TIGER (Instruments)

- **Dataset**: `Instruments`
- **Model**: `LETTER-TIGER`
- **Tokenizer**: `rqvae`
- **Catalog Items**: 9,922
- **Total Interactions**: 206,153
- **Fixed ID Depth**: 4 tokens
- **Minimum Allowable Depth**: 1 token

### 1. Recommendation Performance

| Strategy | Mean Length | Weighted Length | Collisions | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | 4.00 | 4.00 | 0 | 5.85% | 8.33% | 10.38% | 7.09% | 7.75% | completed |
| **shortest_unique** | 2.80 | 2.87 | 27 | 5.66% | 7.74% | 9.53% | 6.71% | 7.28% | completed |
| **popularity:frequency** | 3.69 | 3.25 | 27 | 5.78% | 7.88% | 9.72% | 6.85% | 7.44% | completed |
| **popularity:user_entropy** | 3.33 | 3.03 | 27 | 5.78% | 7.96% | 9.75% | 6.88% | 7.46% | completed |
| **popularity:pagerank** | 3.59 | 3.18 | 27 | 5.81% | 7.90% | 9.78% | 6.87% | 7.47% | completed |
| **popularity:co_occurrence** | 3.63 | 3.22 | 27 | 5.83% | 7.98% | 9.59% | 6.92% | 7.43% | completed |
| **popularity:cf_density** | 3.26 | 3.24 | 27 | 6.02% | 8.43% | 10.36% | 7.24% | 7.86% | completed |
| **residual** | 4.00 | 4.00 | 27 | 6.12% | 8.60% | 10.67% | 7.36% | 8.03% | completed |

### 2. Semantic ID Evaluation & Compression

| Strategy | Catalog Mean | Traffic W-Len | Token Savings | L=1 | L=2 | L=3 | L=4 | Collisions | Spearman ρ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | 4.000 | 4.000 | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0 | - |
| **shortest_unique** | 2.798 | 2.869 | 28.3% | 0.0% | 32.8% | 54.5% | 12.6% | 27 | - |
| **popularity:frequency** | 3.691 | 3.254 | 18.7% | 0.0% | 2.8% | 25.3% | 71.9% | 27 | 1.0000 |
| **popularity:user_entropy** | 3.327 | 3.031 | 24.2% | 0.0% | 11.6% | 44.0% | 44.3% | 27 | 1.0000 |
| **popularity:pagerank** | 3.587 | 3.181 | 20.5% | 0.0% | 4.4% | 32.6% | 63.1% | 27 | 0.9319 |
| **popularity:co_occurrence** | 3.630 | 3.221 | 19.5% | 0.0% | 4.4% | 28.2% | 67.4% | 27 | 0.8315 |
| **popularity:cf_density** | 3.265 | 3.235 | 19.1% | 0.0% | 13.5% | 46.5% | 40.0% | 27 | 0.4135 |
| **residual** | 3.999 | 3.999 | 0.0% | 0.0% | 0.0% | 0.1% | 99.9% | 27 | - |

### 3. Pairwise Exact Semantic ID Agreement Rate (%)

> Percentage of items in catalog that receive an identical Semantic ID across strategies.

| Strategy | fixed | shortest_unique | popularity:frequency | popularity:user_entropy | popularity:pagerank | popularity:co_occurrence | popularity:cf_density | residual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | 100.00% | 12.65% | 71.93% | 44.35% | 63.08% | 67.39% | 39.99% | 99.88% |
| **shortest_unique** | 12.65% | 100.00% | 33.23% | 59.10% | 40.46% | 37.86% | 64.66% | 12.75% |
| **popularity:frequency** | 71.93% | 33.23% | 100.00% | 64.93% | 88.64% | 87.51% | 53.90% | 71.85% |
| **popularity:user_entropy** | 44.35% | 59.10% | 64.93% | 100.00% | 74.29% | 67.76% | 64.10% | 44.36% |
| **popularity:pagerank** | 63.08% | 40.46% | 88.64% | 74.29% | 100.00% | 84.88% | 56.44% | 63.06% |
| **popularity:co_occurrence** | 67.39% | 37.86% | 87.51% | 67.76% | 84.88% | 100.00% | 56.64% | 67.33% |
| **popularity:cf_density** | 39.99% | 64.66% | 53.90% | 64.10% | 56.44% | 56.64% | 100.00% | 40.04% |
| **residual** | 99.88% | 12.75% | 71.85% | 44.36% | 63.06% | 67.33% | 40.04% | 100.00% |

### 4. Pairwise Mean Absolute Length Difference (Tokens)

> Average token length divergence per item (|L_A - L_B|). Lower value indicates closer length profiles.

| Strategy | fixed | shortest_unique | popularity:frequency | popularity:user_entropy | popularity:pagerank | popularity:co_occurrence | popularity:cf_density | residual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | 0.000 | 1.202 | 0.309 | 0.673 | 0.413 | 0.370 | 0.735 | 0.001 |
| **shortest_unique** | 1.202 | 0.000 | 0.893 | 0.529 | 0.789 | 0.832 | 0.466 | 1.200 |
| **popularity:frequency** | 0.309 | 0.893 | 0.000 | 0.364 | 0.114 | 0.126 | 0.532 | 0.310 |
| **popularity:user_entropy** | 0.673 | 0.529 | 0.364 | 0.000 | 0.262 | 0.342 | 0.408 | 0.673 |
| **popularity:pagerank** | 0.413 | 0.789 | 0.114 | 0.262 | 0.000 | 0.152 | 0.497 | 0.413 |
| **popularity:co_occurrence** | 0.370 | 0.832 | 0.126 | 0.342 | 0.152 | 0.000 | 0.499 | 0.371 |
| **popularity:cf_density** | 0.735 | 0.466 | 0.532 | 0.408 | 0.497 | 0.499 | 0.000 | 0.735 |
| **residual** | 0.001 | 1.200 | 0.310 | 0.673 | 0.413 | 0.371 | 0.735 | 0.000 |
