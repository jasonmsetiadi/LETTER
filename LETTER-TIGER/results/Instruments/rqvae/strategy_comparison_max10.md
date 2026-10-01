# Vanilla RQ-VAE Strategy Comparison Report: LETTER-TIGER (Instruments)

- **Dataset**: `Instruments`
- **Model**: `LETTER-TIGER`
- **Tokenizer**: `rqvae`
- **Catalog Items**: 9,922
- **Total Interactions**: 206,153
- **Fixed ID Depth**: 10 tokens
- **Minimum Allowable Depth**: 1 token

### 1. Recommendation Performance

| Strategy | Mean Length | Weighted Length | Collisions | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | 10.00 | 10.00 | 0 | 5.78% | 8.21% | 10.12% | 6.99% | 7.60% | completed |
| **shortest_unique** | 2.89 | 3.06 | 25 | 5.34% | 7.49% | 9.37% | 6.40% | 7.01% | completed |
| **popularity:frequency** | 8.43 | 6.05 | 25 | 5.23% | 7.21% | 9.09% | 6.22% | 6.83% | completed |
| **popularity:user_entropy** | 6.68 | 4.48 | 25 | 5.39% | 7.41% | 9.25% | 6.40% | 6.99% | completed |
| **popularity:pagerank** | 7.99 | 5.64 | 25 | 5.23% | 7.21% | 9.14% | 6.23% | 6.85% | completed |
| **popularity:co_occurrence** | 8.16 | 5.72 | 25 | 5.41% | 7.21% | 9.13% | 6.31% | 6.93% | completed |
| **popularity:cf_density** | 6.34 | 5.88 | 25 | 4.43% | 6.01% | 7.35% | 5.23% | 5.70% | completed |
| **residual** | 9.43 | 9.46 | 25 | 5.83% | 8.28% | 10.00% | 7.06% | 7.65% | completed |

### 2. Semantic ID Evaluation & Compression

| Strategy | Catalog Mean | Traffic W-Len | Token Savings | L=1 | L=2 | L=3 | L=4 | L=5 | L=6 | L=7 | L=8 | L=9 | L=10 | Collisions | Spearman ρ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | 10.000 | 10.000 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0 | - |
| **shortest_unique** | 2.888 | 3.056 | 69.4% | 0.0% | 51.7% | 36.4% | 4.9% | 1.8% | 1.0% | 0.2% | 0.0% | 0.0% | 4.0% | 25 | - |
| **popularity:frequency** | 8.428 | 6.055 | 39.5% | 0.0% | 0.5% | 1.7% | 2.7% | 4.1% | 6.4% | 9.5% | 14.3% | 22.0% | 38.8% | 25 | 1.0000 |
| **popularity:user_entropy** | 6.682 | 4.484 | 55.2% | 0.0% | 5.6% | 11.0% | 8.7% | 9.4% | 10.2% | 11.0% | 12.0% | 13.3% | 18.8% | 25 | 1.0000 |
| **popularity:pagerank** | 7.994 | 5.637 | 43.6% | 0.0% | 0.8% | 2.9% | 3.9% | 6.1% | 8.6% | 11.6% | 15.5% | 19.9% | 30.8% | 25 | 0.9319 |
| **popularity:co_occurrence** | 8.155 | 5.718 | 42.8% | 0.0% | 1.5% | 3.4% | 3.8% | 5.3% | 6.9% | 9.2% | 12.8% | 18.8% | 38.3% | 25 | 0.8315 |
| **popularity:cf_density** | 6.339 | 5.879 | 41.2% | 0.0% | 7.9% | 13.8% | 9.7% | 9.6% | 9.7% | 9.9% | 10.2% | 11.0% | 18.2% | 25 | 0.4135 |
| **residual** | 9.430 | 9.462 | 5.4% | 0.0% | 0.0% | 0.1% | 0.4% | 1.6% | 4.0% | 5.3% | 5.1% | 3.6% | 79.9% | 25 | - |

### 3. Pairwise Exact Semantic ID Agreement Rate (%)

> Percentage of items in catalog that receive an identical Semantic ID across strategies.

| Strategy | fixed | shortest_unique | popularity:frequency | popularity:user_entropy | popularity:pagerank | popularity:co_occurrence | popularity:cf_density | residual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | 100.00% | 4.03% | 38.83% | 18.84% | 30.76% | 38.31% | 18.18% | 79.87% |
| **shortest_unique** | 4.03% | 100.00% | 6.14% | 19.86% | 7.40% | 8.45% | 24.50% | 4.76% |
| **popularity:frequency** | 38.83% | 6.14% | 100.00% | 20.94% | 52.24% | 53.43% | 16.97% | 32.77% |
| **popularity:user_entropy** | 18.84% | 19.86% | 20.94% | 100.00% | 26.83% | 24.99% | 23.93% | 17.89% |
| **popularity:pagerank** | 30.76% | 7.40% | 52.24% | 26.83% | 100.00% | 47.19% | 16.57% | 26.93% |
| **popularity:co_occurrence** | 38.31% | 8.45% | 53.43% | 24.99% | 47.19% | 100.00% | 19.37% | 32.63% |
| **popularity:cf_density** | 18.18% | 24.50% | 16.97% | 23.93% | 16.57% | 19.37% | 100.00% | 18.13% |
| **residual** | 79.87% | 4.76% | 32.77% | 17.89% | 26.93% | 32.63% | 18.13% | 100.00% |

### 4. Pairwise Mean Absolute Length Difference (Tokens)

> Average token length divergence per item (|L_A - L_B|). Lower value indicates closer length profiles.

| Strategy | fixed | shortest_unique | popularity:frequency | popularity:user_entropy | popularity:pagerank | popularity:co_occurrence | popularity:cf_density | residual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | 0.000 | 7.112 | 1.572 | 3.318 | 2.006 | 1.845 | 3.661 | 0.570 |
| **shortest_unique** | 7.112 | 0.000 | 5.541 | 3.795 | 5.107 | 5.268 | 3.452 | 6.542 |
| **popularity:frequency** | 1.572 | 5.541 | 0.000 | 1.746 | 0.511 | 0.607 | 2.634 | 1.748 |
| **popularity:user_entropy** | 3.318 | 3.795 | 1.746 | 0.000 | 1.356 | 1.636 | 2.054 | 3.167 |
| **popularity:pagerank** | 2.006 | 5.107 | 0.511 | 1.356 | 0.000 | 0.689 | 2.486 | 2.074 |
| **popularity:co_occurrence** | 1.845 | 5.268 | 0.607 | 1.636 | 0.689 | 0.000 | 2.506 | 1.972 |
| **popularity:cf_density** | 3.661 | 3.452 | 2.634 | 2.054 | 2.486 | 2.506 | 0.000 | 3.413 |
| **residual** | 0.570 | 6.542 | 1.748 | 3.167 | 2.074 | 1.972 | 3.413 | 0.000 |
