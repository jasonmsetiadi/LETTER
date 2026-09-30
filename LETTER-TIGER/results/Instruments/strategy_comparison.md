# LETTER Strategy Comparison Report: LETTER-TIGER (Instruments)

- **Dataset**: `Instruments`
- **Model**: `LETTER-TIGER`
- **Catalog Items**: 9,922
- **Total Interactions**: 206,153
- **Fixed ID Depth**: 4 tokens
- **Minimum Allowable Depth**: 1 token

### 1. Recommendation Performance

| Strategy | Mean Length | Weighted Length | Collisions | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | 4.00 | 4.00 | 0 | 5.96% | 8.51% | 10.59% | 7.24% | 7.91% | completed |
| **shortest_unique** | 2.57 | 2.72 | 25 | 5.66% | 7.74% | 9.53% | 6.71% | 7.28% | completed |
| **popularity:frequency** | 3.66 | 3.20 | 25 | 5.78% | 7.88% | 9.72% | 6.85% | 7.44% | completed |
| **popularity:user_entropy** | 3.24 | 2.93 | 25 | 5.76% | 7.84% | 9.62% | 6.80% | 7.37% | completed |
| **popularity:pagerank** | 3.55 | 3.11 | 25 | 5.81% | 7.90% | 9.78% | 6.87% | 7.47% | completed |
| **popularity:co_occurrence** | 3.59 | 3.16 | 25 | 5.77% | 7.83% | 9.51% | 6.83% | 7.37% | completed |
| **popularity:cf_density** | 3.15 | 3.14 | 25 | 5.91% | 8.30% | 10.16% | 7.14% | 7.74% | completed |
| **residual** | 4.00 | 4.00 | 25 | 6.12% | 8.60% | 10.67% | 7.36% | 8.03% | completed |

### 2. Semantic ID Evaluation & Compression

| Strategy | Catalog Mean | Traffic W-Len | Token Savings | L=1 | L=2 | L=3 | L=4 | Collisions | Spearman ρ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | 4.000 | 4.000 | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0 | - |
| **shortest_unique** | 2.572 | 2.724 | 31.9% | 0.2% | 48.1% | 46.1% | 5.6% | 25 | - |
| **popularity:frequency** | 3.663 | 3.197 | 20.1% | 0.0% | 3.5% | 26.6% | 69.9% | 25 | 1.0000 |
| **popularity:user_entropy** | 3.244 | 2.933 | 26.7% | 0.0% | 15.5% | 44.7% | 39.8% | 25 | 1.0000 |
| **popularity:pagerank** | 3.547 | 3.113 | 22.2% | 0.0% | 5.6% | 34.0% | 60.3% | 25 | 0.9319 |
| **popularity:co_occurrence** | 3.593 | 3.156 | 21.1% | 0.0% | 5.8% | 29.1% | 65.1% | 25 | 0.8315 |
| **popularity:cf_density** | 3.149 | 3.144 | 21.4% | 0.0% | 20.5% | 44.1% | 35.4% | 25 | 0.4135 |
| **residual** | 3.997 | 3.998 | 0.1% | 0.0% | 0.1% | 0.2% | 99.7% | 25 | - |

### 3. Pairwise Exact Semantic ID Agreement Rate (%)

> Percentage of items in catalog that receive an identical Semantic ID across strategies.

| Strategy | fixed | shortest_unique | popularity:frequency | popularity:user_entropy | popularity:pagerank | popularity:co_occurrence | popularity:cf_density | residual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | 100.00% | 5.64% | 69.85% | 39.81% | 60.34% | 65.08% | 35.44% | 99.72% |
| **shortest_unique** | 5.64% | 100.00% | 25.77% | 51.91% | 32.56% | 30.15% | 57.68% | 5.87% |
| **popularity:frequency** | 69.85% | 25.77% | 100.00% | 59.94% | 87.38% | 85.93% | 48.36% | 69.70% |
| **popularity:user_entropy** | 39.81% | 51.91% | 59.94% | 100.00% | 69.98% | 63.82% | 58.42% | 39.85% |
| **popularity:pagerank** | 60.34% | 32.56% | 87.38% | 69.98% | 100.00% | 83.11% | 50.50% | 60.27% |
| **popularity:co_occurrence** | 65.08% | 30.15% | 85.93% | 63.82% | 83.11% | 100.00% | 51.31% | 64.95% |
| **popularity:cf_density** | 35.44% | 57.68% | 48.36% | 58.42% | 50.50% | 51.31% | 100.00% | 35.53% |
| **residual** | 99.72% | 5.87% | 69.70% | 39.85% | 60.27% | 64.95% | 35.53% | 100.00% |

### 4. Pairwise Mean Absolute Length Difference (Tokens)

> Average token length divergence per item (|L_A - L_B|). Lower value indicates closer length profiles.

| Strategy | fixed | shortest_unique | popularity:frequency | popularity:user_entropy | popularity:pagerank | popularity:co_occurrence | popularity:cf_density | residual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | 0.000 | 1.428 | 0.337 | 0.756 | 0.453 | 0.407 | 0.851 | 0.003 |
| **shortest_unique** | 1.428 | 0.000 | 1.091 | 0.671 | 0.975 | 1.020 | 0.577 | 1.424 |
| **popularity:frequency** | 0.337 | 1.091 | 0.000 | 0.419 | 0.126 | 0.143 | 0.633 | 0.339 |
| **popularity:user_entropy** | 0.756 | 0.671 | 0.419 | 0.000 | 0.306 | 0.391 | 0.488 | 0.756 |
| **popularity:pagerank** | 0.453 | 0.975 | 0.126 | 0.306 | 0.000 | 0.170 | 0.594 | 0.454 |
| **popularity:co_occurrence** | 0.407 | 1.020 | 0.143 | 0.391 | 0.170 | 0.000 | 0.594 | 0.409 |
| **popularity:cf_density** | 0.851 | 0.577 | 0.633 | 0.488 | 0.594 | 0.594 | 0.000 | 0.850 |
| **residual** | 0.003 | 1.424 | 0.339 | 0.756 | 0.454 | 0.409 | 0.850 | 0.000 |
