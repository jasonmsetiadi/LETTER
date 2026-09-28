### LETTER Strategy Comparison: LETTER-TIGER (Instruments)

| Strategy | Mean Length | Weighted Length | Collisions | Hit@1 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | 4.00 | 4.00 | 0 | 5.96% | 8.51% | 10.59% | 7.24% | 7.91% | completed |
| **shortest_unique** | 3.01 | 2.72 | 25 | 5.66% | 7.74% | 9.53% | 6.71% | 7.28% | completed |
| **popularity:frequency** | 3.29 | 3.23 | 25 | 5.51% | 7.72% | 9.48% | 6.63% | 7.19% | completed |
| **popularity:user_entropy** | 3.29 | 3.23 | 25 | 5.51% | 7.72% | 9.48% | 6.63% | 7.19% | completed |
| **popularity:pagerank** | 3.29 | 3.24 | 25 | 5.83% | 7.98% | 9.83% | 6.92% | 7.51% | completed |
| **popularity:target** | 3.28 | 3.27 | 25 | 5.83% | 7.72% | 9.33% | 6.77% | 7.29% | completed |
| **popularity:composite** | 3.29 | 3.23 | 25 | 5.51% | 7.72% | 9.48% | 6.63% | 7.19% | completed |
| **popularity:cf_density** | 3.28 | 3.33 | 25 | 5.71% | 7.99% | 10.00% | 6.86% | 7.51% | completed |
| **residual** | 4.00 | 4.00 | 25 | 6.12% | 8.60% | 10.67% | 7.36% | 8.03% | completed |

### Pairwise Exact Semantic ID Agreement Rate (%)

> Percentage of items in catalog that receive an identical Semantic ID across strategies.

| Strategy | fixed | shortest_unique | popularity:frequency | popularity:user_entropy | popularity:pagerank | popularity:target | popularity:composite | residual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | 100.00% | 5.64% | 29.81% | 29.81% | 29.90% | 29.44% | 29.81% | 99.72% |
| **shortest_unique** | 5.64% | 100.00% | 62.85% | 62.85% | 62.75% | 63.60% | 62.85% | 5.87% |
| **popularity:frequency** | 29.81% | 62.85% | 100.00% | 100.00% | 82.96% | 62.93% | 100.00% | 29.88% |
| **popularity:user_entropy** | 29.81% | 62.85% | 100.00% | 100.00% | 82.96% | 62.93% | 100.00% | 29.88% |
| **popularity:pagerank** | 29.90% | 62.75% | 82.96% | 82.96% | 100.00% | 64.07% | 82.96% | 30.00% |
| **popularity:target** | 29.44% | 63.60% | 62.93% | 62.93% | 64.07% | 100.00% | 62.93% | 29.58% |
| **popularity:composite** | 29.81% | 62.85% | 100.00% | 100.00% | 82.96% | 62.93% | 100.00% | 29.88% |
| **residual** | 99.72% | 5.87% | 29.88% | 29.88% | 30.00% | 29.58% | 29.88% | 100.00% |

### Pairwise Mean Absolute Length Difference (Tokens)

> Average token length divergence per item (|L_A - L_B|). Lower value indicates closer length profiles.

| Strategy | fixed | shortest_unique | popularity:frequency | popularity:user_entropy | popularity:pagerank | popularity:target | popularity:composite | residual |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fixed** | 0.000 | 1.428 | 0.921 | 0.921 | 0.921 | 0.940 | 0.921 | 0.003 |
| **shortest_unique** | 1.428 | 0.000 | 0.507 | 0.507 | 0.507 | 0.488 | 0.507 | 1.424 |
| **popularity:frequency** | 0.921 | 0.507 | 0.000 | 0.000 | 0.173 | 0.433 | 0.000 | 0.920 |
| **popularity:user_entropy** | 0.921 | 0.507 | 0.000 | 0.000 | 0.173 | 0.433 | 0.000 | 0.920 |
| **popularity:pagerank** | 0.921 | 0.507 | 0.173 | 0.173 | 0.000 | 0.417 | 0.173 | 0.920 |
| **popularity:target** | 0.940 | 0.488 | 0.433 | 0.433 | 0.417 | 0.000 | 0.433 | 0.938 |
| **popularity:composite** | 0.921 | 0.507 | 0.000 | 0.000 | 0.173 | 0.433 | 0.000 | 0.920 |
| **residual** | 0.003 | 1.424 | 0.920 | 0.920 | 0.920 | 0.938 | 0.920 | 0.000 |
