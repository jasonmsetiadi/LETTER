# Variable-Length Semantic IDs: Research Roadmap & Architectural Phases

This document outlines the end-to-end research roadmap for **Variable-Length Semantic IDs (SIDs)** in generative recommender systems (LETTER, TIGER, LC-Rec). It details the progression from **Phase 1: Post-Hoc Truncation** (heuristic / decoupled) through **Phase 1.5: Length-Aware Training** (loss on the truncated prefix, external stop rule) to **Phase 2: Native Variable-Length Quantization** (stop decision learned / coupled), and ultimately **Phase 3: Decoder Co-Design**.

---

## 1. Vision & Architectural Evolution

Generative recommenders autoregressively generate item Semantic IDs token-by-token. Fixed-length IDs ($L=4$) force uniform decoding steps on all items, creating high inference latency and memory overhead. Variable-length SIDs compress average sequence length by allocating fewer tokens where possible.

The research evolves across four phases:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Post-Hoc Truncation (Decoupled)                                               │
│  - Train standard fixed-length RQ-VAE (L=4), loss at full depth only                   │
│  - Truncate codes post-hoc via tree sparsity, popularity quantiles, or residual bounds │
│  - Primary Goal: Benchmark empirical limits without modifying the quantizer            │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1.5: Length-Aware Training (Coupled loss, external stop rule)                    │
│  - Run one Phase 1 rule (fidelity / prefix / Huffman) INSIDE the training loop         │
│  - Recon / codebook losses computed only up to each item's stopping depth k(x)         │
│  - Stop rule is a hard threshold or precomputed label: it receives no gradient         │
│  - Primary Goal: Isolate the effect of training-time coupling with the rule held fixed │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Native Variable Quantization (Learned stop decision)                          │
│  - Stop decision is optimized by gradient against fidelity and rate                    │
│  - 4 Design Families: Matryoshka RQ-VAE, Learned Dynamic Halting (ACT),                │
│    Rate-Distortion VQ (EC-RQ-VAE), and Asymmetric Tree-Structured VQ (Tree-VQ)         │
│  - Primary Goal: Eliminate the intermediate code gap; make short prefixes high-fidelity│
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Recommender & Decoder Co-Design (End-to-End)                                  │
│  - Length-calibrated beam search and adaptive Trie constraints                         │
│  - Early-exit decoding with confidence thresholds in TIGER / LC-Rec                    │
│  - Primary Goal: Joint optimization of token decoding latency and recommendation NDCG  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Phase Definition Test

Two questions separate the phases:

- **(a)** Is the quantizer loss computed on the truncated prefix (not only at full depth)?
- **(b)** Is the stop decision itself optimized by gradient (against reconstruction and a rate term)?

| Phase | (a) Loss on truncated prefix | (b) Stop decision trained |
| :--- | :---: | :---: |
| Phase 1 | No | No |
| Phase 1.5 | Yes | No |
| Phase 2 | Yes | Yes |

Phase 3 is orthogonal: it leaves the quantizer unchanged and modifies decoding.


---

## 2. Phase 1: Post-Hoc Truncation (Decoupled Approach)

### Architectural Paradigm
In Phase 1, the quantizer and the truncation algorithm are completely decoupled:
1. **Tokenizer Training**: Train a standard RQ-VAE with $L$ codebook layers on item content embeddings $x \in \mathbb{R}^d$ and collaborative embeddings $e_{\text{CF}}$.
2. **Index Generation**: Quantize all items to fixed-length IDs: $\text{SID}_{\text{fixed}}(i) = (c_1, c_2, \dots, c_L)$.
3. **Post-Hoc Truncation**: Apply a pruning strategy to select an item-specific length $l_i \le L$.

```
                              Phase 1 Architecture
   [Raw Item Embedding x] ──> [Frozen RQ-VAE (L=4)] ──> [Fixed Index: (c1, c2, c3, c4)]
                                                                   │
                                                                   ▼
                                                       [Post-Hoc Truncation Filter]
                                                        ├── Naive Shortest Unique Prefix
                                                        ├── Huffman / Popularity-Tiered
                                                        └── Residual-Fidelity Gated
                                                                   │
                                                                   ▼
                                                       [Variable Index: (c1, ..., cl)]
```

### Strategies Explored in Phase 1
- **Naive Shortest Unique Prefix (SUP)**: Cuts at shallowest prefix where catalog count is 1.
- **Huffman-Inspired (Popularity-Tiered)**: Uses interaction distribution $P(i)$ to protect the long tail at $L_{\max}=4$ while compressing head items to $L_{\min}=2$.
- **Fidelity-Based (Residual-Gated)**: Truncates at layer $l$ only if cumulative distortion $\mathcal{E}(l) \le \tau$.

### Fundamental Theoretical Limitations of Phase 1
While Phase 1 provides immediate sequence compression without retraining the quantizer, it suffers from two inherent architectural mismatches:

#### Limitation 1: Intermediate Code Neglect (Decoupled Objective)
Standard RQ-VAE optimizes the reconstruction loss exclusively at the final layer:
$$\mathcal{L}_{\text{recon}} = \left\| x - \text{Decoder}\left(\sum_{l=1}^4 z_l\right) \right\|_2^2$$
Intermediate sums $\hat{x}_1 = z_1$ and $\hat{x}_2 = z_1 + z_2$ are merely partial residuals; the decoder is never trained to decode $\hat{x}_1$ or $\hat{x}_2$ independently. Truncating to 2 tokens forces downstream models to rely on representations that were never optimized to be self-sufficient.

#### Limitation 2: Collaborative Alignment Detachment
LETTER's core strength is its collaborative alignment loss ($\mathcal{L}_{\text{CF}}$), which contrasts quantized embeddings against CF vectors (e.g. SASRec):
$$\mathcal{L}_{\text{CF}} = -\log \frac{\exp(\langle \hat{x}_4, e_{\text{CF}} \rangle / \tau)}{\sum_j \exp(\langle \hat{x}_4, e'_j \rangle / \tau)}$$
Notice that $\mathcal{L}_{\text{CF}}$ is evaluated **only on the 4-layer representation $\hat{x}_4$**. 
When an item is truncated to length 2 ($c_1, c_2$), **its collaborative alignment is discarded**, depriving the downstream recommender of critical collaborative filtering signals.

### Status in the Literature
No surveyed variable-length SID paper is pure Phase 1 (fixed-length training followed by post-hoc truncation), and none reports a post-hoc truncation control at matched average length. This baseline is therefore missing from the literature and is what Milestone 1 provides. Each Phase 1 strategy (residual-fidelity, shortest unique prefix, Huffman) reappears in Phase 1.5 as an in-training rule.

---

## 2.5 Phase 1.5: Length-Aware Training (Coupled Loss, External Stop Rule)

*Phase 1.5 is a category introduced in this roadmap, not a term from the surveyed papers.*

### Architectural Paradigm
Phase 1.5 sits between Phase 1 and Phase 2. The quantizer is trained with a loss that depends on each item's length, but the length itself is still decided by a fixed rule:
1. **Stop rule**: One of Phase 1's three rules (residual fidelity, prefix/subtree size, popularity) determines a stopping depth $k(x)$ for each item, as a hard threshold or a precomputed label.
2. **Length-conditioned loss**: Reconstruction (and codebook/commitment) terms are summed only up to $k(x)$:
$$\mathcal{L}_{\text{recon}}^{(k(x))} = \left\| z - \sum_{l=1}^{k(x)} q_l \right\|_2^2$$
3. **No gradient to the stop decision**: Nothing trains the threshold or the labels. If the rule stops too early, only the codebooks can compensate.

### Signal Taxonomy
| Signal | Question it asks | Stop rule | Literature example |
| :--- | :--- | :--- | :--- |
| **Fidelity** | Does the code explain the item? | Residual norm $\le \epsilon$ or routing confidence $\ge \tau$ | CapsID |
| **Prefix** | Is the item already identified? | Items under the current prefix $\le \tau$ (relaxed shortest unique prefix) | SA2CRQ |
| **Huffman** | How often is the item needed? | Popularity-rank quantile; a classifier may learn to imitate the labels | VarLenRec |

The surveyed papers also disagree on direction. VarLenRec gives popular items short IDs. SA2CRQ stops sparse paths early, and ADA-SID gives popular items more slots. Direction should be an explicit ablation.

### Limitations of Phase 1.5
1. **Untrained stop decision**: The stop signal gets no reconstruction gradient. A badly set threshold cannot be corrected by training.
2. **Single-depth supervision**: Each item is supervised at one stopping depth, not at all prefixes as in Matryoshka (Family A) or dVAE.
3. **Collaborative alignment**: $\mathcal{L}_{\text{CF}}$ is absent at truncated depths in all surveyed papers unless explicitly added at $k(x)$.
4. **Partial conditioning**: Some designs condition only part of the loss on length (e.g., VarLenRec supervises reconstruction at the target length but sums the quantization loss over all $K$ layers).

### Verification Flags
The following classifications follow each paper's equations and have not been checked against code:
- **CapsID**: Phase 1.5 only if training runs early-stopped forward passes. If training runs all $L_{\max}$ layers and stops only when emitting IDs, it is Phase 1.
- **SA2CRQ**: Phase 1.5 only if the path statistics are computed online. If they come from a fixed-depth pass, the stop rule is effectively Phase 1. The paper's head-item stage also mostly reaches full depth, and its stage-2 anchor codebooks are frozen, so the amount of truncation-aware training is unverified.
- **VarLenRec**: It is unclear whether reconstruction uses the target length $\hat{L}$ or the predictor's output.

### Literature Summary
| Paper | Phase | Signal | Loss on truncated prefix | Stop decision trained |
| :--- | :--- | :--- | :--- | :--- |
| CapsID | 1.5 (unverified) | Fidelity | Yes (inferred) | No, hard threshold |
| SA2CRQ | 1.5 (unverified) | Prefix | Yes, per equations | No, hard threshold |
| VarLenRec | 1.5 | Huffman | Reconstruction yes, quantization loss no | Classifier imitates fixed labels |
| dVAE | 2 | Fidelity (rate-distortion) | All prefixes, weighted by $q(L)$ | Yes |
| ADA-SID | Out of scope | Popularity proxy | Not prefix-based | Gated sparsity |

---

## 3. Phase 2: Native Variable-Length Quantization (The Quantizer Design Space)

In Phase 2, the length mechanism is part of the model: the stop decision is optimized by gradient against reconstruction fidelity and a rate (length) term, and the quantizer is supervised at every prefix it may be truncated to. Rather than relying on a single concept, Phase 2 encompasses **four distinct architectural families**:

> **Note on Family A.** Matryoshka supervision satisfies test (a) at all depths, but on its own it does not train a stop decision (the comparison table lists "multi-depth supervision + threshold"). Family A is therefore Phase 2 on the supervision axis and Phase 1.5 on the stop axis unless it is paired with a learned stop mechanism (Family B or C), or with a threshold calibrated on the multi-depth losses.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        PHASE 2 QUANTIZER DESIGN SPACE                                  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Family A: Matryoshka / Multi-Scale RQ-VAE (Nested Supervision)                         │
│           - Standard cascade, but supervises all prefix depths l in {1..L} simultaneously│
│                                                                                        │
│ Family B: Learned Dynamic Halting / ACT-RQ-VAE (Self-Terminating Quantizer)            │
│           - Gated halting mechanism decides item length end-to-end via rate penalty     │
│                                                                                        │
│ Family C: Rate-Distortion & Entropy-Constrained Quantization (EC-RQ-VAE)               │
│           - Information-theoretic Lagrangian: Loss = Distortion + lambda * CodeLength  │
│                                                                                        │
│ Family D: Asymmetric Tree-Structured VQ (Tree-VQ)                                      │
│           - Non-uniform codebook tree where leaf nodes naturally terminate at depths 1..L│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Family A: Matryoshka RQ-VAE (Multi-Depth Supervision)
*(Inspired by Matryoshka Representation Learning, NeurIPS 2022)*

```
                  Phase 2: Matryoshka RQ-VAE (M-RQ-VAE)
                               x (Item Embedding)
                                       │
                                   [Encoder]
                                       │
                                       v
                                   [RQ Layers]
                     Level 1            Level 2            Level 4
                    (z1 only)          (z1 + z2)     (z1 + z2 + z3 + z4)
                        │                  │                  │
                    [Decoder]          [Decoder]          [Decoder]
                        │                  │                  │
                    ▼   ▼   ▼          ▼   ▼   ▼          ▼   ▼   ▼
                  L_recon(1)         L_recon(2)         L_recon(4)
                  + L_CF(1)          + L_CF(2)          + L_CF(4)
```

#### Variant A.1: Deterministic Multi-Depth Joint Loss (Base M-RQ-VAE)
*(Inspired by Matryoshka Representation Learning, NeurIPS 2022)*

Instead of computing losses at depth $L$ alone, the total training objective is a weighted multi-depth sum across all prefix depths $l \in \{1, \dots, L\}$:

$$\mathcal{L}_{\text{total}} = \sum_{l=1}^L w_l \cdot \Big[ \underbrace{\|x - \text{Decoder}(\hat{x}_l)\|_2^2}_{\text{Multi-Depth Reconstruction Loss}} + \alpha \cdot \underbrace{\mathcal{L}_{\text{CF}}(\hat{x}_l, e_{\text{CF}})}_{\text{Multi-Depth Collaborative Loss}} \Big] + \beta \mathcal{L}_{\text{commit}}$$

where $\hat{x}_l = \sum_{j=1}^l z_j$, and $w_l$ are depth-importance weights (e.g. $w_l = 1.0$ or exponentially decaying).

- **Theoretical Advantage**: Layer 1 captures coarse semantic categories, Layer 2 captures sub-categories, and Layers 3–4 capture fine residuals. Crucially, $\mathcal{L}_{\text{CF}}(\hat{x}_l)$ is applied at **every depth** so an item truncated to length 2 retains direct collaborative alignment.

---

#### Variant A.2: Stochastic Prefix-Dropout (Truncation-Aware Training)
During forward training passes, the quantizer stochastically samples a dynamic quantization depth $l \sim \mathcal{P}(\{1, \dots, L\})$ per batch:
- With probability $p_1$: stop after 1 layer.
- With probability $p_2$: stop after 2 layers.
- With probability $p_L$: proceed to all $L$ layers.

The decoder reconstructs $x$ using only the sampled prefix $\hat{x}_l$.

- **Theoretical Advantage**: Prevents deeper codebook layers from acting as an "information crutch" for shallow layers, regularizing the codebook so any prefix $c_{1:l}$ is robust on its own.

---

#### Variant A.3: Frequency-Gated Quantization Loss
To bridge Huffman coding directly into the quantizer loss, the loss at depth $l$ for item $i$ is weighted by empirical frequency $f_i$:

$$\mathcal{L}(i) = \sum_{l=1}^L \gamma_l(f_i) \cdot \Big( \|x_i - \text{Decoder}(\hat{x}_l)\|^2 + \alpha \mathcal{L}_{\text{CF}}(\hat{x}_l, e_{\text{CF}}^{(i)}) \Big)$$

- **For popular items**: $\gamma_1(f_i)$ and $\gamma_2(f_i)$ are scaled up, forcing low reconstruction and high CF alignment for head items at shallow layers.
- **For rare items**: $\gamma_4(f_i)$ is prioritized, ensuring full 4-layer convergence for the long tail.
*(Note: This variant also serves as the conceptual bridge to Family C: Rate-Distortion Quantization).*

---

### Family B: Learned Dynamic Halting (ACT-RQ-VAE / Ponder-Quantizer)
*(Inspired by Adaptive Computation Time and PonderNet)*

Rather than assigning lengths by static heuristics (popularity quantiles or arbitrary error thresholds), the quantizer **learns when to stop quantizing autonomously**.

```
                   Family B: Dynamic Halting Architecture
                        x (Item Embedding) ──> [Encoder]
                                                   │
  ┌────────────────────────────────────────────────┴───────────────────────────────┐
  │                                                                                │
  ▼                                                                                ▼
[Layer 1: z1] ──> [Halting Unit h1] ──(h1 >= 1-eps)──> EMIT <eos_id> (Length = 1)
       │                  │
       ▼ (1 - h1)         ▼
[Layer 2: z2] ──> [Halting Unit h2] ──(h1+h2 >= 1-eps)─> EMIT <eos_id> (Length = 2)
       │                  │
       ▼ (1 - h1 - h2)    ▼
[Layer 3: z3] ──> [Halting Unit h3] ──...
```

- **Mechanism**: After each quantizer layer $l$, a lightweight halting gate outputs a scalar halting probability $h_l \in [0, 1]$:
  $$h_l = \sigma\left(W_h [z_l \,\|\, r_l] + b_h\right)$$
  where $r_l = x - \hat{x}_{l-1}$ is the current residual.
- **Objective with Parsimony Penalty**:
  $$\mathcal{L} = \sum_{l=1}^L p_l \Big( \|x - \text{Decoder}(\hat{x}_l)\|^2 + \alpha \mathcal{L}_{\text{CF}}(\hat{x}_l) \Big) + \lambda_{\text{rate}} \cdot \mathbb{E}[N_{\text{steps}}(x)]$$
  where $p_l$ is the halting distribution and $\mathbb{E}[N_{\text{steps}}]$ is the expected token length.
- **Key Advantage**: The network naturally learns the optimal rate-distortion trade-off: semantically distinct/simple items stop after 1 or 2 tokens, while complex/densely packed items continue to 4 tokens.
- **Reference implementation (dVAE, Khrylchenko 2026)**: The closest existing instance of Families B and C. A per-step stop probability induces $q(L)$, reconstruction is weighted over all prefixes by $q(L=l)$, and the rate term is $\lambda\,\mathbb{E}[L] - H(q(L))$, all differentiable with no REINFORCE. It uses a shared vocabulary across positions and a causal transformer decoder, so it is **not** a drop-in replacement for standard RQ-VAE. It has no $\mathcal{L}_{\text{CF}}$ at any depth.

---

### Family C: Rate-Distortion & Entropy-Constrained VQ (EC-RQ-VAE)
*(Derived from Information-Theoretic Entropy-Constrained Vector Quantization, Chou et al.)*

In classical source coding, the optimal code length of an item $i$ is proportional to its self-information $-\log P(i)$. EC-RQ-VAE embeds this principle directly into the continuous optimization objective:

$$\min_{\theta, \mathcal{C}} \mathbb{E}_{x \sim \mathcal{D}} \Big[ \underbrace{\mathcal{D}(x, \hat{x}_{l(x)})}_{\text{Distortion: Reconstruction + CF}} + \lambda \cdot \underbrace{\mathcal{R}(l(x), f_x)}_{\text{Rate Penalty: Expected Decoding Cost}} \Big]$$

- **Rate Formulations**:
  1. *Interaction-Weighted Token Cost*: $\mathcal{R} = l(x) \cdot \frac{f_x}{\sum_j f_j}$ (penalizes long IDs heavily if the item is frequently accessed by users).
  2. *Codebook Shannon Entropy*: $\mathcal{R} = -\sum_{c \in \mathcal{C}} P(c) \log P(c)$ (encourages high-probability semantic clusters to be shallow).
- **Key Advantage**: Directly unifies Huffman coding theory and neural vector quantization into a single Lagrangian multiplier objective.
- **Open gap**: No surveyed paper has tried an item-weighted rate term $\lambda \cdot f_x \cdot \mathbb{E}[L]$ inside a learned fidelity-based halting rule. dVAE uses a uniform $\lambda$ and VarLenRec uses popularity labels without any learned halting. Combining the two is a candidate contribution.

---

### Family D: Asymmetric Tree-Structured VQ (Tree-VQ)

Standard RQ-VAE uses a rigid Cartesian product of $L$ flat codebooks: $\mathcal{C}_1 \times \mathcal{C}_2 \times \dots \times \mathcal{C}_L$ (a uniform tree where every branch has depth $L$).

Tree-VQ replaces this with an **inherently asymmetric hierarchical tree codebook**:

```
                        Family D: Tree-VQ Codebook
                                   Root
                                  /    \
                              Node A   Node B
                              /   \       \
                           A1     A2    (Item 5: Leaf, Depth 2)
                          /  \     \
                      Item 1 Item 2 (Item 3: Leaf, Depth 3)
                     (Depth 4)
```

- **Mechanism**:
  - Codebook nodes at depth 2 or 3 can be designated as **terminal leaves** (representing a finished item ID).
  - Dense semantic regions branch 4 levels deep to prevent collisions.
  - Distinct or isolated semantic regions terminate at depth 1 or 2.
- **Key Advantage**: The tree topology itself dictates item length. There are no redundant codebook parameters, and every path from root to leaf is guaranteed to be unique and valid.

---

### Comparison of Phase 2 Architectural Families

| Dimension | Family A: Matryoshka (M-RQ-VAE) | Family B: Dynamic Halting (ACT) | Family C: Rate-Distortion (EC-RQ) | Family D: Tree-Structured VQ |
| :--- | :--- | :--- | :--- | :--- |
| **Length Decision** | Multi-depth supervision + threshold | Learned gating unit $h_l$ | Lagrangian rate penalty $\lambda \mathcal{R}$ | Tree topology (leaf depth) |
| **CF Loss ($\mathcal{L}_{\text{CF}}$)** | Explicit at all depths | Weighted by halting prob $p_l$ | Scaled by rate term | Evaluated at leaf nodes |
| **Downstream Compatibility** | Drop-in replacement for standard RQ-VAE | Replaces the RQ-VAE quantizer; halting gate and rate term are added (the dVAE reference also changes the decoder and token embeddings) | Depends on the rate formulation | Requires tree-routing encoder |
| **Primary Strength** | Simple, highly stable training | End-to-end learned lengths | Strong information-theory basis | Zero collision redundancy |

### Out of Scope: Parallel-Slot Designs
Designs without residual levels, such as ADA-SID (parallel shared/text/vision/behavior experts with separate codebooks and a ReLU-gated behavioral router), have no prefix structure, so none of Phases 1 to 3 applies cleanly. Dropped slots become padding tokens, so the sequence may not shorten. Its popularity-gated alignment weight is still relevant to Variant A.3 (frequency-gated loss).

---

## 4. Phase 3: Recommender & Decoder Co-Design

Phase 3 addresses how downstream sequence recommenders (LETTER-TIGER and LETTER-LC-Rec) decode, search, and score variable-length IDs at inference time.

### Why Standard Decoders Fail on Variable-Length IDs
In fixed-length recommendation ($L=4$), every item requires exactly 4 autoregressive steps under a Constrained Prefix Tree (Trie). When items have variable lengths (e.g., 2, 3, or 4 tokens), standard decoders encounter three structural pathologies:
1. **Mathematical Length Bias**: Shorter IDs accumulate fewer negative log-probabilities, unfairly dominating beam search.
2. **Wasted Decoding Steps**: The model continues generating tokens even after a prefix uniquely identifies a single item in the catalog.
3. **Static Rigidity**: Item length is fixed offline, ignoring whether the user's immediate recommendation context is obvious (high certainty) or ambiguous (needs fine-grained disambiguation).

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Phase 3 Decoder Architecture                     │
├────────────────────────────────────────────────────────────────────────┤
│  Autoregressive Step t                                                 │
│       │                                                                │
│       ▼                                                                │
│  [Logits / Next-Token Distribution P(c_t | c_<t)]                      │
│       │                                                                │
│       ├── 1. Length-Calibrated Beam Scoring (Normalized Log-Likelihood)│
│       ├── 2. Trie Leaf Early Exit (Subtree Size |S(c_1:t)| == 1)       │
│       ├── 3. Dynamic Confidence Exit (Entropy < eps & Top1 > theta)    │
│       └── 4. Ragged KV-Cache Management (Early slot release)           │
└────────────────────────────────────────────────────────────────────────┘
```

---

### 1. Length-Calibrated Beam Search (Eliminating Length Bias)
In autoregressive generation, sequence score is the cumulative sum of token log-probabilities:
$$\text{Score}(s) = \sum_{t=1}^{|s|} \log P(c_t \mid c_{<t}, \text{user history})$$

Because each $\log P \le 0$ (a negative number), every additional decoding step adds a penalty:
- A **2-token item** only accumulates 2 penalties (e.g., $-0.5 + -0.6 = \mathbf{-1.1}$).
- A **4-token item** accumulates 4 penalties (e.g., $-0.5 + -0.6 + -0.4 + -0.3 = \mathbf{-1.8}$).

Without correction, beam search exhibits extreme length bias toward shorter items, degrading recommendation precision. Phase 3 implements polynomial length normalization:

$$\text{Score}_{\text{norm}}(s) = \frac{\sum_{t=1}^{|s|} \log P(c_t \mid c_{<t}, \text{history})}{\left( \frac{5 + |s|}{6} \right)^\alpha}$$

where $\alpha \in [0.6, 1.0]$ is a length penalty hyperparameter ensuring 2-, 3-, and 4-token items compete strictly on semantic relevance.

---

### 2. Trie Leaf Early-Exit (Adaptive Prefix Stopping)
During constrained beam search over the catalog Trie, many prefixes become unique before reaching depth $L_{\max}$:

```
                     Root
                    /    \
                 c1=10   c1=42
                /          \
             c2=5         c2=15  <── [Subtree size |S| = 1: Uniquely Item 777]
            /    \          |
         c3=1    c3=8     (Item 777)
          |        |
       (Item 1) (Item 2)
```

At node $(c_1=42, c_2=15)$, only **Item 777** exists in the entire catalog. Generating $c_3$ and $c_4$ is computationally redundant.
- **Phase 3 Mechanism**: The Trie constraint mask evaluates subtree cardinality $|\mathcal{S}(c_{1:t})|$.
- When $|\mathcal{S}(c_{1:t})| = 1$, the search terminates immediately, emits an implicit `<eos>`, and returns the resolved item.

---

### 3. Context-Aware Dynamic Lengths (Confidence-Gated Decoding)
In Phases 1 and 2, an item's length is **static** (determined offline). In Phase 3, decoding depth adapts **dynamically to user context**:

- **Clear Intent (High Certainty)**: When a user's recent history strongly signals a specific item, the model's top-1 token probability is high ($\max_c P(c) > \theta$) and prediction entropy is near zero. The decoder halts at layer 2.
- **Ambiguous Intent (Exploratory / Broad)**: When user history is diverse, entropy is high. The decoder continues to layers 3 and 4 to resolve fine-grained attribute differences.

$$\text{Dynamic Exit at Step } t \iff \max_{c} P(c \mid c_{<t}, \text{history}) > \theta_{\text{confidence}} \quad \text{AND} \quad \mathcal{H}(P) < \epsilon$$

---

### 4. Serving & Hardware Optimization (KV Cache & Ragged Batching)
In generative recommenders (especially LLM-based backbones like LC-Rec / LLaMA), inference latency is heavily memory-bandwidth bound due to reading Key-Value (KV) caches at every decoding step.

Co-designing the decoder with variable-length IDs yields:
- **Early KV-Cache Deallocation**: Completed beams release GPU memory immediately without waiting for longer sequences in the batch.
- **Ragged / Paged Attention Batching**: Eliminates padding overhead in mixed-length batches, translating sequence length reductions directly into **$1.5\times$ – $1.8\times$ wall-clock throughput speedups**.

---

## 5. Comparative Roadmap Matrix

| Dimension | Phase 1: Post-Hoc Truncation | Phase 2: Native Quantization (M-RQ-VAE) | Phase 3: Decoder Co-Design |
| :--- | :--- | :--- | :--- |
| **Quantizer Type** | Standard RQ-VAE (frozen) | Matryoshka RQ-VAE (M-RQ-VAE) | M-RQ-VAE + Adaptive Trie |
| **Intermediate Representation Quality** | Sub-optimal (not trained to stand alone) | **Optimal** (explicitly trained at all depths) | Optimal |
| **Collaborative Alignment at $l < 4$** | **Lost** (CF loss applied only at $L=4$) | **Preserved** (multi-depth CF loss) | Preserved |
| **Sequence Length Savings** | 25% – 35% | 25% – 40% | **35% – 50%** |
| **Inference Latency Speedup** | Up to 1.35x | Up to 1.45x | **Up to 1.80x** |
| **Recommendation NDCG** | Small drop vs fixed (0.5% – 1.0%) | **Matches or beats fixed baseline** | Matches or beats fixed baseline |
| **Implementation Complexity** | Low (script-level post-processing) | Medium (modified loss & training loop) | Medium-High (decoding modifications) |

---

## 6. Actionable Research Milestones

```
  [Milestone 1: Establish Phase 1 Baseline]  <── (CURRENT STATUS)
   ├── Evaluate Shortest Unique Prefix (SUP) on Instruments & Games
   ├── Evaluate Huffman / Popularity-Tiered Strategy
   └── Evaluate Residual-Fidelity Strategy
         │
         ▼
  [Milestone 2: Design & Implement Matryoshka RQ-VAE (Phase 2)]
   ├── Modify models/rqvae.py to output layer-wise reconstructions
   ├── Implement multi-depth CF contrastive loss: sum_l Loss_CF(x_q_l)
   └── Train M-RQ-VAE and verify layer 1 & layer 2 reconstruction error
         │
         ▼
  [Milestone 3: Benchmark Phase 1 vs. Phase 2]
   ├── Compare Hit@K and NDCG@K of Phase 1 vs. Phase 2 at equal token lengths
   └── Verify whether M-RQ-VAE eliminates the 1.0% accuracy drop of short IDs
         │
         ▼
  [Milestone 4: Paper / Publication Deliverable]
   └── "Matryoshka Semantic IDs: Native Variable-Length Quantization for Generative Recommendation"
```