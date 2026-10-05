# LETTER

This is the pytorch implementation of our paper:

> [Learnable Item Tokenization for Generative Recommendation](https://arxiv.org/abs/2405.07314)

## Overview
We propose LETTER (a LEarnable Tokenizer for generaTivE Recommendation), which integrates hierarchical semantics, collaborative signals, and code assignment diversity to satisfy the essential requirements of identifiers. 
LETTER incorporates Residual Quantized VAE for semantic regularization, a contrastive alignment loss for collaborative regularization, and a diversity loss to mitigate code assignment bias. We instantiate LETTER on two generative recommender models and propose a ranking-guided generation loss to augment their ranking ability theoretically. 

![image.png](https://s2.loli.net/2024/05/12/PveBMV23SRa1lrJ.png)

## Requirements

```
torch==1.13.1+cu117
accelerate
bitsandbytes
deepspeed
evaluate
k-means-constrained
peft
sentencepiece
tqdm
transformers
```

## LETTER Tokenizer

### Preprocess item embeddings

The RQ-VAE tokenizer consumes an embedding row for each processed item. Generate
the required `<dataset>.emb-<model>-td.npy` file from the dataset's
`<dataset>.item.json` metadata with a local Hugging Face model checkpoint:

```
bash preprocess_item_embeddings.sh \
  --dataset Instruments \
  --gpu-id 0
```

The default model is `bert-base-uncased`, and the script writes
`data/Instruments/Instruments.emb-bert-base-uncased-td.npy`. It refuses to replace an
existing output unless `--overwrite` is supplied, and accepts `--data-root`,
`--plm-name`, `--plm-checkpoint`, `--max-sent-len`, and `--python` overrides.
`--plm-checkpoint` may be either an existing local model directory or a valid
Hugging Face model ID.

### Train

```
bash RQ-VAE/train_tokenizer.sh 
```

### Tokenize

```
bash RQ-VAE/tokenize.sh 
```

### Variable-length semantic IDs

The downstream recommenders accept index files where each item has between one
and the configured maximum number of consecutive residual-code tokens (for
example, `["<a_12>", "<b_7>"]` or
`["<a_12>", "<b_7>", "<c_3>", "<d_9>"]`). Generate a reproducible,
collision-free variable-length index from a fixed index with:

```
python RQ-VAE/truncate_indices.py \
  --input data/Instruments/Instruments.index.json \
  --output data/Instruments/Instruments.varlen.index.json \
  --min-length 1 \
  --max-length 4
```

This utility uses the Shortest Unique Prefix (SUP) method, which finds the minimal
prefix length per item that uniquely distinguishes it in the catalog Trie without adding
new collisions. Inherent base collisions present in the input index at the maximum
length are preserved rather than causing failure (use `--strict` to disallow).

Train and evaluate with the resulting index file. LETTER-TIGER constrains
generation with a catalog trie; LETTER-LC-Rec does the same after the response
marker, allowing EOS after every complete item ID. Pass the generated suffix
to the downstream scripts, for example `--index_file .varlen.index.json`.

## Instantiation

### Unified end-to-end pipeline

After generating item embeddings, run the complete end-to-end pipeline using `run_pipeline.sh`:

```bash
# Fixed-length pipeline (default mode: fixed, default tokenizer: rqvae)
bash run_pipeline.sh --dataset Instruments

# Variable-length pipeline (mode: varlen, default tokenizer: rqvae)
bash run_pipeline.sh --dataset Instruments --mode varlen

# Run with LETTER tokenizer (collaborative alignment and diversity regularization)
bash run_pipeline.sh --dataset Instruments --tokenizer letter
```

The runner stores tokenizer checkpoints in `checkpoint/<dataset>/<tokenizer>/`,
index files in `data/<dataset>/<tokenizer>/` (with fallback to `data/<dataset>/`
for existing legacy indices), model checkpoints in `ckpt/<dataset>/<tokenizer>/`, and
evaluation metrics in `results/<dataset>/<tokenizer>/`.

To train and evaluate both TIGER and LC-Rec, provide a local LLaMA checkpoint and GPUs:

```bash
bash run_pipeline.sh \
  --dataset Instruments \
  --models tiger,lcrec \
  --base-model /absolute/path/to/llama
```

#### Tokenizer Selection
Use `--tokenizer rqvae` (default) for standard Vanilla RQ-VAE (semantic reconstruction
only, no collaborative embeddings required). Use `--tokenizer letter` to run the
LETTER tokenizer with collaborative alignment loss and diversity regularization.

#### Variable-Length Truncation
When running with `--mode varlen`, item IDs are truncated from an intermediate
fixed index using collision-free prefix pruning:
- `--strategy`: `shortest_unique` (default), `popularity`, `collaborative`, or `residual`.
- `--min-length` / `--max-length`: ID length bounds (defaults: 1 and 4).
- `--collab-signal`: `frequency` (default), `user_entropy`, `pagerank`, `co_occurrence`, or `cf_density`.

By default, `run_pipeline.sh` runs all stages from scratch. Use `--resume` to check and reuse existing checkpoints/indices at each stage, and `--tokenizer-only` to stop after index generation.

### LETTER-TIGER

```
cd LETTER-TIGER
bash run_train.sh
```

### LETTER-LC-Rec

```
cd LETTER-LC-Rec
bash run_train.sh
```

## Citation
If you find our work is useful for your research, please consider citing: 
```
@inproceedings{wang2024learnableitemtokenizationgenerative,
  title = {Learnable Item Tokenization for Generative Recommendation},
  author = {Wang, Wenjie and Bao, Honghui and Lin, Xinyu and Zhang, Jizhi and Li, Yongqi and Feng, Fuli and Ng, See-Kiong and Chua, Tat-Seng},
  booktitle = {International Conference on Information and Knowledge Management},
  year = {2024}
}
```
