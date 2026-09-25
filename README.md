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
bash data_process/preprocess_item_embeddings.sh \
  --dataset Instruments \
  --gpu-id 0
```

The default model is `google/flan-t5-xl`, loaded through its encoder-only
`T5EncoderModel`, and the script writes
`data/Instruments/Instruments.emb-flan-t5-xl-td.npy`. It refuses to replace an
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

This utility requires the input IDs to be unique at the selected maximum
length; regenerate or otherwise resolve fixed-ID collisions before truncating.

Train and evaluate with the resulting index file. LETTER-TIGER constrains
generation with a catalog trie; LETTER-LC-Rec does the same after the response
marker, allowing EOS after every complete item ID. Pass the generated suffix
to the downstream scripts, for example `--index_file .varlen.index.json`.

## Instantiation

### Fixed-length end-to-end pipeline

After generating item embeddings, run the tokenizer, index generation, and
TIGER pipeline with:

```
bash run_fixed_length_pipeline.sh --dataset Instruments
```

The runner creates `data/Instruments/Instruments.index.fixed.json` instead of
overwriting the bundled index. It defaults to TIGER on two GPUs. To also train
and evaluate LC-Rec, provide a local LLaMA checkpoint and four GPUs:

```
bash run_fixed_length_pipeline.sh \
  --dataset Instruments \
  --models tiger,lcrec \
  --base-model /absolute/path/to/llama
```

Use `--rqvae-checkpoint PATH` to skip RQ-VAE training, `--models tiger` or
`--models lcrec` to select a downstream model, and `--skip-evaluation` to omit
the final test stage. It uses the paper-script defaults `--alpha 0.01` and
`--beta 0.0001`; override them when needed. Existing generated indexes are
protected unless `--overwrite-index` is supplied.

### Variable-length end-to-end pipeline

The variable-length runner first creates a fixed index and then truncates each
item's code sequence to a collision-free length between one and four tokens:

```
bash run_variable_length_pipeline.sh --dataset Instruments
```

It writes `data/Instruments/Instruments.index.varlen.json` and trains TIGER
with it. Use `--models tiger,lcrec --base-model /absolute/path/to/llama` to run
both downstream models. The fixed index produced for truncation must contain
no duplicate complete IDs; the runner stops with an error if that condition is
not met instead of creating an ambiguous variable-length mapping.

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
