# How TabPFN’s transformer differs from language-model transformers

TabPFN uses a transformer backbone, but the “sequence” and the “tokens” are **tabular**
objects (rows/features), not text tokens.

## What is a “token” in TabPFN?

**Language models**
- Tokenize text into a 1D sequence of token IDs → embedding lookup table.

**TabPFN**
- Operates on tensors shaped like **(rows, batch, features)**.
- You can see this immediately in `PerFeatureTransformer.forward`:
  - `seq_len, batch_size, num_features = x["main"].shape` in `src/tabpfn/architectures/base/transformer.py:328`.
- Here, `seq_len` is the number of **rows** (train + test), and `num_features` is the number of **columns** (often grouped into “feature blocks” internally).

## Context → query instead of next-token prediction

**Language models**
- Usually use causal self-attention: predict the next token, mask “future” tokens.

**TabPFN**
- Uses a train/test split inside the sequence:
  - `single_eval_pos = y["main"].shape[0]` in `src/tabpfn/architectures/base/transformer.py:351`.
- Rows `[:single_eval_pos]` act as **context** (training rows with labels), and rows `[single_eval_pos:]` are the **query** rows (test rows to predict).
- There’s no “text-style” causal mask; instead the model knows where the split is and the code can cache / reuse the training representation.

## Two-axis attention: rows and features

Most LM transformer blocks have one self-attention that attends over the 1D token sequence.

TabPFN’s base block (`PerFeatureEncoderLayer`) has (optionally):
- **attention between feature blocks** (columns/groups)
- **attention between items** (rows)

You can see these being constructed here:
- `self.self_attn_between_features` and `self.self_attn_between_items` in `src/tabpfn/architectures/base/layer.py:171` and `src/tabpfn/architectures/base/layer.py:188`.

This is one of the key differences: TabPFN explicitly models interactions across rows and across columns.

## Where the transformer lives in this repo

- Main transformer module: `class PerFeatureTransformer` in `src/tabpfn/architectures/base/transformer.py:92`
  - `__init__`: `src/tabpfn/architectures/base/transformer.py:104`
  - `forward`: `src/tabpfn/architectures/base/transformer.py:302`
- Transformer blocks:
  - `layer_creator = lambda: PerFeatureEncoderLayer(...)` in `src/tabpfn/architectures/base/transformer.py:197`
  - `self.transformer_encoder = LayerStack.of_repeated_layer(...)` in `src/tabpfn/architectures/base/transformer.py:210`

## How TabPFN encoders differ from LM embeddings

**Language models**
- “Encoder” typically means token embeddings + positional embeddings (token IDs → vectors).

**TabPFN**
- “Encoders” are small modules that convert numeric tensors into the transformer’s embedding space.
- The base architecture uses a `SequentialEncoder`, which is a pipeline of steps:
  - `class SequentialEncoder` in `src/tabpfn/architectures/base/encoders.py:314`.

### X encoder (features)

Default initialization uses:
- `LinearInputEncoderStep(num_features=1, emsize=...)` in `src/tabpfn/architectures/base/transformer.py:158`.

Important: `num_features=1` here is **not** “number of dataframe columns”. It means “one scalar channel per feature-slot” at that point in the model. TabPFN handles variable column counts by reshaping/grouping and by attention structure—not by resizing this linear layer per dataset.

### Y encoder (targets)

TabPFN also explicitly encodes training labels into the context:
- `NanHandlingEncoderStep()` + `LinearInputEncoderStep(num_features=2, ...)` in `src/tabpfn/architectures/base/transformer.py:169`.

The building blocks live here:
- `class LinearInputEncoderStep` in `src/tabpfn/architectures/base/encoders.py:477`
- `class NanHandlingEncoderStep` in `src/tabpfn/architectures/base/encoders.py:612`

## Positional information: columns vs word positions

**Language models**
- Use positional encodings to represent *token order* in a 1D sequence.

**TabPFN**
- Uses **feature positional embeddings** so the model can distinguish columns:
  - configured in `src/tabpfn/architectures/base/transformer.py:254`
  - applied in forward around `src/tabpfn/architectures/base/transformer.py:614`
  - note: `"subspace"` mode uses a fixed `COL_EMBEDDING` loaded in `src/tabpfn/architectures/base/transformer.py:38`

## Where does the “+1 feature” come from?

If you start with a dataset that has `F` raw features (e.g., California Housing has 8),
you may see `F' = F + 1` inside the model. A common reason is the **fingerprint feature**,
which is enabled by default:

- Config: `InferenceConfig.FINGERPRINT_FEATURE` defaults to `True` in `src/tabpfn/inference_config.py:101`.
- Pipeline wiring: if enabled, TabPFN adds the step in `src/tabpfn/preprocessing.py:719`.
- Implementation: the step appends one extra feature column in `src/tabpfn/preprocessors/add_fingerprint_features_step.py:78`.

The fingerprint feature is a hash-based synthetic column intended to help the model
distinguish identical/duplicate rows without depending on row order.
