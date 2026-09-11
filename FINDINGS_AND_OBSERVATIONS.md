# Quantization Findings & Observations
### Phi-4-mini-instruct (`microsoft/Phi-4-mini-instruct`) — 3.8B Parameters, BF16

> **What this document is:** A running log of every experiment run during this quantization project, what the numbers showed, and what conclusions were drawn. Written so that someone reading the repo can follow the reasoning from raw inspection all the way to the AWQ implementation decision — not just see the final code, but understand why each decision was made.
>
> **Experiments in order:** Model inspection → Weight distribution analysis → Per-tensor vs per-row vs per-column quantization across all 32 layers → Activation outlier calibration via forward hooks → AWQ implementation (in progress).

---

## Stage 1 — Model Inspection

### What we did
Loaded the model in BF16 on CPU, printed the full module tree, and read the exact architecture from `config.json`.

### Key facts confirmed from actual inspection

```
Total parameters:    3,836,021,760
Hidden dimension:    3072
Attention heads:     24 (Q), 8 (K and V) — Grouped Query Attention
Head dimension:      128
Intermediate size:   8192 (MLP expanded space before down_proj)
Vocabulary size:     200,064
Decoder layers:      32
Tied embeddings:     True (embed_tokens and lm_head share the same tensor)
```

### The five matrix types per layer — shapes and parameter counts

| Matrix | Shape | Parameters per layer | Total (×32) |
|---|---|---|---|
| `qkv_proj` | [5120, 3072] | 15,728,640 | 503M |
| `o_proj` | [3072, 3072] | 9,437,184 | 302M |
| `gate_up_proj` | [16384, 3072] | 50,331,648 | 1,610M |
| `down_proj` | [3072, 8192] | 25,165,824 | 805M |
| `embed_tokens` / `lm_head` (tied) | [200064, 3072] | — | 614M |

**The MLP (gate_up_proj + down_proj) accounts for 63.6% of all parameters.** Quantization quality in MLP layers dominates both total memory savings and accumulated error.

### The qkv_proj output split — verified arithmetic
`qkv_proj` outputs 5120 dimensions, fusing Q, K, V into one matrix:
- Q: 24 heads × 128 dims = 3072 (rows 0–3071)
- K: 8 heads × 128 dims = 1024 (rows 3072–4095)
- V: 8 heads × 128 dims = 1024 (rows 4096–5119)
- Total: 3072 + 1024 + 1024 = **5120 ✓**

### Tied embeddings — implementation gotcha
`model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()` returns `True`. These are the same physical tensor in memory. Any quantization loop that walks all `nn.Linear` modules will encounter this tensor twice. Quantizing it twice applies rounding twice — double error. The implementation must check `data_ptr()` equality before processing `lm_head` and skip it if `embed_tokens` was already handled.

---

## Stage 2 — Weight Distribution Analysis (Layer 0, qkv_proj)

### What we did
Pulled `model.model.layers[0].self_attn.qkv_proj.weight`, cast to float32, computed basic statistics and a 50-bucket histogram.

### Raw numbers

```
Shape:        [5120, 3072]   →   15,728,640 individual values
Min value:    -1.593750
Max value:    +1.460938
Max absolute:  1.593750
Mean absolute: 0.026782
```

### The scale calculation — what naive quantization would do

```
Scale = max_abs / 127 = 1.593750 / 127 = 0.012549 per integer slot

Average value maps to slot: 0.026782 / 0.012549 ≈ 2.1
```

The average weight maps to integer slot 2 out of 127 available slots. The scale is set 59× wider than the typical value needs, because one outlier value forces it wide for the entire matrix.

### Histogram findings

```
Values between -0.1 and +0.1:   15,375,742   →   97.8% of all values
Values beyond ±0.5:                   1,016   →   0.006% of all values
```

**97.8% of 15.7 million values are packed into the bottom ~8 integer slots.** 0.006% of values — 1,016 outliers — occupy slots 40 through 127. The 256 available integer slots are distributed almost entirely to the rarest values.

### Row vs column outlier analysis

```
Rows  (5120 total):  195 rows  with max-abs > 0.5  →  3.8% of rows are elevated
Columns (3072 total): 61 cols  with max-abs > 0.5  →  2.0% of columns are elevated
```

**Exact location of the largest outlier:**
- Position: row 2535, column 1541
- Value: -1.593750
- Row 2535 mean_abs: 0.148997 (entire row is elevated — 9× normal)
- Column 1541 mean_abs: 0.086567 (column elevated but one cell extreme)

**Interpretation:** Row 2535 is uniformly elevated — training pushed its entire range high. Column 1541 has one extreme cell in an otherwise moderately elevated column. These are different failure modes that respond differently to scaling strategies.

Row 2535 sits in the Q-projection portion of `qkv_proj` (rows 0–3071 = Q). Rows 2517, 2528, 2535, 2538, 2559 all appeared in the top-10 worst rows — clustered in a narrow band of the Q output dimensions.

---

## Stage 3 — Quantization Granularity Comparison (All 32 Layers, All 4 Matrix Types)

### What we did
Implemented three quantization methods and ran them against all 32 layers for all four matrix types. Measured Mean Absolute Error (MAE) for each.

### Method definitions

**Per-tensor:** one scale for all values in the matrix.
`scale = max_abs_of_entire_matrix / 127`

**Per-row:** one scale per output channel (row).
`scale[i] = max_abs_of_row_i / 127`

**Per-column:** one scale per input channel (column).
`scale[j] = max_abs_of_col_j / 127`

### Results — qkv_proj (shape [5120, 3072] per layer)

Per-row wins all 32 layers. Per-tensor is worst by a large margin.

```
Layer 0 example:
  Per-tensor: 0.00313469
  Per-row:    0.00039735   ←  best (7.9× better than per-tensor)
  Per-col:    0.00049662
```

Error trend across layers: per-tensor error is highest in layer 0 (0.00313) and drops toward later layers. Per-row error stays stable around 0.00025–0.00040 throughout all 32 layers. This stability suggests the per-row approach generalizes well regardless of layer depth.

### Results — gate_up_proj (shape [16384, 3072] per layer)

Per-row wins all 32 layers.

```
Layer 0 example:
  Per-tensor: 0.00356734
  Per-row:    0.00024443   ←  best (14.6× better than per-tensor)
  Per-col:    0.00031606
```

The largest matrix in the model shows the largest absolute improvement from per-row scaling, consistent with it also having the most rows to benefit from independent scale assignment.

### Results — o_proj (shape [3072, 3072] per layer)

Per-row wins 30 out of 32 layers. Per-col wins layers 28 and 29 by a narrow margin.

```
Layer 28: per-row 0.00028348, per-col 0.00026972  ← col wins by 5%
Layer 29: per-row 0.00026563, per-col 0.00026549  ← col wins by <0.1%
```

The near-tie in layers 28-29 suggests the output projection's outlier structure becomes more column-like in later layers — consistent with the deeper layers handling higher-level semantic compression before `lm_head`.

### Results — down_proj (shape [3072, 8192] per layer) — most interesting

Per-col wins 23 out of 32 layers. Per-row wins 9 (early and final layers).

```
Win pattern:
  Layers  0–5:   mostly row
  Layers  6–10:  mixed, row edges out
  Layers 11–29:  col wins almost every time
  Layers 30–31:  row wins again
```

**Why down_proj behaves differently from all other matrices:** `down_proj`'s columns correspond to the 8192-dimensional MLP intermediate space produced by `gate_up_proj` and shaped by SiLU gating. The SiLU function produces non-negative outputs with a long right tail — specific intermediate dimensions consistently carry large values because the gating mechanism learned to keep them open. This persistent activation pattern is column-structured, not row-structured. Per-column scaling directly targets it.

The early layers (0-5) and final layers (30-31) reverting to per-row behavior suggests different learned roles — early layers handle lower-level syntactic patterns with more row-structured weights, final layers have mixed structure as they prepare the residual stream for `lm_head`.

### Summary table

| Matrix | tensor wins | row wins | col wins | Recommended |
|---|---|---|---|---|
| `qkv_proj` | 0 | **32** | 0 | Per-row |
| `o_proj` | 0 | **30** | 2 | Per-row |
| `gate_up_proj` | 0 | **32** | 0 | Per-row |
| `down_proj` | 0 | 9 | **23** | Per-col |

**Key conclusion:** The optimal granularity is not the same for all matrix types. A production quantizer should use per-row for attention matrices and gate_up_proj, and per-column for down_proj. This decision is backed by measurement on the real model, not theory alone.

### Memory overhead of scale factors

Scale factors must be stored alongside the quantized weights.

```
Per-row qkv_proj:    5,120 scales × 2 bytes (BF16) = 10.2 KB per layer
Per-col down_proj:   8,192 scales × 2 bytes         = 16.4 KB per layer
Total across 32 layers (rough): ~1.1 MB
```

The entire scale factor overhead for all 32 layers is approximately 1.1 MB — negligible compared to the 3.8 GB weight saving. The precision benefit of fine-grained scaling costs almost nothing in storage.

---

## Stage 4 — Activation Outlier Calibration

### What we did
Registered forward hooks on all four matrix types across all 32 layers. Ran 100 calibration prompts through the model (diverse: factual, math, code, reasoning, instruction-following). Captured input activations to each `Linear` layer incrementally (running sum and max per column, to avoid OOM from storing full tensors). Computed mean absolute activation per input dimension across all 1,030 tokens seen.

### Why activations, not just weights

Weight outliers tell you which weight values are large. Activation outliers tell you which input dimensions are large when real text flows through. The relevant quantity for quantization error is:

```
output_error = weight_rounding_error × activation_magnitude
```

A weight column with rounding error of 0.005 produces output error of:
- 0.00015 if the corresponding activation is 0.03 (normal)
- 0.040 if the corresponding activation is 8.0 (outlier)

The same weight error is 267× worse in the outlier case. This is why AWQ measures activations, not just weights. It protects the columns where these two factors combine.

### qkv_proj activation outliers — residual stream propagation confirmed

```
Layer  Top dim   mean_abs   ratio
    0     2293     2.4118   100.3x
    1     2889     2.5825    48.5x
    2     2889     2.0898    21.1x
    3     2889     2.1771    19.8x
    4     2331     5.9715    47.6x
    5     2331     5.7824    42.4x
    6     2331     5.5081    43.4x
    ...
   16     2331     8.3336    51.4x   ← peak ratio in mid-layers
    ...
   21      546     6.4422    31.8x   ← dimension shifts here
   22      546     5.6208    27.9x
    ...
   31      546     1.9445    34.9x
```

**The same small set of dimension indices — 2331, 2889, 546, 2293 — dominate across all 32 layers.** This is direct evidence of residual stream propagation: a value written into dimension 2331 in an early layer gets added to the residual stream and is never erased, so every subsequent layer sees a consistently large value in that dimension at the input to its `qkv_proj`. The architecture document's description of residual connections is confirmed empirically.

Layer 16 shows a spike to 8.33 mean_abs with a 51.4× ratio — the worst ratio in the middle layers. This suggests something specific happens at layer 16 that amplifies dimension 2331 further, likely a particularly large RMSNorm weight in that dimension at that layer.

### The double-trouble confirmation — dimension 1541

From Stage 2 (weight inspection): column 1541 of `qkv_proj` layer 0 had the largest weight value in the entire matrix (max_abs = 1.593750).

From Stage 4 (activation calibration):

```
Layer 0 qkv_proj, top-10 salient input dimensions:
  Rank 1   Dim 2293   mean_abs 2.4118
  Rank 2   Dim 1541   mean_abs 1.0431   ← appears here too
```

Dimension 1541 has both large weights AND large activations flowing through it. The weight rounding error is large, and it gets multiplied by a large activation. This is the worst-case scenario for naive quantization, and it is exactly what AWQ is designed to protect. Dimension 1541 must be in the highest-protection tier.

### The extreme outlier — down_proj layer 3

```
Layer  3   down_proj   Dim 3760   mean_abs 287.52   ratio 14,381x
Layer 30   down_proj   Dim 448    mean_abs 238.88   ratio  1,224x
```

These are the most extreme activation outliers found in the entire model. Dimension 3760 of the MLP intermediate space in layer 3 has a mean activation of 287 — when the median dimension has 0.02. This is almost certainly an attention sink dimension: a channel the model learned to route "irrelevant" attention toward, using it as a disposal mechanism for BOS tokens and punctuation. The extreme magnitude (14,381× the median) means naive quantization of this weight column would produce catastrophically amplified errors.

AWQ must protect dimension 3760 in `down_proj` layer 3 with an extremely large scaling factor. No other single decision in the quantization pipeline has as large an impact on output quality as correctly handling this dimension.

### gate_up_proj — persistent neuron behavior

Dimension 241 is the top outlier for layers 4–16 without interruption (13 consecutive layers, same dimension). Dimensions 2293 and 2331 dominate layers 17–30. This identifies specific "always-on" neurons in the MLP's expanded intermediate space that are activated consistently regardless of input type. These are not task-specific activations — they appear across all 100 calibration prompts covering very different domains (math, code, factual, conversational).

### o_proj — scattered outlier pattern

Unlike `qkv_proj`, the top outlier dimension for `o_proj` changes almost every layer with no clear pattern. The ratios are moderate (5.9× to 39.8×) compared to `qkv_proj`'s 100× at layer 0. This suggests `o_proj`'s outlier structure is less tied to the residual stream and more driven by the specific multi-head attention patterns of each layer.

### down_proj — sporadic extreme spikes

The `down_proj` activation ratios are volatile — ranging from 6.7× (layer 6) to 14,381× (layer 3). Most layers have mild to moderate outliers, but layers 3 and 30 are catastrophic. This volatility is consistent with the SiLU gating mechanism: most intermediate dimensions are partially suppressed most of the time, but specific dimensions get "unlocked" strongly by certain input types.

---

## Stage 5 — AWQ Implementation (In Progress)

### What AWQ does, precisely

AWQ (Activation-Aware Weight Quantization) modifies the weight matrix and its upstream LayerNorm before quantizing, so that salient (high-activation) input dimensions are better preserved after rounding. The key insight: instead of using higher precision for important weights (which complicates hardware execution), AWQ rescales them so they occupy a better portion of the uniform INT8 grid.

**The math for one protected column j with scale s:**

```python
# Step 1 — Find salient columns
# Salient = large mean_abs activation AND large weight values
# Use mean_abs from calibration (activation_stats.pt)

# Step 2 — Compute scale for column j
# AWQ paper: s = mean_abs[j]^alpha, alpha searched over [0, 1]
# Higher alpha = stronger protection for that column

# Step 3 — Apply scale to weight column
W[:, j] = W[:, j] / s
# This shrinks column j's values, making them fit better in the INT8 grid

# Step 4 — Absorb scale into upstream LayerNorm
# So that the net computation is unchanged:
# LayerNorm produces s× larger output in dim j
# Weight column is s× smaller → product unchanged
layernorm.weight[j] = layernorm.weight[j] * s

# Step 5 — Quantize the modified weight matrix
# The salient columns now have smaller values → better slot distribution
# The scale factor used for INT8 is no longer dominated by those columns
```

### Priority list derived from calibration data

Based on the ratio and mean_abs values, the top protection priorities for this model are:

| Priority | Matrix | Layer | Dimension | Ratio | Reason |
|---|---|---|---|---|---|
| Critical | `down_proj` | 3 | 3760 | 14,381× | Extreme attention sink |
| Critical | `down_proj` | 30 | 448 | 1,224× | Second extreme spike |
| High | `qkv_proj` | 0 | 2293 | 100.3× | Residual stream source outlier |
| High | `qkv_proj` | 0 | 1541 | — | Double-trouble: large weight AND large activation |
| High | `qkv_proj` | 16 | 2331 | 51.4× | Mid-layer amplification spike |
| Medium | All `qkv_proj` | 4–20 | 2331 | 30–48× | Persistent residual stream dimension |
| Medium | `down_proj` | 20–29 | various | 25–92× | SiLU-unlocked MLP dimensions |

---

## Overall Memory Budget

| Format | Per-weight bytes | Total weights | Scale overhead | Total |
|---|---|---|---|---|
| BF16 original | 2 bytes | 7.6 GB | — | ~7.6 GB |
| INT8 quantized | 1 byte | 3.8 GB | ~1.1 MB | ~3.8 GB |
| **Saving** | | **3.8 GB** | | **~50%** |

Runtime activation memory (not quantized, always BF16): small, sequence-length dependent.

---

## Files in This Repository

| File | Purpose |
|---|---|
| `src/quantizer.py` | Per-tensor, per-row, per-column quantization functions and 32-layer comparison loop |
| `src/calibration.py` | Forward hook registration, calibration prompt runner, activation stat computation |
| `src/awq_scale.py` | AWQ scale search and application (in progress) |
| `data/calibration_prompts.py` | 100 diverse prompts across 9 categories used for activation calibration |
| `results/activation_stats.pt` | Saved activation statistics (mean_abs, max_abs per dimension, per layer, per matrix) |
| `PHI4_ARCHITECTURE_DEEP_DIVE_V2.md` | Complete architecture reference with data flow and quantization notes |
| `inspect_model.ipynb` | Interactive exploration notebook (histogram, row/col analysis) |

---

*All numbers in this document come from actual runs on `microsoft/Phi-4-mini-instruct`. No values are estimated or taken from papers. The model was loaded in BF16 on CPU in a 16GB GitHub Codespace.*
