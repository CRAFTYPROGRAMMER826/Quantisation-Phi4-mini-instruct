# Quantization Granularity: Per-Tensor vs Per-Row vs Per-Column

> **What this document answers:** How do we actually round a matrix of numbers into INT8, at the code level and the math level? Why would you choose one scale for a whole matrix versus one scale per row versus one scale per column? What does the PyTorch syntax in this repo's `quantizer.py` actually do, line by line?
>
> **Assumed background:** None. If you know what a matrix is (a grid of numbers with rows and columns), you can follow this document.

---

## Part 1: What "Granularity" Means

Every quantization method needs a **scale factor** — a number that translates between the original float values and the compressed INT8 integers. The question this document answers is: *how many scale factors should one matrix have?*

- **Per-tensor:** the whole matrix shares **one single scale factor**.
- **Per-row:** each row of the matrix gets **its own scale factor**.
- **Per-column:** each column of the matrix gets **its own scale factor**.

More scale factors means more precision (each group of numbers gets a scale tailored to its own range) but also more memory overhead (you have to store all those extra scale numbers) and more implementation complexity. This document walks through all three with a worked example small enough to compute by hand, then explains why one of them tends to matter most for real transformer weight matrices.

---

## Part 2: The Core Formula (Same for All Three)

Regardless of granularity, the underlying math for turning a group of float numbers into INT8 is always the same three-step recipe:

```
Step 1 — SCALE:      scale = max(abs(values in the group)) / 127

Step 2 — QUANTIZE:   integer_value = round(original_value / scale)
                      then clamp integer_value to the range [-127, 127]

Step 3 — DEQUANTIZE: reconstructed_value = integer_value * scale
```

**Why divide by 127, not 128?** INT8 can technically represent -128 to 127 (256 total values), but we deliberately only use -127 to 127 to keep the range **symmetric** around zero. This matters because our weights are centered around zero (roughly as many negative as positive values), and a symmetric range means zero always maps exactly to integer 0 with no bias in either direction.

**What "the group" means changes based on granularity:**
- Per-tensor: the group is *the entire matrix* — one scale for all values.
- Per-row: the group is *one row* — every row computes its own scale independently.
- Per-column: the group is *one column* — every column computes its own scale independently.

---

## Part 3: Worked Example by Hand

Let's use a tiny 3×4 matrix (3 rows, 4 columns) so every number can be traced by hand. This is not a real weight matrix — it's small on purpose, to make the arithmetic followable.

```
        col0    col1    col2    col3
row0 [  0.02,   0.03,   0.01,   1.20 ]
row1 [  0.04,   0.02,   0.03,   0.90 ]
row2 [  0.90,   0.03,   0.02,   0.01 ]
```

Notice the design: column 3 has two large values (1.20, 0.90) and one small one. Row 2 has one large value (0.90) mixed with small ones. This mimics, in miniature, what you found in your real `qkv_proj` matrix — some rows and some columns are "worse" than others.

### 3a. Per-Tensor Quantization (one scale for everything)

```
max(abs(all 12 values)) = 1.20

scale = 1.20 / 127 = 0.009449

Every single value in the matrix is divided by this same 0.009449:

row0: [0.02/0.009449, 0.03/0.009449, 0.01/0.009449, 1.20/0.009449]
    = [2.12 → 2,      3.18 → 3,      1.06 → 1,      127.0 → 127]

row1: [4.23 → 4,       2.12 → 2,      3.18 → 3,      95.3 → 95]

row2: [95.3 → 95,      3.18 → 3,      2.12 → 2,      1.06 → 1]
```

**What went wrong here:** look at row0, col0. The original value was 0.02. After quantizing and dequantizing: `2 * 0.009449 = 0.0189`. Error = `|0.02 - 0.0189| = 0.0011`. That looks small in isolation — but relative to the original value of 0.02, that's a **5.5% error**. Compare that to col3, row0: original 1.20, reconstructed `127 * 0.009449 = 1.1999...` — nearly perfect, because that value is close to what set the scale in the first place.

**The pattern:** values close to the matrix-wide maximum reconstruct almost perfectly. Values far below it (like the many 0.01–0.04 values) lose a large fraction of their precision, because they only occupy the bottom 1–4 integer slots out of 127 available.

### 3b. Per-Row Quantization (each row gets its own scale)

```
Row 0: max(abs(0.02, 0.03, 0.01, 1.20)) = 1.20 → scale_row0 = 1.20/127 = 0.009449
Row 1: max(abs(0.04, 0.02, 0.03, 0.90)) = 0.90 → scale_row1 = 0.90/127 = 0.007087
Row 2: max(abs(0.90, 0.03, 0.02, 0.01)) = 0.90 → scale_row2 = 0.90/127 = 0.007087

Row 0 quantized: [2, 3, 1, 127]              (same as before — row 0's own max was already 1.20)
Row 1 quantized: [0.04/0.007087, 0.02/0.007087, 0.03/0.007087, 0.90/0.007087]
               = [5.6 → 6,        2.8 → 3,       4.2 → 4,       127.0 → 127]
Row 2 quantized: [127.0 → 127,    4.2 → 4,       2.8 → 3,       1.4 → 1]
```

**What improved:** row1's value 0.04 now maps to integer slot 6, instead of slot 4 under per-tensor scaling. It has more integer slots available to represent its own (smaller) range, because it's no longer sharing a scale with row0's much larger 1.20 value. Row 1 and row 2 each got a tighter, more appropriate scale for their own actual range.

**What did NOT improve:** row0 still has the same problem it had before — its own small values (0.02, 0.03, 0.01) are still being crushed by its own large value (1.20), because per-row scaling doesn't help when the outlier and the normal values are *in the same row*.

### 3c. Per-Column Quantization (each column gets its own scale)

```
Col 0: max(abs(0.02, 0.04, 0.90)) = 0.90 → scale_col0 = 0.007087
Col 1: max(abs(0.03, 0.02, 0.03)) = 0.03 → scale_col1 = 0.000236
Col 2: max(abs(0.01, 0.03, 0.02)) = 0.03 → scale_col2 = 0.000236
Col 3: max(abs(1.20, 0.90, 0.01)) = 1.20 → scale_col3 = 0.009449
```

Look at column 1: its own maximum is only 0.03 — much smaller than the matrix-wide maximum of 1.20. Because it gets its *own* scale, its values (0.03, 0.02, 0.03) now map to slots very close to the full 127 range:

```
Col 1 quantized: [0.03/0.000236, 0.02/0.000236, 0.03/0.000236]
               = [127.0 → 127,    84.7 → 85,     127.0 → 127]
```

Compare that to how column 1's values were treated under per-tensor scaling — they mapped to slots 3, 2, 3 out of 127. Now they map to slots 127, 85, 127. **Dramatically more precision for that column**, because column 1 never contained an outlier and no longer has to share a scale with columns that do.

---

## Part 4: Why Column-Wise Matters More for Transformer Weight Matrices Specifically

This is the part that connects back to the actual model, not just the toy example.

Recall from the architecture document: a `Linear` layer computes `output = input @ W.T`. Each **row** of `W` corresponds to one *output* feature. Each **column** of `W` corresponds to one *input* feature — meaning, one specific dimension of the 3,072-dimensional residual stream.

Here is the key fact discovered by the outlier-feature research this project is built on: **outlier behavior in transformer activations is concentrated in specific input dimensions, consistently, across almost every token.** If dimension 1541 of the residual stream tends to carry unusually large values (as you found in the real `qkv_proj` matrix — column 1541 was one of the worst), that is not a coincidence of this one weight matrix. That same dimension 1541 will tend to be large in the activations feeding into *every* layer, because of the residual stream mechanism (see the architecture README, Part 2): a value written into a dimension early on propagates forward through every subsequent layer's addition.

This means outlier columns are not independent, random noise scattered across each matrix — they are a **structural property of specific dimensions**, tied to the residual stream itself. Per-column scaling directly targets this structural pattern. Per-row scaling does not, because rows correspond to output features, which do not have this same cross-layer persistence.

This is also exactly why AWQ (the more advanced method this project builds toward) works on **input channels**, i.e. columns — not rows. AWQ identifies which columns correspond to consistently-large activation dimensions (by running real calibration text through the model, not just looking at weights in isolation) and specifically protects those columns before quantizing.

---

## Part 5: Reading the Actual Code, Line by Line

Here is the per-column function from this repository's `src/quantizer.py`, with every line explained for someone new to PyTorch syntax.

```python
def quantize_per_col(w: torch.Tensor):
    max_abs_per_col = w.abs().max(dim=0, keepdim=True).values
    scales = max_abs_per_col / 127.0
    q = (w / scales).round().clamp(-127, 127).to(torch.int8)
    dequantized = q.float() * scales
    return q, scales, dequantized
```

**Line by line:**

`w.abs()` — takes the absolute value of every number in the matrix, element by element. A matrix full of negative and positive numbers becomes a matrix of all-positive numbers, same shape.

`.max(dim=0, keepdim=True)` — this is the part that confuses most newcomers, so slow down here.
- `dim=0` in PyTorch refers to the **first** dimension of the tensor's shape. For a matrix of shape `[rows, cols]`, dimension 0 is the *rows* dimension and dimension 1 is the *columns* dimension.
- When you call `.max(dim=0, ...)`, you are saying: "collapse dimension 0 (the rows) by taking the maximum — do this separately for each position along dimension 1 (the columns)." In plain English: **for each column, look down through all the rows and find the biggest value.**
- `keepdim=True` means: after collapsing, keep a dimension of size 1 in that spot instead of removing it entirely. So a `[5120, 3072]` matrix becomes `[1, 3072]` instead of just `[3072]`. This matters for the next line.

`.values` — the `.max()` operation actually returns two things bundled together: the maximum values themselves, and the *indices* (positions) where those maximums occurred. We only want the values here, so we grab `.values` and discard the indices.

`scales = max_abs_per_col / 127.0` — straightforward division, applied to all 3,072 numbers in `max_abs_per_col` at once. Result: one scale value per column, shape `[1, 3072]`.

`(w / scales)` — this is where `keepdim=True` from earlier pays off. `w` has shape `[5120, 3072]`. `scales` has shape `[1, 3072]`. These shapes don't match exactly, but PyTorch has a feature called **broadcasting**: when one tensor has a size of 1 in some dimension, PyTorch automatically "stretches" it to match the other tensor's size in that dimension, without actually copying data in memory. So `scales`, despite only having 1 row, gets treated as if it had 5,120 identical rows — meaning every row of `w` gets divided by the *same* set of 3,072 column-scales. This is exactly the per-column behavior we want: column `j` of every row gets divided by `scales[0, j]`.

`.round()` — rounds every value to the nearest whole number. `2.7` becomes `3.0`. `-1.3` becomes `-1.0`.

`.clamp(-127, 127)` — safety net. Rounding can occasionally produce a value slightly outside `[-127, 127]` due to floating point quirks. `.clamp()` forces any value below -127 up to -127, and any value above 127 down to 127.

`.to(torch.int8)` — converts the data type from floating point to actual 8-bit integers. This is the step that produces the real memory savings — from this point forward, each value occupies 1 byte instead of 4 (float32) or 2 (bfloat16).

`q.float() * scales` — reverses the process. Converts the int8 values back to floating point (`.float()`), then multiplies by the same per-column scales (broadcasting again) to produce the reconstructed approximation.

The `quantize_per_row` function is identical except it uses `dim=1` instead of `dim=0` (collapse across columns instead of rows) and produces scales of shape `[5120, 1]` instead of `[1, 3072]` — one scale per row instead of per column.

The `quantize_per_tensor` function skips the `dim=` argument entirely — `w.abs().max()` with no dimension argument collapses *everything* into a single number, giving you one scale for the whole matrix.

---

## Part 6: Comparison Table

| Method | Scales stored | Precision | Memory overhead | Fixes row-outliers? | Fixes col-outliers? |
|---|---|---|---|---|---|
| Per-tensor | 1 | Lowest | Negligible | No | No |
| Per-row | 5,120 (for qkv_proj) | Medium | ~10 KB per matrix | Yes | No |
| Per-col | 3,072 (for qkv_proj) | Medium-High | ~6 KB per matrix | No | Yes |
| AWQ (per-col + learned protection) | 3,072 + search | Highest | ~6 KB + calibration cost | No | Yes, specifically |

The memory overhead for extra scale factors is trivially small compared to the matrix itself (qkv_proj alone is ~31.5MB in BF16) — so the real trade-off is not memory, it's implementation complexity and the cost of running calibration data through the model to find the right scales (relevant for AWQ specifically, not for plain per-row/per-col).

---

## Part 7: What This Repository Actually Measures

The experiment in `src/quantizer.py`, run against layer 0's `qkv_proj` matrix, computes the **mean absolute reconstruction error** for all three methods side by side. The expectation, based on the reasoning above, is:

```
error_per_tensor  >  error_per_row  >  error_per_col
```

Per-tensor should be worst (single global scale, dominated by rare outliers). Per-row should improve on it (rows that happen to be uniformly elevated, like row 2535 found during inspection, get their own appropriate scale). Per-column should improve further, and matter *more* than per-row specifically because column outliers correspond to persistent, structural residual-stream dimensions rather than incidental row-level elevation — the mechanism explained in Part 4.

The actual numbers from running this on the real model are recorded in this repository's results log, not hardcoded here, because they should come from your own run — not be taken on faith from documentation.
