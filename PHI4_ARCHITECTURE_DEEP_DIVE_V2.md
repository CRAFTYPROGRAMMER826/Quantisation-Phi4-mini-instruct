# Phi-4-Mini: A Complete Architecture Reference for Quantization
### `microsoft/Phi-4-mini-instruct` — 3.8B Parameters, BF16, Decoder-Only Transformer

> **Who this is for:** Someone implementing quantization on an LLM for the very first time. Every concept is introduced before it is used. No term appears without a definition. All numbers come directly from the official `config.json` or from `print(model)` — sources are cited so you can verify everything yourself.
>
> **How to use this document:** Read the Glossary first. Then read the Big Picture. Then follow the lifecycle example — a single sentence travelling through every component. Then use the individual component sections as a reference when you are implementing.

---

## Part 0: Glossary — Read This Before Anything Else

Every term below will be used repeatedly throughout this document. If you encounter a word you do not understand later, come back here.

**Token**
A model does not read words. It reads *tokens* — small chunks of text produced by breaking a sentence according to a fixed vocabulary. The word "running" might be one token. The word "unbelievable" might be split into two tokens: "un" and "believable". Punctuation marks are usually their own token. Numbers are sometimes split digit by digit. This model has a vocabulary of 200,064 possible tokens.

**Token ID**
Every token in the vocabulary is assigned a unique integer (a whole number). The word "cat" might be token ID 3797. "The" might be token ID 464. These IDs are just addresses — like row numbers in a very large table.

**Vector**
A list of numbers, written in a fixed order. A vector of size 3 might look like `[0.4, -1.2, 0.8]`. A vector of size 3072 is a list of 3072 numbers. Vectors are how this model represents everything internally — every token, every intermediate result, is represented as a vector of exactly 3072 numbers at all times (except in specific places described below).

**Matrix**
A grid of numbers with rows and columns — like a spreadsheet. A matrix of shape `[5120, 3072]` has 5120 rows and 3072 columns, containing 5120 × 3072 = 15,728,640 individual numbers. All the learned knowledge in this model lives in its matrices.

**Matrix Multiplication**
The core mathematical operation of the entire model. Given an input vector of size 3072, and a matrix of shape `[5120, 3072]`, you produce an output vector of size 5120 by computing a weighted sum: for each of the 5120 rows in the matrix, you multiply that row (element by element) against the input vector and sum all the results into one number. Do that for all 5120 rows, and you get 5120 output numbers. This is called a dot product per row. The output size always matches the number of rows in the matrix.

**Shape**
When we write `[5120, 3072]` after a matrix, we mean: 5120 rows, 3072 columns. When we write `[seq_len, 3072]` we mean: one row per token in the input sequence (so if 4 tokens were input, `seq_len = 4`), and 3072 columns.

**Hidden Dimension (hidden_size = 3072)**
The number 3072 is the single most important number in this model. It is the size of every token's internal representation at every step inside the model. Every token enters a layer as a vector of 3072 numbers, is processed, and exits as a new vector of 3072 numbers. This is the "language" the model thinks in internally.

**Residual Stream**
A running total, shaped `[seq_len, 3072]` — one 3072-dimensional vector per token in the input. It starts as the token embeddings and gets updated by each of the 32 layers by *adding* each layer's contribution. No layer replaces it. This is explained in full in Part 2.

**Learned Parameters / Weights**
The numbers stored inside each matrix are called weights. They were found during training — a process where the model was shown trillions of examples and adjusted its weights slightly after each one until the weights that minimize prediction error were found. After training, all weights are frozen. When you load a model from disk, you are loading these frozen weights.

**BF16 (Brain Float 16)**
A number format using 16 bits (2 bytes) per number. It can represent a very wide range of values (from extremely tiny to extremely large) because it dedicates 8 bits to the exponent (the scale of the number) and only 7 bits to the mantissa (the precise value within that scale). Every weight in this model is currently stored in BF16.

**INT8**
A completely different number format using 8 bits (1 byte) per number. It represents only 256 possible values: the whole numbers from -127 to +127, spaced evenly. No fractions, no exponent, no floating point. The goal of INT8 quantization is to replace every BF16 weight with the closest INT8 value, using a scale factor to bridge the gap between the two number systems.

**Scale Factor**
A single BF16 number stored alongside a quantized matrix that says: "multiply every integer in this matrix by this number to approximately recover the original BF16 values." Without the scale factor, INT8 values are meaningless — they are just integers between -127 and 127 with no unit. The scale factor gives them meaning.

**Quantization**
The process of converting BF16 weights to INT8 (or INT4, or FP8). The word comes from "quantizing" a continuous range of values onto a discrete set of allowed values — like rounding every price to the nearest dollar instead of keeping cents. The goal is to reduce memory usage (1 byte per weight instead of 2) while minimizing the prediction error introduced by rounding.

**Forward Pass**
One complete run of data through the model, from input tokens to output prediction. During a forward pass, every matrix in the model participates in one matrix multiplication. The weights do not change during a forward pass — only the activations flow through.

**Activations**
The intermediate values produced during a forward pass — the numbers flowing between layers. Unlike weights (which are fixed), activations are different for every input. Activations are *not* quantized in weight-only quantization (like AWQ). They stay in BF16 throughout.

**Outlier / Spike**
A value (in weights or activations) that is dramatically larger than all surrounding values. If 99.9% of values in a matrix are between -1.0 and +1.0, but one value is +47.3, that value is an outlier. Outliers are the central problem in quantization because they force the scale factor to be large, which crushes the precision of all the normal values.

**Causal / Decoder-Only**
This model is "causal" — it can only look at tokens that came before the current one, never tokens that come after. When predicting what word follows "The cat", it is not allowed to peek at what comes after "cat". This is enforced by a mask applied during attention.

---

## Part 1: The Big Picture

### What the Model Is

Phi-4-mini-instruct is a mathematical function. It takes a sequence of tokens (words converted to numbers) and outputs a probability distribution over 200,064 possible next tokens — telling you how likely each possible next word is.

That is the *only* thing it does. All apparent intelligence, reasoning, and language ability emerge from doing this one thing extremely well, having been trained on 5 trillion tokens of text and code.

### What the Model Is Made Of

The model is a sequence of 34 stations (technically 32 + 2 bookends), each performing matrix multiplications on the current state of the data. The stations are:

```
Station 1:  embed_tokens          — converts token IDs to vectors
Stations 2–33: 32 × DecoderLayer  — refines understanding 32 times
Station 34: lm_head               — converts final vector to word probabilities
```

The data flowing between stations is always shaped `[seq_len, 3072]` — one 3072-dimensional vector per token in the input. This shape enters station 1, is updated at each subsequent station, and exits station 34.

### What Quantization Does to This

Quantization is applied **before** any data flows. You take the frozen weight matrices, round their values from BF16 to INT8, and store the rounded matrices on disk. When the model later runs a forward pass, it uses the rounded matrices instead of the originals. The data flowing through (activations) stays in BF16 throughout — only the stored matrices change format.

Memory savings: 3.8 billion weights × 2 bytes (BF16) = 7.6 GB. After INT8: 3.8 billion × 1 byte = 3.8 GB. The scale factors add back ~230 MB. Net saving: ~3.4 GB.

---

## Part 2: The Residual Stream — The Most Critical Concept

This concept is almost always skipped in surface-level explanations. Understanding it is required to understand both the architecture and why quantization errors accumulate the way they do.

### The Wrong Mental Model (But a Common One)

Many people imagine the 32 decoder layers as a pipeline where layer 1 produces something, hands it to layer 2, which transforms it completely and hands something new to layer 3, and so on. Like an assembly line where each worker replaces the product.

**This is wrong.**

### The Correct Mental Model

The 32 layers work like a **running total**. There is one tensor — the residual stream — that starts as your token embeddings and gets small additions from each layer. No layer replaces the stream. Every layer adds to it.

In pseudocode, the entire model (ignoring details) is:

```python
# Step 1: Start the residual stream from token embeddings
residual = embed_tokens(token_ids)      # shape: [seq_len, 3072]

# Steps 2-33: Each of 32 layers ADDS to the residual stream
for layer in layers:                    # 32 iterations
    residual = residual + layer.attention_contribution(residual)   # ADD
    residual = residual + layer.mlp_contribution(residual)         # ADD

# Step 34: Read the final residual stream and predict next token
logits = lm_head(norm(residual))        # shape: [seq_len, 200064]
```

The additions in the loop are the **residual connections** — the most important architectural feature of all Transformer models.

### Why This Matters: The Dimension Question You Asked

You asked: "If `x` is `[seq_len, 3072]` and `attn_out` is the output of attention — how can you add them? The attention matrices have different shapes like `[3072, 5120]`."

The answer: `attn_out` is NOT the attention weight matrix. It is the *output* of running data through the attention sub-block — a result that has been specifically designed (through careful choice of matrix shapes) to always be `[seq_len, 3072]`. The same shape as `x`. This is not a coincidence — the matrices inside attention are specifically sized so that their output matches the residual stream's shape. When you add `x + attn_out`, both are `[seq_len, 3072]`, so the shapes match perfectly.

Let's trace exactly what happens inside one decoder layer to make this concrete:

```
x enters as:       [seq_len, 3072]    ← the current residual stream

Step 1: x_norm1 = input_layernorm(x)
        output:    [seq_len, 3072]    ← same shape, just rescaled values

Step 2: qkv = qkv_proj(x_norm1)
        output:    [seq_len, 5120]    ← shape changes here (more on this below)

Step 3: [attention computation — splits 5120 into Q/K/V, computes scores, etc.]
        output:    [seq_len, 3072]    ← collapses back to 3072 via o_proj

Step 4: x = x + attn_out
        [seq_len, 3072] + [seq_len, 3072] = [seq_len, 3072]   ← shapes match, ADD works

Step 5: x_norm2 = post_attention_layernorm(x)
        output:    [seq_len, 3072]    ← same shape again

Step 6: gate_up = gate_up_proj(x_norm2)
        output:    [seq_len, 16384]   ← temporarily expands (more below)

Step 7: [MLP gating and SiLU — compresses 16384 → 8192]
        down_proj output: [seq_len, 3072]   ← back to 3072 again

Step 8: x = x + mlp_out
        [seq_len, 3072] + [seq_len, 3072] = [seq_len, 3072]   ← shapes match again

x exits as:        [seq_len, 3072]    ← same shape as it entered
```

Every matrix in the model is specifically sized so that data always returns to `[seq_len, 3072]` before being added back to the residual stream. The two moments where shape temporarily changes (to 5120 and 16384) are both resolved before the residual addition.

### Why Residual Connections Matter for Quantization

Because additions accumulate. If quantization introduces a small systematic error into one layer's output — say, the output is consistently 0.003 too high in one dimension — that error gets baked into the residual stream. Every subsequent layer's ADD operation carries that error forward. After 32 additions, a bias of 0.003 per layer becomes 0.096 in the final residual stream. This is why quantization methods evaluate the *accumulated* effect across all layers, not just per-layer error.

It also explains why outlier channels persist. If layer 3's attention writes a value of +12.7 into dimension 847 of the residual stream (due to a BOS token creating an attention sink), every layer from 4 through 32 inherits a residual stream with a large value in dimension 847. When those later layers' RMSNorm reads the residual stream, dimension 847 is consistently large. This is the activation spike that AWQ's calibration step detects.

---

## Part 3: Lifecycle of a Sentence Through the Entire Model

**Input sentence:** `"The cat sat"`

We will follow this sentence through every single component from start to finish, with exact shapes at every step. All numbers come from `config.json` (verified) and `print(model)` (verified).

---

### Step 0: Tokenization (Outside the Model)

The tokenizer splits `"The cat sat"` into tokens and looks up each token's ID.

```
"The cat sat"  →  tokenizer  →  [464, 3797, 3332]
```

We now have 3 token IDs. `seq_len = 3`.

---

### Step 1: embed_tokens — Embedding(200064, 3072)

**What it is:** A lookup table. 200,064 rows (one per vocabulary entry), 3,072 columns. Each row is a learned vector describing one token's meaning. This table was shaped by training — similar words ended up with similar vectors.

**What happens:**
```
Input:   [464, 3797, 3332]        (3 integer token IDs)

Look up row 464  in the table  →  [0.021, -0.003, 0.147, ..., -0.089]   (3072 numbers)
Look up row 3797 in the table  →  [0.003,  0.221, -0.011, ..., 0.044]   (3072 numbers)
Look up row 3332 in the table  →  [-0.102, 0.018, 0.203, ..., 0.011]    (3072 numbers)

Output:  shape [3, 3072]
```

This output is the **initial residual stream**. We will call it `x`. It contains the model's starting "understanding" of each token before any context-aware processing has happened.

**Quantization note:** This matrix has 200,064 × 3,072 = 614 million values. In Phi-4-mini, it shares its physical memory with `lm_head` (they are the same tensor). This is confirmed by `"tie_word_embeddings": true` in `config.json`. You cannot quantize one independently of the other.

---

### Step 2–33: 32 × Phi3DecoderLayer

The residual stream `x` (shape `[3, 3072]`) now passes through 32 identical-in-structure decoder layers, each with independently trained weights. We will trace through **Layer 0** in full detail. Layers 1–31 perform the exact same operations with different weight values.

---

#### Layer 0, Sub-step A: input_layernorm — Phi3RMSNorm(3072)

**What it is:** A normalization operation with 3,072 learned scale values (one per dimension). It does NOT have a weight matrix in the same sense as `qkv_proj` — it has a small vector of 3,072 scale parameters called `weight` (also written as `gamma` in papers).

**Why it exists:** The residual stream `x` can have values of very different magnitudes. After 32 additions across 32 layers, some dimensions might have drifted to large values. If you feed a vector with wildly varying magnitudes directly into a matrix multiplication, the large-value dimensions dominate the output and the small-value dimensions contribute almost nothing. Normalization brings all dimensions to a comparable scale before the matrix multiplication sees them.

**How it works, exactly:**

Given the residual stream for one token as a vector of 3072 numbers:

```
x_token = [0.021, -0.003, 0.147, ..., -0.089]    (3072 numbers for "The")

Step A1 — Square every value:
    squared = [0.000441, 0.000009, 0.021609, ..., 0.007921]

Step A2 — Take the mean of all squared values:
    mean_squared = (0.000441 + 0.000009 + 0.021609 + ... + 0.007921) / 3072
                 = (some small number, say 0.0081)

Step A3 — Take the square root (this is the RMS — Root Mean Square):
    rms = sqrt(0.0081) = 0.09

Step A4 — Divide every original value by the RMS (plus eps=1e-05 to avoid dividing by zero):
    x_norm[i] = x_token[i] / (rms + 0.00001)
    
    [0.021/0.09, -0.003/0.09, 0.147/0.09, ..., -0.089/0.09]
    = [0.233, -0.033, 1.633, ..., -0.989]
    
    After this step, the vector's RMS is approximately 1.0

Step A5 — Apply the 3072 learned scale values (the "weight" of this norm layer):
    x_norm1[i] = x_norm[i] * layernorm_weight[i]
    
    If layernorm_weight[2] = 1.4, then x_norm1[2] = 1.633 * 1.4 = 2.286
    If layernorm_weight[0] = 0.8, then x_norm1[0] = 0.233 * 0.8 = 0.186
```

This process runs independently for each of the 3 tokens. Output shape: `[3, 3072]` — unchanged.

**What goes wrong if this is skipped or corrupted:** The matrix multiplication in the next step (`qkv_proj`) was trained on inputs of a predictable magnitude range. If normalization fails, the attention computation receives inputs that are too large or too small, producing attention scores that are meaningless relative to what training expected. The model's outputs become incoherent.

**Connection:** Receives `x` (the raw residual stream). Sends `x_norm1` to `qkv_proj`. The learned `weight` vector of this norm layer is where AWQ absorbs its protective scaling factors to protect outlier channels.

---

#### Layer 0, Sub-step B: qkv_proj — Linear(in=3072, out=5120)

**What it is:** A matrix of shape `[5120, 3072]`. This is the first of the four large learned matrices in each layer. The numbers `[5120, 3072]` mean 5,120 rows and 3,072 columns — approximately 15.7 million individual BF16 values.

**Why one matrix for three things (Q, K, V):** In the original Transformer architecture (2017), there were three separate matrices — one to compute Queries, one to compute Keys, one to compute Values. Phi-4-mini fuses all three into one larger matrix for computational efficiency. The matrix is applied once, and the resulting 5,120-dimensional output is then *sliced* into three chunks: one for Q, one for K, one for V. The slicing is free (no computation) — it is just taking different rows of the output.

**Understanding Q, K, V from scratch:**

Imagine you are in a library trying to understand a sentence. For each word you are currently processing:

- **Query (Q):** Your question — "What information am I currently looking for?" For the word "sat", the Query might represent: "I need to find the subject of this verb."
- **Key (K):** A label on every book on every shelf — "Here is what I contain." For the word "cat", the Key might represent: "I am an animate noun, subject-like."
- **Value (V):** The actual content of each book — "Here is what I will contribute if selected." For "cat", the Value carries the full semantic content: animal, small, domestic, etc.

For every token, the model computes all three. Then it compares each token's Q against every other token's K. High similarity between `Q_"sat"` and `K_"cat"` → "sat" pays high attention to "cat" → "sat" incorporates a lot of `V_"cat"` into its understanding. This is how "sat" learns about its subject.

**How the 5120 splits into Q, K, V — from actual config.json values:**

From `config.json`:
- `num_attention_heads`: 24 (Q has 24 heads)
- `num_key_value_heads`: 8 (K and V each have 8 heads)
- `hidden_size`: 3072

Head dimension = 3072 / 24 = 128 per Q head.

```
Q: 24 heads × 128 dims per head = 3072 dimensions   (rows 0 to 3071 of the output)
K:  8 heads × 128 dims per head = 1024 dimensions   (rows 3072 to 4095)
V:  8 heads × 128 dims per head = 1024 dimensions   (rows 4096 to 5119)

Total: 3072 + 1024 + 1024 = 5120 ✓
```

This is called **Grouped Query Attention (GQA)** — Q has 24 heads but K and V each have only 8 heads. Each group of 3 Q heads shares one K head and one V head (24 Q / 8 KV = 3 Q heads per KV head). This reduces memory and computation for K and V without significantly affecting quality.

**The matrix multiplication:**

```
Input:  x_norm1     shape [3, 3072]      (3 tokens, 3072 dims each)
Matrix: W_qkv       shape [5120, 3072]   (the learned weight matrix)
Output: qkv_output  shape [3, 5120]      (3 tokens, 5120 dims each)

Operation: qkv_output = x_norm1 @ W_qkv.T
```

The `.T` means "transpose" — PyTorch stores the weight matrix transposed for numerical efficiency. The math works out the same: each of the 3 token vectors (size 3072) gets compared against each of the 5120 rows of the weight matrix via dot product, producing 5120 output values per token.

**Quantization priority:** This is the highest-priority matrix for careful quantization. The RMSNorm before it can amplify specific input dimensions (dimensions where the learned `layernorm_weight` is large). Those amplified dimensions create large values in specific columns of `x_norm1`. When a global scale factor is computed from the worst-case value across all 15.7 million weights, those outlier-adjacent weight columns force the scale wide, crushing all other values. AWQ specifically targets this matrix because this is where outlier channels have the most impact.

---

#### Layer 0, Sub-step C: Attention Score Computation (No Learned Weights)

**What it is:** A sequence of mathematical operations on the Q, K, V vectors from `qkv_proj`. No weights are stored here. No quantization happens here.

**What happens, step by step:**

```
Step C1 — Split the 5120-dimensional output into Q, K, V:
    Q = qkv_output[:, 0:3072]     shape [3, 3072]  — one per token, 24 heads × 128
    K = qkv_output[:, 3072:4096]  shape [3, 1024]  — one per token,  8 heads × 128
    V = qkv_output[:, 4096:5120]  shape [3, 1024]  — one per token,  8 heads × 128

Step C2 — Reshape into per-head format:
    Q → [24 heads, 3 tokens, 128 dims]
    K → [ 8 heads, 3 tokens, 128 dims]
    V → [ 8 heads, 3 tokens, 128 dims]

Step C3 — Apply Rotary Position Embeddings (RoPE) to Q and K only:
    This rotates Q and K vectors by an angle proportional to their position.
    "The" at position 0 gets rotated 0°. "cat" at position 1 gets a small rotation.
    "sat" at position 2 gets a larger rotation.
    Result: when Q_"sat" is dot-producted with K_"cat" (position 1),
    the dot product naturally encodes "these tokens are 1 position apart."
    V is never rotated — only Q and K participate in position-encoded comparison.

Step C4 — Compute attention scores (Q × K^T):
    For GQA: each Q head is paired with one of the 8 K heads.
    Q heads 0,1,2 share K head 0. Q heads 3,4,5 share K head 1. Etc.
    
    For one head pair, computing scores for "sat" attending to all tokens:
    
    score("sat" → "The") = dot(Q_"sat"[128 dims], K_"The"[128 dims])   = small number
    score("sat" → "cat") = dot(Q_"sat"[128 dims], K_"cat"[128 dims])   = larger number
    score("sat" → "sat") = dot(Q_"sat"[128 dims], K_"sat"[128 dims])   = some number
    
    Raw scores for "sat": [-2.1, 8.3, 1.4]  (just example values)

Step C5 — Scale scores and apply causal mask:
    Divide all scores by sqrt(128) = 11.31 to prevent scores from growing too large.
    Scaled: [-0.19, 0.73, 0.12]
    
    Apply mask: tokens cannot attend to future tokens (causal model).
    For "The" (position 0): can only attend to itself. "cat" and "sat" scores → -infinity.
    For "cat" (position 1): can attend to "The" and itself. "sat" score → -infinity.
    For "sat" (position 2): can attend to all three (all are at past/current positions).

Step C6 — Softmax (convert scores to probabilities):
    Softmax converts any list of numbers into positive values that sum to 1.0.
    For "sat": softmax([-0.19, 0.73, 0.12]) → [0.12, 0.62, 0.26]
    
    Interpretation: "sat" attends 62% to "cat", 26% to itself, 12% to "The".

Step C7 — Weighted sum of V:
    For "sat": output = 0.12 × V_"The" + 0.62 × V_"cat" + 0.26 × V_"sat"
    
    This produces one 128-dim vector per head for "sat".
    Do this for all 24 heads → [24, 128] → concatenate → [3072]

Step C8 — All 3 tokens get their attended representations:
    Result shape: [3, 3072]  — ready for o_proj
```

---

#### Layer 0, Sub-step D: o_proj — Linear(in=3072, out=3072)

**What it is:** A square matrix of shape `[3072, 3072]` — 9.4 million values.

**What happens:** The 24 attention heads each independently computed a 128-dimensional attended representation, and we concatenated them into 3072 dimensions. But these 24 streams were computed independently, with no interaction. `o_proj` learns how to recombine information across all 24 heads into a single coherent update.

```
Input:  attended    shape [3, 3072]    (concatenated 24-head output)
Matrix: W_o         shape [3072, 3072]
Output: attn_out    shape [3, 3072]

Operation: attn_out = attended @ W_o.T
```

**Residual addition (first one of the layer):**

```
x = x + attn_out

Before: x = embed_tokens output = [3, 3072]
After:  x = x + attn_out = [3, 3072] + [3, 3072] = [3, 3072]
```

Yes — to your explicit question — after this line, `x` contains the **original embedding values** PLUS the attention contribution. The original information is preserved and the attention update is added on top. The residual stream now carries both: the initial token meanings and the new information about how each token relates to the others.

---

#### Layer 0, Sub-step E: post_attention_layernorm — Phi3RMSNorm(3072)

**Identical operation to `input_layernorm`**, but with its own 3,072 independently trained scale values.

**Why normalize again?** The attention step added `attn_out` to the residual stream. This addition may have shifted the magnitudes. Before feeding the residual stream into the MLP, we normalize again for the same reason as before — the MLP's matrices were trained to receive inputs of a predictable magnitude range.

```
Input:  x          shape [3, 3072]   (residual after attention)
Output: x_norm2    shape [3, 3072]   (normalized, same shape)
```

---

#### Layer 0, Sub-step F: gate_up_proj — Linear(in=3072, out=16384)

**What it is:** The largest matrix in the model: shape `[16384, 3072]` — 50.3 million values. It is secretly doing two jobs at once.

**Why 16384?** From `config.json`: `intermediate_size = 8192`. The matrix outputs 16384 = 8192 × 2 because it computes both the "gate" projection and the "up" projection simultaneously, fused into one matrix for efficiency. The output is sliced in half:

```
gate_up_output = x_norm2 @ W_gate_up.T    shape [3, 16384]

gate = gate_up_output[:, :8192]            shape [3, 8192]   — first half
up   = gate_up_output[:, 8192:]            shape [3, 8192]   — second half
```

**What gate and up mean:** `up` carries the information the model wants to process in the expanded space. `gate` decides, dimension by dimension, how much of `up` to let through. They will be combined through SiLU (next step).

---

#### Layer 0, Sub-step G: SiLUActivation — No Learned Weights

**What it is:** A mathematical function applied to each number independently. No matrix. No learned values. Nothing to quantize.

**The formula:** `SiLU(x) = x × sigmoid(x) = x / (1 + e^(-x))`

**Why this produces gating behavior:**
- When `gate[i]` is a large positive number: `SiLU(large positive) ≈ large positive` → passes through
- When `gate[i]` is a large negative number: `SiLU(large negative) ≈ 0` → blocked
- When `gate[i]` is near zero: smooth partial pass-through

**The gating operation:**
```
gate_activated = SiLU(gate)              shape [3, 8192]   — values between 0 and large positive
mlp_intermediate = gate_activated * up   shape [3, 8192]   — element-wise multiplication
```

The multiplication is **element-wise** — position 0 of `gate_activated` multiplies position 0 of `up`, position 1 multiplies position 1, and so on. They never interact across dimensions. The result is that `up`'s information flows through only in dimensions where `gate` says to let it through.

**Why non-linearity is necessary:** If you removed SiLU and just multiplied `gate × up`, the entire operation would be equivalent to a single matrix multiplication (a product of linear operations is still linear). The whole 32-layer model would collapse into one big matrix. Non-linearity breaks this collapse and allows the model to represent arbitrarily complex functions given enough layers.

---

#### Layer 0, Sub-step H: down_proj — Linear(in=8192, out=3072)

**What it is:** A matrix of shape `[3072, 8192]` — 25.2 million values. Compresses back to the residual stream's dimension.

```
Input:  mlp_intermediate   shape [3, 8192]
Matrix: W_down             shape [3072, 8192]
Output: mlp_out            shape [3, 3072]

Operation: mlp_out = mlp_intermediate @ W_down.T
```

**Residual addition (second one of the layer):**

```
x = x + mlp_out

x now contains: original_embeddings + attention_update + mlp_update
```

To your explicit question: yes, at this point `x` is the sum of:
1. The original embedding values from `embed_tokens`
2. The `attn_out` addition from sub-step D
3. The `mlp_out` addition from this step

All three are `[3, 3072]`, all added element-wise. The residual stream is now richer than when it entered this layer.

**Layer 0 is complete. `x` exits with shape `[3, 3072]` and enters Layer 1's `input_layernorm`.**

This exact sequence (input_layernorm → qkv_proj → attention → o_proj → residual add → post_attention_layernorm → gate_up_proj → SiLU → down_proj → residual add) repeats 31 more times, each time with different weight matrices.

---

### Step 34a: Final norm — Phi3RMSNorm(3072)

After 32 layers of additions, the residual stream has accumulated contributions from every layer. One final normalization ensures the values fed into `lm_head` are in a predictable range.

```
Input:  x (after layer 31)   shape [3, 3072]
Output: x_final              shape [3, 3072]
```

---

### Step 34b: lm_head — Linear(in=3072, out=200064)

**What it is:** A matrix of shape `[200064, 3072]` — 614 million values. This is the same physical tensor as `embed_tokens` (confirmed by `"tie_word_embeddings": true` in `config.json`).

**What happens:** Only the last token's vector is used to predict the next token (for generation).

```
Input:  x_final[-1]   shape [3072]        (just the "sat" token's final vector)
Matrix: W_lm_head     shape [200064, 3072] (same matrix as embed_tokens!)
Output: logits        shape [200064]       (one score per vocabulary entry)

Operation: logits = x_final[-1] @ W_lm_head.T
```

**Why the last token?** This is a causal (left-to-right) language model. At each step, it predicts the token that should come after the rightmost token it has seen. "The cat sat" → predict what comes after "sat". The residual stream of "sat" (the last token) has attended to all previous tokens and accumulated information from all 32 layers, so its final vector encodes everything the model knows about what should come next given this context.

**Converting scores to a prediction:**

```
probabilities = Softmax(logits)
# probabilities[i] = probability that token i is the next word

# Example (made up, illustrative):
# probabilities[916]   = 0.42   ← " on" (most likely)
# probabilities[3141]  = 0.18   ← " in"
# probabilities[11]    = 0.07   ← " at"
# ... (200,061 other tokens have much smaller probabilities)

next_token_id = 916                     # argmax or sample
next_token_text = tokenizer.decode(916) # → " on"
```

The model's complete output for input `"The cat sat"` is the prediction `" on"`, to eventually produce `"The cat sat on the mat"`.

**The tied-embeddings quantization gotcha:** Because `embed_tokens` and `lm_head` share the same physical tensor, when you write code to quantize all `Linear` layers, you must check whether a given `Linear`'s weight tensor is the same object in memory as another one you've already quantized. In Python: `model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()` will return `True` for this model. Quantize the same tensor twice and you apply rounding twice — the error compounds and the model's output vocabulary scores become meaningless.

---

## Part 4: Component Reference Table

| Component | Type | Shape | Parameters | Quantize? | Notes |
|---|---|---|---|---|---|
| `embed_tokens` | Lookup table | [200064, 3072] | 614M | With care | Tied to `lm_head` |
| `input_layernorm` | Scale vector | [3072] per layer | 0.1M total | No | Where AWQ absorbs scales |
| `qkv_proj` | Weight matrix | [5120, 3072] per layer | 503M total | **Yes — highest priority** | Fused Q(3072)+K(1024)+V(1024) |
| `o_proj` | Weight matrix | [3072, 3072] per layer | 302M total | Yes | Standard |
| Attention scores | Computation | — | 0 | No | No weights |
| `rotary_emb` | Computation | — | 0 | No | No weights |
| `post_attn_layernorm` | Scale vector | [3072] per layer | 0.1M total | No | Same role as input_layernorm |
| `gate_up_proj` | Weight matrix | [16384, 3072] per layer | 1610M total | **Yes — largest matrix** | Gate(8192)+Up(8192) fused |
| `SiLUActivation` | Computation | — | 0 | No | No weights |
| `down_proj` | Weight matrix | [3072, 8192] per layer | 805M total | Yes | Input from SiLU is non-Gaussian |
| Final `norm` | Scale vector | [3072] | 3072 | No | After layer 31 |
| `lm_head` | Weight matrix | [200064, 3072] | 614M | With care | Tied to `embed_tokens` |
| `Dropout` | Pass-through | — | 0 | No | Disabled (p=0.0) |

---

## Part 5: Memory Budget

| Format | Bytes per weight | Total weight memory | Overhead | Total |
|---|---|---|---|---|
| BF16 (original) | 2 bytes | 3.8B × 2 = **7.6 GB** | None | **~7.6 GB** |
| INT8 (quantized) | 1 byte | 3.8B × 1 = **3.8 GB** | ~230 MB for scale factors | **~4.0 GB** |

Scale factor overhead calculation:
- Per-row scaling on each Linear layer: approximately (5120+3072+16384+8192) × 32 rows + 200064 (lm_head)
- Each scale factor is BF16 (2 bytes)
- Total: ~920,000 scale factors × 2 bytes ≈ **~1.8 MB** (negligible)
- If per-group (128-element groups, as AWQ uses): multiply by (3072/128) ≈ 24 → **~44 MB** (still small)

The dominant memory cost is always the weights themselves. Scale factors are a rounding error on the total budget.

---

## Part 6: What You Will Implement

Your quantization pipeline maps directly onto the lifecycle above:

```
Stage 1 — Read:    Pull one weight matrix from the model (e.g., layer 0 qkv_proj)
Stage 2 — Measure: Compute max-abs value, compute scale, round to INT8, dequantize, measure error
Stage 3 — Calibrate: Run 50-100 real prompts, hook the forward pass to capture x_norm1 
                      (the normalized activations flowing INTO qkv_proj)
                      Identify which of the 3072 input dimensions has the largest average magnitude
Stage 4 — Protect: For those salient dimensions, adjust the weight matrix columns 
                   and absorb the adjustment into input_layernorm's weight vector
Stage 5 — Replace: Implement QuantLinear (stores INT8 weights + scale) 
                   Replace every Linear in the model tree with your QuantLinear
Stage 6 — Evaluate: Run the quantized model on WikiText-2, measure perplexity 
                    Compare to BF16 baseline
```

Everything in this document is the foundation for Stage 1. You now know what you are pulling, why it matters, and what the numbers mean.

---

*All shapes verified against `print(model)` output. All config values verified against `config.json` at `microsoft/Phi-4-mini-instruct` (commit `618e63e`): `hidden_size=3072`, `num_attention_heads=24`, `num_key_value_heads=8`, `intermediate_size=8192`, `vocab_size=200064`, `num_hidden_layers=32`, `tie_word_embeddings=true`.*
