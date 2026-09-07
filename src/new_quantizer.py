import torch
from transformers import AutoModelForCausalLM

# ── paste your three functions from quantizer.py here ──
# quantize_per_tensor(w), quantize_per_row(w), quantize_per_col(w)
# reconstruction_error(original, dequantized)

import torch

def quantize_per_tensor(w: torch.Tensor):
    """
    Naive whole-matrix quantization.
    One scale for all values in the matrix.
    
    Args:
        w: float32 tensor of shape [rows, cols]
    Returns:
        q:           int8 tensor, same shape
        scale:       single float scalar
        dequantized: float32 tensor, same shape (the reconstructed approximation)
    """
    # Step 1: find the largest absolute value in the entire matrix
    # This single value determines how wide each integer slot is
    max_abs = w.abs().max()
    
    # Step 2: compute scale — how much real-number range each integer slot covers
    # Dividing by 127 because INT8 goes from -127 to +127 (we use 127, not 128,
    # to keep the range symmetric around zero)
    scale = max_abs / 127.0
    
    # Step 3: divide every value by scale, round to nearest integer,
    # clamp to [-127, 127] to handle any floating point overshoot,
    # cast to int8 (1 byte per value instead of 4 bytes for float32)
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    
    # Step 4: dequantize — multiply integers back by scale to get
    # approximate float values. These will NOT exactly match originals.
    # The gap between original and dequantized is quantization error.
    dequantized = q.float() * scale
    
    return q, scale, dequantized


def reconstruction_error(original: torch.Tensor, dequantized: torch.Tensor):
    """
    Measures how much information was lost during quantization.
    Mean Absolute Error: average absolute difference per weight value.
    Lower is better.
    """
    return (original - dequantized).abs().mean().item()


def quantize_per_row(w: torch.Tensor):
    """
    Per-row quantization.
    Each of the 5120 rows gets its own scale based on that row's max-abs.
    Rows with large values get a wide scale.
    Rows with small values get a tight scale (more precision for them).
    
    Args:
        w: float32 tensor of shape [rows, cols]
    Returns:
        q:           int8 tensor, same shape
        scales:      float32 tensor of shape [rows, 1]
        dequantized: float32 tensor, same shape
    """
    # Compute max absolute value for EACH ROW independently
    # dim=1 collapses across columns, keepdim=True keeps shape [5120, 1]
    # so we can broadcast against the [5120, 3072] matrix later
    max_abs_per_row = w.abs().max(dim=1, keepdim=True).values  # shape [5120, 1]
    
    # One scale per row — shape [5120, 1]
    scales = max_abs_per_row / 127.0
    
    # Divide each row by its own scale
    # Broadcasting: [5120, 3072] / [5120, 1] → each row divided by its own scalar
    q = (w / scales).round().clamp(-127, 127).to(torch.int8)
    
    # Multiply each row back by its own scale to reconstruct
    dequantized = q.float() * scales
    
    return q, scales, dequantized


def quantize_per_col(w: torch.Tensor):
    """
    Per-column quantization.
    Each of the 3072 columns gets its own scale.
    
    Columns of the weight matrix correspond to INPUT dimensions —
    the 3072-dimensional residual stream feeding into this layer.
    If one input dimension consistently has large activations (outlier channel),
    the corresponding weight column tends to also be large.
    This is why AWQ works on columns, not rows.
    
    Args:
        w: float32 tensor of shape [rows, cols]  
    Returns:
        q:           int8 tensor, same shape
        scales:      float32 tensor of shape [1, cols]
        dequantized: float32 tensor, same shape
    """
    # dim=0 collapses across rows, keepdim=True keeps shape [1, 3072]
    max_abs_per_col = w.abs().max(dim=0, keepdim=True).values  # shape [1, 3072]
    
    scales = max_abs_per_col / 127.0
    
    # Broadcasting: [5120, 3072] / [1, 3072] → each column divided by its own scalar
    q = (w / scales).round().clamp(-127, 127).to(torch.int8)
    
    dequantized = q.float() * scales
    
    return q, scales, dequantized



print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-mini-instruct",
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True
)

# Matrix types to inspect — one per layer
TARGETS = {
    "qkv_proj":    lambda l: l.self_attn.qkv_proj.weight,
    "o_proj":      lambda l: l.self_attn.o_proj.weight,
    "gate_up_proj":lambda l: l.mlp.gate_up_proj.weight,
    "down_proj":   lambda l: l.mlp.down_proj.weight,
}

results = {}   # keyed by matrix_name, value = list of (layer_idx, et, er, ec)

for mat_name, getter in TARGETS.items():
    print(f"\n{'='*60}")
    print(f"Matrix: {mat_name}")
    print(f"{'='*60}")
    print(f"{'Layer':>6} {'Shape':>14} {'Per-Tensor':>12} {'Per-Row':>12} {'Per-Col':>12} {'Best':>8}")
    print("-"*70)

    layer_results = []

    for layer_idx in range(32):
        layer = model.model.layers[layer_idx]

        # pull the weight tensor, detach from graph, cast to float32
        w = getter(layer).detach().float()

        # run all three quantization methods
        _, _, deq_t = quantize_per_tensor(w)
        _, _, deq_r = quantize_per_row(w)
        _, _, deq_c = quantize_per_col(w)

        # compute errors
        et = reconstruction_error(w, deq_t)
        er = reconstruction_error(w, deq_r)
        ec = reconstruction_error(w, deq_c)

        # find which method won
        errors = [et, er, ec]
        names  = ['tensor', 'row', 'col']
        best   = names[errors.index(min(errors))]

        print(f"{layer_idx:>6} {str(tuple(w.shape)):>14} {et:>12.8f} {er:>12.8f} {ec:>12.8f} {best:>8}")
        layer_results.append((layer_idx, et, er, ec, best))

    results[mat_name] = layer_results

# ── Summary table ──
print(f"\n{'='*60}")
print("SUMMARY — How often each method wins per matrix type")
print(f"{'='*60}")
print(f"{'Matrix':>14} {'tensor wins':>12} {'row wins':>12} {'col wins':>12}")
print("-"*54)
for mat_name, rows in results.items():
    tw = sum(1 for _,_,_,_,b in rows if b=='tensor')
    rw = sum(1 for _,_,_,_,b in rows if b=='row')
    cw = sum(1 for _,_,_,_,b in rows if b=='col')
    print(f"{mat_name:>14} {tw:>12} {rw:>12} {cw:>12}")