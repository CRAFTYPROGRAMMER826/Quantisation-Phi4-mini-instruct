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


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM
    import torch

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-4-mini-instruct",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True
    )

    # Pull layer 0 qkv_proj, convert to float32 for clean arithmetic
    w = model.model.layers[0].self_attn.qkv_proj.weight.detach().float()
    print(f"Matrix shape: {w.shape}")
    print(f"Total values: {w.numel():,}\n")

    # Run all three quantization methods
    q_tensor, scale_tensor, deq_tensor = quantize_per_tensor(w)
    q_row,    scales_row,   deq_row    = quantize_per_row(w)
    q_col,    scales_col,   deq_col    = quantize_per_col(w)

    # Compute errors
    err_tensor = reconstruction_error(w, deq_tensor)
    err_row    = reconstruction_error(w, deq_row)
    err_col    = reconstruction_error(w, deq_col)

    print("=== RECONSTRUCTION ERROR (Mean Absolute Error) ===")
    print(f"Per-tensor (1 scale total):       {err_tensor:.8f}")
    print(f"Per-row    ({w.shape[0]} scales):      {err_row:.8f}")
    print(f"Per-col    ({w.shape[1]} scales):     {err_col:.8f}")
    
    print("\n=== MEMORY COST OF SCALE FACTORS ===")
    print(f"Per-tensor: 1 scale value")
    print(f"Per-row:    {w.shape[0]:,} scale values")
    print(f"Per-col:    {w.shape[1]:,} scale values")

    print("\n=== INT8 MATRIX MEMORY ===")
    original_bytes = w.numel() * 2          # BF16 = 2 bytes
    quantized_bytes = w.numel() * 1         # INT8 = 1 byte
    print(f"Original BF16:  {original_bytes / 1e6:.2f} MB")
    print(f"Quantized INT8: {quantized_bytes / 1e6:.2f} MB")
    print(f"Saving:         {(1 - quantized_bytes/original_bytes)*100:.1f}%")