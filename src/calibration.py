import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data.calibration_prompts import CALIBRATION_PROMPTS

# ── CONFIG ──────────────────────────────────────────────────────
MODEL_ID        = "microsoft/Phi-4-mini-instruct"
MAX_SEQ_LEN     = 128     # truncate prompts to this many tokens
                           # keeps RAM manageable, still captures outlier behavior
DEVICE          = "cpu"
# ────────────────────────────────────────────────────────────────


def load_model_and_tokenizer():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    model.eval()   # disable dropout, put model in inference mode
    return model, tokenizer


def register_hooks(model):
    """
    Attach a forward hook to every Linear layer we care about.

    A forward hook is a function PyTorch calls automatically
    every time that layer runs a forward pass. The hook receives:
        module  — the layer itself
        input   — a tuple of tensors fed INTO the layer
        output  — the tensor produced BY the layer (we ignore this)

    We save input[0] — the activation arriving at this layer.
    input is a tuple because layers can technically receive multiple
    inputs; for Linear layers, input[0] is always the one tensor.

    We store activations in a dict keyed by a string like
    "layer_0_qkv_proj" so we can look them up later.
    """
    # This dict will accumulate activation tensors across all prompts
    # Structure: { "layer_N_matrix_name": [tensor_from_prompt_1, tensor_from_prompt_2, ...] }
    activation_store = {}

    hooks = []   # keep references so we can remove them later

    # Which sub-modules to hook, and what name to give them
    targets = {
        "qkv_proj":    lambda layer: layer.self_attn.qkv_proj,
        "o_proj":      lambda layer: layer.self_attn.o_proj,
        "gate_up_proj":lambda layer: layer.mlp.gate_up_proj,
        "down_proj":   lambda layer: layer.mlp.down_proj,
    }

    for layer_idx in range(32):
        layer = model.model.layers[layer_idx]

        for mat_name, getter in targets.items():
            key = f"layer_{layer_idx}_{mat_name}"
            activation_store[key] = []   # start with empty list for this layer+matrix

            # Closure — captures key by reference so each hook knows its own name
            def make_hook(k):
                def hook_fn(module, input, output):
                    # input[0] shape: [seq_len, hidden_dim]
                    # We detach so we don't accidentally track gradients
                    # We move to float32 for clean statistics
                    # We keep it on CPU since that's where the model is
                    act = input[0].detach().float()
                    activation_store[k].append(act)
                return hook_fn

            # register_forward_hook attaches our function to the module
            # it returns a handle we can use to remove the hook later
            handle = getter(layer).register_forward_hook(make_hook(key))
            hooks.append(handle)

    return activation_store, hooks


def remove_hooks(hooks):
    """Always remove hooks after use — they consume memory and slow inference."""
    for handle in hooks:
        handle.remove()


def run_calibration(model, tokenizer):
    """
    Run every calibration prompt through the model.
    Hooks collect activations automatically during each forward pass.
    Returns the activation store — a dict of lists of tensors.
    """
    activation_store, hooks = register_hooks(model)

    print(f"\nRunning {len(CALIBRATION_PROMPTS)} calibration prompts...")
    print("Hooks registered. Activations will be captured automatically.\n")

    with torch.no_grad():   # no_grad means PyTorch skips gradient tracking
                             # critical for memory — forward-only, no backprop
        for i, prompt in enumerate(CALIBRATION_PROMPTS):

            # Tokenize — convert text to token IDs
            # truncation=True enforces MAX_SEQ_LEN
            # return_tensors="pt" gives us PyTorch tensors directly
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN
            )

            # Run the forward pass
            # The hooks fire automatically inside this call
            # We don't care about the output (next-token logits) here
            _ = model(**inputs)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(CALIBRATION_PROMPTS)} prompts")

    print("\nAll prompts processed. Removing hooks...")
    remove_hooks(hooks)

    return activation_store


def compute_activation_stats(activation_store):
    """
    For each layer+matrix, concatenate all activation tensors collected
    across prompts and compute per-column statistics.

    Why per-column?
        Each column index corresponds to one input dimension of the
        residual stream (or MLP intermediate space for down_proj).
        We want to know: which input DIMENSIONS are consistently large?
        That's a column-wise question, not a row-wise one.

    Returns a dict:
        { "layer_N_matrix_name": {
            "mean_abs":  tensor of shape [hidden_dim]  — per-column mean abs activation
            "max_abs":   tensor of shape [hidden_dim]  — per-column max abs activation
            "std":       tensor of shape [hidden_dim]  — per-column std deviation
          }
        }
    """
    stats = {}

    print("\nComputing per-column activation statistics...")

    for key, tensor_list in activation_store.items():
        if not tensor_list:
            continue

        # Each tensor in tensor_list has shape [seq_len, hidden_dim]
        # Concatenate along dim=0 to get [total_tokens, hidden_dim]
        # total_tokens = sum of seq lengths across all prompts
        all_activations = torch.cat(tensor_list, dim=0)

        # Compute statistics across ALL tokens (dim=0)
        # Result shape: [hidden_dim] — one number per input dimension
        mean_abs = all_activations.abs().mean(dim=0)   # average magnitude per column
        max_abs  = all_activations.abs().max(dim=0).values  # worst case per column
        std      = all_activations.std(dim=0)           # how much each column varies

        stats[key] = {
            "mean_abs": mean_abs,
            "max_abs":  max_abs,
            "std":      std,
            "n_tokens": all_activations.shape[0]   # total tokens seen
        }

    return stats


def print_outlier_report(stats, top_k=10):
    """
    For each layer+matrix, print:
    1. The top-k input dimensions by mean_abs (these are AWQ's salient channels)
    2. The ratio of the worst column to the median column
       (a high ratio means outliers are severe and AWQ protection matters more)
    """
    print("\n" + "="*70)
    print("ACTIVATION OUTLIER REPORT")
    print("="*70)
    print("For each matrix: top columns by mean absolute activation")
    print("High mean_abs = this input dimension is consistently large")
    print("High ratio    = outlier is far from typical = AWQ protection needed")
    print("="*70)

    # Group by matrix type for cleaner reading
    matrix_types = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]

    for mat_name in matrix_types:
        print(f"\n{'─'*60}")
        print(f"Matrix type: {mat_name}")
        print(f"{'─'*60}")
        print(f"{'Layer':>6} {'Tokens':>8} {'Top column':>12} {'mean_abs':>10} {'median':>10} {'ratio':>8}")
        print(f"{'':>6} {'':>8} {'(dim index)':>12} {'':>10} {'':>10} {'':>8}")
        print("-"*70)

        for layer_idx in range(32):
            key = f"layer_{layer_idx}_{mat_name}"
            if key not in stats:
                continue

            s = stats[key]
            mean_abs = s["mean_abs"]

            # Find the single most active column
            top_val,  top_idx  = mean_abs.max(dim=0)
            median_val = mean_abs.median()
            ratio = (top_val / (median_val + 1e-8)).item()

            print(f"{layer_idx:>6} {s['n_tokens']:>8} "
                  f"{top_idx.item():>12} {top_val.item():>10.5f} "
                  f"{median_val.item():>10.5f} {ratio:>8.1f}x")

    # Now print the top-k salient columns for the first layer of qkv_proj
    # as a detailed example — this is exactly what AWQ's calibration uses
    print(f"\n{'='*70}")
    print("DETAILED VIEW — Layer 0 qkv_proj: top-10 salient input dimensions")
    print("These are the columns AWQ would protect with learned scaling")
    print("="*70)

    key = "layer_0_qkv_proj"
    if key in stats:
        mean_abs = stats[key]["mean_abs"]
        top = mean_abs.topk(top_k)
        print(f"{'Rank':>6} {'Column (dim)':>14} {'mean_abs':>12} {'max_abs':>12}")
        print("-"*50)
        for rank, (val, idx) in enumerate(zip(top.values, top.indices)):
            max_val = stats[key]["max_abs"][idx].item()
            print(f"{rank+1:>6} {idx.item():>14} {val.item():>12.5f} {max_val:>12.5f}")


def save_stats(stats, path="results/activation_stats.pt"):
    """
    Save the computed statistics to disk so you don't have to rerun
    calibration every time you want to experiment with AWQ scaling.
    torch.save / torch.load uses Python's pickle format.
    """
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save only the stat tensors, not the raw activations (those are huge)
    torch.save(stats, path)
    print(f"\nStats saved to {path}")
    print("Load later with: stats = torch.load('results/activation_stats.pt')")


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    activation_store = run_calibration(model, tokenizer)
    stats            = compute_activation_stats(activation_store)
    print_outlier_report(stats)
    save_stats(stats)