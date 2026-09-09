import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

try:
    from data.calibration_prompts import CALIBRATION_PROMPTS
except ImportError:
    from calibration_prompts import CALIBRATION_PROMPTS

MODEL_ID    = "microsoft/Phi-4-mini-instruct"
MAX_SEQ_LEN = 128
DEVICE      = "cpu"


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
    model.eval()
    return model, tokenizer


def register_hooks(model):
    """
    Instead of storing full activation tensors (which causes the shape
    mismatch and OOM issues), we compute running statistics INSIDE the
    hook itself, one prompt at a time.

    For each layer+matrix we track:
        sum_abs  — running sum of abs values per column
        max_abs  — running max of abs values per column
        n_tokens — total number of tokens seen so far

    After all prompts: mean_abs = sum_abs / n_tokens
    This is mathematically identical to concatenating everything and
    computing stats, but uses O(hidden_dim) memory instead of
    O(n_prompts * seq_len * hidden_dim).
    """
    stats_store = {}
    hooks = []

    targets = {
        "qkv_proj":     lambda layer: layer.self_attn.qkv_proj,
        "o_proj":       lambda layer: layer.self_attn.o_proj,
        "gate_up_proj": lambda layer: layer.mlp.gate_up_proj,
        "down_proj":    lambda layer: layer.mlp.down_proj,
    }

    for layer_idx in range(32):
        layer = model.model.layers[layer_idx]

        for mat_name, getter in targets.items():
            key = f"layer_{layer_idx}_{mat_name}"

            # Pre-initialise with None — we'll set shapes on first call
            stats_store[key] = {
                "sum_abs": None,
                "max_abs": None,
                "n_tokens": 0
            }

            def make_hook(k):
                def hook_fn(module, input, output):
                    act = input[0].detach().float()

                    # input[0] can arrive as [batch, seq_len, hidden_dim]
                    # or as [seq_len, hidden_dim] depending on the layer.
                    # Flatten everything into [n_tokens, hidden_dim] safely.
                    if act.dim() == 3:
                        # shape [batch, seq_len, hidden_dim]
                        # batch is always 1 for our single-prompt inference
                        act = act.view(-1, act.shape[-1])
                    elif act.dim() == 2:
                        # shape [seq_len, hidden_dim] — already correct
                        pass
                    else:
                        # unexpected shape — skip this activation safely
                        return

                    # hidden_dim for this matrix
                    hdim = act.shape[-1]

                    s = stats_store[k]

                    if s["sum_abs"] is None:
                        # First time we see this layer — initialise accumulators
                        s["sum_abs"] = torch.zeros(hdim)
                        s["max_abs"] = torch.zeros(hdim)

                    # Update running sum and max, column by column
                    # act.abs().sum(dim=0) sums absolute values across tokens
                    # giving shape [hidden_dim]
                    s["sum_abs"] += act.abs().sum(dim=0)
                    s["max_abs"]  = torch.max(s["max_abs"], act.abs().max(dim=0).values)
                    s["n_tokens"] += act.shape[0]

                return hook_fn

            handle = getter(layer).register_forward_hook(make_hook(key))
            hooks.append(handle)

    return stats_store, hooks


def remove_hooks(hooks):
    for handle in hooks:
        handle.remove()


def run_calibration(model, tokenizer):
    stats_store, hooks = register_hooks(model)

    print(f"\nRunning {len(CALIBRATION_PROMPTS)} calibration prompts...")

    with torch.no_grad():
        for i, prompt in enumerate(CALIBRATION_PROMPTS):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN
            )
            _ = model(**inputs)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{len(CALIBRATION_PROMPTS)} prompts")

    print("\nAll prompts processed. Removing hooks...")
    remove_hooks(hooks)
    return stats_store


def compute_final_stats(stats_store):
    """
    Convert running sums into final mean_abs values.
    Everything else (max_abs, n_tokens) is already final.
    """
    print("Computing final statistics...")
    final = {}
    for key, s in stats_store.items():
        if s["sum_abs"] is None or s["n_tokens"] == 0:
            continue
        final[key] = {
            "mean_abs": s["sum_abs"] / s["n_tokens"],
            "max_abs":  s["max_abs"],
            "n_tokens": s["n_tokens"]
        }
    return final


def print_outlier_report(stats, top_k=10):
    print("\n" + "="*70)
    print("ACTIVATION OUTLIER REPORT")
    print("="*70)
    print("mean_abs = average absolute activation per input dimension")
    print("ratio    = top column / median column  (higher = worse outlier)")
    print("="*70)

    matrix_types = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]

    for mat_name in matrix_types:
        print(f"\n{'─'*65}")
        print(f"Matrix: {mat_name}")
        print(f"{'─'*65}")
        print(f"{'Layer':>6} {'Tokens':>8} {'Top dim':>10} "
              f"{'mean_abs':>10} {'median':>10} {'ratio':>8}")
        print("-"*65)

        for layer_idx in range(32):
            key = f"layer_{layer_idx}_{mat_name}"
            if key not in stats:
                continue

            mean_abs   = stats[key]["mean_abs"]
            top_val, top_idx = mean_abs.max(dim=0)
            median_val = mean_abs.median()
            ratio      = (top_val / (median_val + 1e-8)).item()

            print(f"{layer_idx:>6} {stats[key]['n_tokens']:>8} "
                  f"{top_idx.item():>10} {top_val.item():>10.4f} "
                  f"{median_val.item():>10.4f} {ratio:>8.1f}x")

    # Detailed top-10 for layer 0 qkv_proj — the AWQ salient channels
    print(f"\n{'='*65}")
    print("DETAIL — Layer 0 qkv_proj: top-10 salient input dimensions")
    print("These dimensions would receive AWQ protective scaling")
    print("="*65)
    key = "layer_0_qkv_proj"
    if key in stats:
        mean_abs = stats[key]["mean_abs"]
        top = mean_abs.topk(top_k)
        print(f"{'Rank':>5} {'Dim':>8} {'mean_abs':>12} {'max_abs':>12}")
        print("-"*42)
        for rank, (val, idx) in enumerate(zip(top.values, top.indices)):
            max_val = stats[key]["max_abs"][idx].item()
            print(f"{rank+1:>5} {idx.item():>8} {val.item():>12.4f} {max_val:>12.4f}")


def save_stats(stats, path="results/activation_stats.pt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(stats, path)
    print(f"\nStats saved to {path}")
    print("Reload with: stats = torch.load('results/activation_stats.pt')")


if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    stats_store      = run_calibration(model, tokenizer)
    stats            = compute_final_stats(stats_store)
    print_outlier_report(stats)
    save_stats(stats)