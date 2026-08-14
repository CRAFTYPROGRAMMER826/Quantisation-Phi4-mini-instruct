# torch is the fundamental numerical computing library
# It gives us the 'tensor' object — the GPU/CPU-aware equivalent of a numpy array
# Every weight matrix in the model is a torch.Tensor
import torch

# transformers is HuggingFace's library
# AutoModelForCausalLM: a smart loader that reads config.json,
#   figures out the right model class (Phi3ForCausalLM in our case),
#   builds the module tree, and fills it with weights from the .safetensors file
# AutoTokenizer: loads the vocabulary and handles text → token ID conversion
from transformers import AutoModelForCausalLM, AutoTokenizer


model_id = "microsoft/Phi-4-mini-instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True   # Phi-4-mini uses custom code in the repo, this allows it to run
)

print("Loading model... (this will take 1-3 minutes and ~8GB RAM)")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,   # load weights in BF16, matching the original format
    device_map="cpu",              # explicitly put everything on CPU — no GPU assumed
    trust_remote_code=True
)

print("Model loaded.")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# model.model  →  the Phi3Model (inner body, excluding lm_head)
# .layers      →  the ModuleList of 32 DecoderLayers
# [0]          →  Layer 0 specifically (index 0 of the list)
# .self_attn   →  the Phi3Attention sub-module inside that layer
# .qkv_proj   →  the Linear layer (contains .weight and optionally .bias)
# .weight      →  the actual tensor of numbers — shape [5120, 3072]
layer0_qkv = model.model.layers[0].self_attn.qkv_proj.weight

print(f"Type:   {type(layer0_qkv)}")
print(f"Shape:  {layer0_qkv.shape}")
print(f"Dtype:  {layer0_qkv.dtype}")
print(f"Device: {layer0_qkv.device}")


# .detach() — creates a copy disconnected from PyTorch's computation graph
#   We do this because we're going to do manual math on this tensor
#   Without detach(), PyTorch would try to track every operation for gradient computation
#   which wastes memory and isn't needed since we're not training
#
# .float() — converts from BF16 to FP32 for our inspection math
#   BF16 arithmetic can have small rounding surprises during computation
#   FP32 gives us clean, predictable numbers for analysis
#   We are NOT changing the model's stored weights — this is a local copy
w = layer0_qkv.detach().float()

# These four numbers are your baseline — you will reference them constantly
print(f"Min value:       {w.min().item():.6f}")
print(f"Max value:       {w.max().item():.6f}")
print(f"Max absolute:    {w.abs().max().item():.6f}")
print(f"Mean absolute:   {w.abs().mean().item():.6f}")