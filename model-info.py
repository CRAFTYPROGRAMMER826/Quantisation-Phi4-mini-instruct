from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "microsoft/Phi-4-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",   # you're on 16GB RAM, no GPU assumed — stay explicit about this
)

print(model)    