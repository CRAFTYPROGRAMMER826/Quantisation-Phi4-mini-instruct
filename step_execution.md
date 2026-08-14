We First explore QKV proj layers because they are the very first Linear Layers Linear layers are the one that have big matrices and occupy significant memory  
Weneed to quantize those matrices primarily


<br>

```
###Modelfile order (this is an order of storage and not an order fo execution)
1. input_layernorm    ← runs FIRST
2. qkv_proj           ← runs SECOND
3. attention scores   ← runs THIRD
4. o_proj             ← runs FOURTH
5. residual add       ← FIFTH
6. post_attn_layernorm← SIXTH
7. gate_up_proj       ← SEVENTH
8. SiLU + gating      ← EIGHTH
9. down_proj          ← NINTH
10. residual add      ← TENTH

```

<br>

```
###Modelfile Printout
Phi3DecoderLayer(
  self_attn:
    qkv_proj: Linear(3072 → 5120)    ← BIG MATRIX, quantize this
    o_proj:   Linear(3072 → 3072)    ← BIG MATRIX, quantize this
  mlp:
    gate_up_proj: Linear(3072 → 16384) ← BIG MATRIX, quantize this
    down_proj:    Linear(8192 → 3072)  ← BIG MATRIX, quantize this
  input_layernorm:  RMSNorm(3072)     ← tiny vector, do NOT quantize
  post_attn_layernorm: RMSNorm(3072)  ← tiny vector, do NOT quantize

```