"""OLMoE parameter counts (from config.json saved in Step 1), measured quality profile Q(k),
and measured expert union per speculative verify window."""
import os, json
import numpy as np, pandas as pd
from config import DATA_DIR, K_LEVELS, SPEC

def _need(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing. Run Step 1 of the notebook first.")
    return path

def load_olmoe_config():
    with open(_need("olmoe_config.json")) as f:
        return json.load(f)

def param_counts():
    """Parameters read from memory per decoded token: shared part and one expert slot."""
    c = load_olmoe_config()
    h, i, L = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"]
    E, V = c["num_experts"], c["vocab_size"]
    nh = c["num_attention_heads"]
    nkv = c.get("num_key_value_heads") or nh
    hd = h // nh
    attn = 2 * h * h + 2 * h * nkv * hd                  # q, o, k, v projections
    shared = L * (attn + h * E) + V * h                  # attention + router + lm_head
    expert = L * 3 * h * i                               # one active expert slot, all layers
    return float(shared), float(expert), int(E)

def load_quality():
    """dict k -> (Q, ppl) with Q(8) = 1, from data/k_profile.csv (measured)."""
    df = pd.read_csv(_need("k_profile.csv")).set_index("k")
    ref = df.loc[max(K_LEVELS), "ppl"]
    return {k: (float(ref / df.loc[k, "ppl"]), float(df.loc[k, "ppl"])) for k in K_LEVELS}

def load_union():
    """dict k -> mean number of distinct experts per layer over a window of gamma+1
    consecutive tokens (measured in Step 1 from OLMoE router logits)."""
    df = pd.read_csv(_need("expert_union.csv"))
    df = df[df.window == SPEC["gamma"] + 1].set_index("k")
    return {k: float(df.loc[k, "union"]) for k in K_LEVELS}
