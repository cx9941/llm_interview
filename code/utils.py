# -*- coding: utf-8 -*-
import os, json, math
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_jsonl_prompts(path: str) -> Dataset:
    assert os.path.exists(path), f"文件不存在: {path}"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            p = str(obj.get("prompt", "")).strip()
            if p: rows.append({"prompt": p})
    assert rows, "数据为空"
    return Dataset.from_list(rows)

def load_jsonl_dpo(path: str) -> Dataset:
    assert os.path.exists(path), f"文件不存在: {path}"
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            p = str(obj.get("prompt", "")).strip()
            c = obj.get("chosen", None)
            r = obj.get("rejected", None)
            rows.append({"prompt": p, "chosen": c, "rejected": r})
    assert rows, "数据为空"
    return Dataset.from_list(rows)


def tokenize_prompts(tokenizer: AutoTokenizer, prompts: List[str], max_len: int, device) -> Dict[str, torch.Tensor]:
    enc = tokenizer(prompts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    return {k: v.to(device) for k, v in enc.items()}

def left_pad(x: torch.Tensor, pad_id: int, tgt_len: int) -> torch.Tensor:
    if x.size(0)==tgt_len: return x
    pad = torch.full((tgt_len-x.size(0),), pad_id, dtype=x.dtype, device=x.device)
    return torch.cat([pad, x], dim=0)

def gather_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logprobs = torch.log_softmax(logits, dim=-1)
    return logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
