# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

@dataclass
class RunArgs:
    method: str = field(default="ppo", metadata={"help": "ppo | grpo | dpo | gspo"})
    model_name_or_path: str = field(default="pretrained_moels/Qwen2.5-0.5B", metadata={"help": "HF 仓库名或本地路径"})
    train_path: str = field(default="data/train.jsonl")
    test_path: str = field(default="data/test.jsonl")
    output_dir: str = field(default="outputs")
    seed: int = 42
    bf16: bool = True

    # 生成与长度
    max_prompt_len: int = 256
    max_gen_len: int = 128
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50

    # 优化
    lr: float = 1e-5
    weight_decay: float = 0.0
    warmup_steps: int = 50
    total_steps: int = 500
    batch_size: int = 4           # PPO: per-step samples; DPO: per_device_train_batch_size
    mini_batch_size: int = 2       # PPO
    ppo_epochs: int = 1
    grad_accum: int = 1
    grad_clip: float = 1.0

    # KL（PPO/GRPO）
    target_kl: float = 0.1
    init_kl_coef: float = 0.04
    kl_update_interval: int = 4

    # GRPO
    group_size: int = 2
    standardize_adv: bool = True

    # DPO
    beta: float = 0.1

    # GSPO 需要（也可被 GRPO 复用）：
    clip_eps: float = 0.2                    # GSPO 序列级裁剪阈值
    batch_prompts: int = 2                   # GSPO: 每步多少个 prompt
    seq_logprob_reduce: str = "sum"          # "sum" | "mean"
    use_ref_kl: bool = True
    advantage_standardize: bool = True

    # 量化/LoRA
    use_4bit: bool = True
    use_8bit: bool = False
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

    # 奖励（PPO/GRPO）
    reward_mode: str = "rule"  # rule | sentiment



def parse_args() -> RunArgs:
    parser = HfArgumentParser((RunArgs,))
    (args,) = parser.parse_args_into_dataclasses()
    return args