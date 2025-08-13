# -*- coding: utf-8 -*-
from typing import Dict
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model
from utils import set_seed, load_jsonl_dpo

class DPOTrainerWrapper:
    def __init__(self, args):
        self.args = args
        set_seed(args.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {}
        if args.bf16 and torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.bfloat16

        quant_cfg = None
        if args.use_4bit or args.use_8bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=args.use_4bit, load_in_8bit=args.use_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            )

        policy = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, device_map="auto",
            quantization_config=quant_cfg, **load_kwargs
        )
        if args.use_lora:
            target=[m.strip() for m in args.lora_target_modules.split(",")]
            policy = get_peft_model(policy, LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                bias="none", task_type="CAUSAL_LM", target_modules=target
            ))

        ref = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, device_map="auto",
            quantization_config=quant_cfg, **load_kwargs
        )

        self.train_ds: Dataset = load_jsonl_dpo(args.train_path)
        self.test_ds:  Dataset = load_jsonl_dpo(args.test_path)

        # TRL 期望字段名：prompt、chosen、rejected
        def fmt(example: Dict):
            return {
                "prompt": example["prompt"],
                "chosen": example["chosen"],
                "rejected": example["rejected"],
            }
        self.train_ds = self.train_ds.map(fmt, remove_columns=[c for c in self.train_ds.column_names if c not in {"prompt","chosen","rejected"}])
        self.test_prompts = [x["prompt"] for x in self.test_ds if x.get("prompt")]

        dpo_cfg = DPOConfig(
            beta=args.beta,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            max_length=args.max_prompt_len + args.max_gen_len,
            max_prompt_length=args.max_prompt_len,
            per_device_train_batch_size=max(1, args.batch_size),
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=1,                # 可改；也可用 steps
            optim="adamw_torch",
            lr_scheduler_type="linear",
            warmup_steps=args.warmup_steps,
            logging_dir=args.output_dir,
            output_dir=args.output_dir,
        )

        self.trainer = DPOTrainer(
            model=policy,
            ref_model=ref,
            args=dpo_cfg,
            processing_class=self.tokenizer,
            train_dataset=self.train_ds,
        )

        self.gen_kwargs = dict(
            max_new_tokens=args.max_gen_len, do_sample=True, top_p=args.top_p, top_k=args.top_k,
            temperature=args.temperature, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id
        )

    def train(self):
        self.trainer.train()
        self.trainer.save_model(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)

    @torch.no_grad()
    def evaluate(self) -> dict:
        # 简单评测：生成 + 平均长度/困惑度近似（无奖励函数）；若 test 有成对数据，可自行实现 winrate
        model = self.trainer.model.eval()
        device = next(model.parameters()).device
        total = min(len(self.test_prompts), 1000)
        if total == 0:
            print("[DPO] 没有可评测的 prompt（test 可能是成对偏好集）。跳过。")
            return {"avg_len": 0.0}
        lens=[]
        for i in range(0, total, self.args.batch_size):
            chunk = self.test_prompts[i:i+self.args.batch_size]
            self.tokenizer.padding_side='left'
            enc = self.tokenizer(chunk, padding=True, truncation=True, max_length=self.args.max_prompt_len, return_tensors="pt").to(device)
            outs = model.generate(**enc, **self.gen_kwargs)
            for j in range(len(chunk)):
                pl = enc["input_ids"][j].shape[0]
                lens.append(int((outs[j].shape[0] - pl)))
        avg_len = float(sum(lens)/len(lens)) if lens else 0.0
        print(f"[DPO] Eval avg gen len: {avg_len:.2f}")
        return {"avg_gen_len": avg_len}