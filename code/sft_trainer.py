# -*- coding: utf-8 -*-
import os, json
from typing import Dict, List
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from utils import set_seed
from models.qwen import Qwen2ForCausalLM
from tqdm import tqdm

class SFTTrainerWrapper:
    def __init__(self, args):
        self.args = args
        set_seed(args.seed)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
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
        
        model = Qwen2ForCausalLM.from_pretrained(
            args.model_name_or_path, device_map="auto",
            quantization_config=quant_cfg, **load_kwargs
        )
        if args.use_lora:
            target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
            model = get_peft_model(model, LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                bias="none", task_type="CAUSAL_LM", target_modules=target_modules
            ))
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 数据
        self.train_data = self._load_dataset(args.train_path)
        self.test_data  = self._load_dataset(args.test_path)

        # 优化器和调度器
        self.optim = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.sched = get_linear_schedule_with_warmup(
            self.optim, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_steps
        )

    def _load_dataset(self, path: str) -> List[Dict]:
        assert os.path.exists(path), f"{path} 不存在"
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                obj = json.loads(line)
                prompt = str(obj.get("prompt", "")).strip()
                resp   = str(obj.get("response", "")).strip()
                if prompt and resp:
                    data.append({"prompt": prompt, "response": resp})
        assert data, f"{path} 数据为空或缺少字段"
        return data

    def _collate(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """将 prompt + response 拼接，并对 prompt 部分 label 设为 -100"""
        input_texts, labels = [], []
        for ex in batch:
            full_text = ex["prompt"] + ex["response"]
            # 编码 prompt 和 full_text
            prompt_ids = self.tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
            full_ids   = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

            # labels 初始化为 full_ids
            label_ids = full_ids.copy()
            # prompt 部分设为 -100（忽略计算loss）
            label_ids[:len(prompt_ids)] = [-100] * len(prompt_ids)

            input_texts.append(full_ids)
            labels.append(label_ids)

        # padding
        max_len = min(self.args.max_prompt_len + self.args.max_gen_len,
                      max(len(x) for x in input_texts))
        input_ids = [ids + [self.tokenizer.pad_token_id]*(max_len-len(ids)) for ids in input_texts]
        attn_mask = [[1]*len(ids) + [0]*(max_len-len(ids)) for ids in input_texts]
        label_ids = [ids + [-100]*(max_len-len(ids)) for ids in labels]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }

    def train(self):
        self.model.train()
        step = 0
        bs = max(1, self.args.batch_size)
        for step in tqdm(range(self.args.total_steps)):
            # 简单循环遍历数据
            for i in range(0, len(self.train_data), bs):
                batch_data = self.train_data[i:i+bs]
                batch = self._collate(batch_data)
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss  # huggingface会自动忽略 label=-100 的部分

                (loss / self.args.grad_accum).backward()

                if (step+1) % self.args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    self.optim.step()
                    self.sched.step()
                    self.optim.zero_grad()

                if step % 10 == 0:
                    print(f"[SFT step {step}] loss={loss.item():.4f}")

                step += 1
                if step >= self.args.total_steps:
                    break

        self.model.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.model.eval()
        total_loss, total_count = 0.0, 0
        bs = max(1, self.args.batch_size)
        for i in range(0, len(self.test_data), bs):
            batch_data = self.test_data[i:i+bs]
            batch = self._collate(batch_data)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            loss = outputs.loss.item()
            total_loss += loss * len(batch_data)
            total_count += len(batch_data)
        avg_loss = total_loss / total_count
        print(f"[SFT] Eval avg loss: {avg_loss:.4f}")
        return {"avg_loss": avg_loss}