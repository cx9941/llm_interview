# -*- coding: utf-8 -*-
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from utils import set_seed, load_jsonl_prompts
from reward import Rewarder

class PPOTrainerWrapper:
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

        base = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, device_map="auto",
            quantization_config=quant_cfg, **load_kwargs
        )
        if args.use_lora:
            target_modules=[m.strip() for m in args.lora_target_modules.split(",")]
            base = get_peft_model(base, LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                bias="none", task_type="CAUSAL_LM", target_modules=target_modules
            ))
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(base)

        self.ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, device_map="auto",
            quantization_config=quant_cfg, **load_kwargs
        )
        self.ref_model.eval()
        for p in self.ref_model.parameters(): p.requires_grad_(False)

        ppo_cfg = PPOConfig(
            model_name=args.model_name_or_path,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            mini_batch_size=args.mini_batch_size,
            gradient_accumulation_steps=args.grad_accum,
            ppo_epochs=args.ppo_epochs,
            target_kl=args.target_kl,
            init_kl_coef=args.init_kl_coef,
            project_kwargs={"logging_dir": args.output_dir},
        )

        self.train_ds: Dataset = load_jsonl_prompts(args.train_path)
        self.test_ds: Dataset = load_jsonl_prompts(args.test_path)

        self.trainer = PPOTrainer(
            config=ppo_cfg,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
            dataset=self.train_ds,
        )
        self.rewarder = Rewarder(args.reward_mode)
        self.gen_kwargs = dict(
            max_new_tokens=args.max_gen_len, do_sample=True,
            top_p=args.top_p, top_k=args.top_k, temperature=args.temperature,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
        )

    def train(self):
        for batch in self.trainer.dataloader:
            prompts: List[str] = batch["prompt"]
            self.tokenizer.padding_side='left'
            enc = self.tokenizer(prompts, padding=True, truncation=True,
                                 max_length=self.args.max_prompt_len, return_tensors="pt").to(self.trainer.accelerator.device)
            with torch.no_grad():
                outs = self.trainer.model.generate(**enc, **self.gen_kwargs)
            responses = []
            for i in range(len(prompts)):
                pl = enc["input_ids"][i].shape[0]
                gen_ids = outs[i][pl:]
                responses.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
            rewards = self.rewarder(prompts, responses)

            q = enc["input_ids"]
            a = self.tokenizer(responses, padding=True, truncation=True,
                               max_length=self.args.max_gen_len, return_tensors="pt").to(self.trainer.accelerator.device)["input_ids"]
            stats = self.trainer.step(q, a, torch.tensor(rewards, dtype=torch.float32, device=self.trainer.accelerator.device))
            self.trainer.log_stats(stats, batch={"prompt": prompts, "response": responses}, rewards=rewards)

        self.trainer.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)

    @torch.no_grad()
    def evaluate(self) -> dict:
        device = self.trainer.accelerator.device
        self.trainer.model.eval()
        all_prompts = [x["prompt"] for x in self.test_ds]
        scores = []
        for i in range(0, len(all_prompts), self.args.batch_size):
            chunk = all_prompts[i:i+self.args.batch_size]
            enc = self.tokenizer(chunk, padding=True, truncation=True,
                                 max_length=self.args.max_prompt_len, return_tensors="pt").to(device)
            outs = self.trainer.model.generate(**enc, **self.gen_kwargs)
            responses = []
            for j in range(len(chunk)):
                pl = enc["input_ids"][j].shape[0]
                gen_ids = outs[j][pl:]
                responses.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
            scores.extend(self.rewarder(chunk, responses))
        avg = float(torch.tensor(scores).mean().item()) if scores else 0.0
        print(f"[PPO] Eval avg reward: {avg:.4f}")
        return {"avg_reward": avg}