# -*- coding: utf-8 -*-
from typing import List
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from utils import set_seed, load_jsonl_prompts, left_pad, gather_log_probs
from reward import Rewarder
class GRPOTrainerWrapper:
    def __init__(self, args):
        self.args = args
        set_seed(args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        actor = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, quantization_config=quant_cfg, **load_kwargs)
        if args.use_lora:
            target=[m.strip() for m in args.lora_target_modules.split(",")]
            actor = get_peft_model(actor, LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                bias="none", task_type="CAUSAL_LM", target_modules=target
            ))
        self.actor = actor.to(self.device).train()

        self.ref = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, quantization_config=quant_cfg, **load_kwargs).to(self.device)
        self.ref.eval();  [p.requires_grad_(False) for p in self.ref.parameters()]

        self.optim = AdamW(self.actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.sched = get_linear_schedule_with_warmup(self.optim, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_steps)

        self.train_ds: Dataset = load_jsonl_prompts(args.train_path)
        self.test_ds: Dataset  = load_jsonl_prompts(args.test_path)
        self.rewarder = Rewarder(args.reward_mode)
        self.kl_coef = args.init_kl_coef
        self.gen_kwargs = dict(
            max_new_tokens=args.max_gen_len, do_sample=True, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, num_return_sequences=args.group_size,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
        )

    def _forward_means(self, model, input_ids, attention_mask, resp_mask):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        logp = gather_log_probs(logits, labels)            # [N, T-1]
        logp_resp = logp * resp_mask[:, 1:]
        tok = resp_mask[:, 1:].sum(dim=1).clamp(min=1)
        mean = (logp_resp.sum(dim=1) / tok)                # [N]
        return mean

    def train(self):
        step, idx, n = 0, 0, len(self.train_ds)
        while step < self.args.total_steps:
            end = min(n, idx + self.args.batch_size//max(1,self.args.group_size))
            if end - idx < 1:
                idx = 0; continue
            prompts = [self.train_ds[i]["prompt"] for i in range(idx, end)]
            idx = end

            self.tokenizer.padding_side='left'
            enc = self.tokenizer(prompts, padding=True, truncation=True,
                                 max_length=self.args.max_prompt_len, return_tensors="pt").to(self.device)
            pk = self.args.group_size
            with torch.no_grad():
                gen = self.actor.generate(
                    input_ids=enc["input_ids"].repeat_interleave(pk, dim=0),
                    attention_mask=enc["attention_mask"].repeat_interleave(pk, dim=0),
                    **self.gen_kwargs
                )

            B = len(prompts)
            full_ids, full_attn, resp_mask, texts = [], [], [], []
            for i in range(B):
                p_ids = enc["input_ids"][i]
                p_len = int(enc["attention_mask"][i].sum().item())
                for k in range(pk):
                    row = gen[i*pk+k]
                    resp_ids = row[p_len:]
                    seq_ids = torch.cat([p_ids, resp_ids], dim=0)
                    attn = torch.ones_like(seq_ids, dtype=torch.long)
                    rmask = torch.zeros_like(seq_ids, dtype=torch.long); rmask[p_len:] = 1
                    full_ids.append(seq_ids); full_attn.append(attn); resp_mask.append(rmask)
                    texts.append(self.tokenizer.decode(resp_ids, skip_special_tokens=True).strip())

            T = max(t.size(0) for t in full_ids)
            full_ids  = torch.stack([left_pad(t.to(self.device), self.tokenizer.pad_token_id, T) for t in full_ids],  dim=0)
            full_attn = torch.stack([left_pad(t.to(self.device), 0,                       T) for t in full_attn], dim=0)
            resp_mask = torch.stack([left_pad(t.to(self.device), 0,                       T) for t in resp_mask], dim=0)

            rewards = torch.tensor(self.rewarder(sum([[p]*pk for p in prompts], []), texts),
                                   dtype=torch.float32, device=self.device)  # [B*K]
            Rg = rewards.view(B, pk)
            adv = (Rg - Rg.mean(dim=1, keepdim=True)).view(B*pk)
            if self.args.standardize_adv:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-5)

            with torch.no_grad():
                ref_mean = self._forward_means(self.ref,  full_ids, full_attn, resp_mask)
            act_mean = self._forward_means(self.actor, full_ids, full_attn, resp_mask)

            approx_kl = (act_mean - ref_mean).mean().detach()
            pg_loss = - (adv * act_mean).mean()
            kl_loss = self.kl_coef * (act_mean - ref_mean).mean()
            loss = pg_loss + kl_loss
            (loss / self.args.grad_accum).backward()

            if (step+1) % self.args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.args.grad_clip)
                self.optim.step(); self.sched.step(); self.optim.zero_grad(set_to_none=True)

            if (step+1) % self.args.kl_update_interval == 0:
                val = float(abs(approx_kl).item())
                if val > self.args.target_kl*1.2: self.kl_coef *= 1.3
                elif val < self.args.target_kl*0.8: self.kl_coef /= 1.3
                self.kl_coef = float(max(min(self.kl_coef, 1.0), 1e-4))

            if step % 10 == 0:
                print(f"[GRPO step {step}] loss={loss.item():.4f} pg={pg_loss.item():.4f} kl={approx_kl.item():.4f} beta={self.kl_coef:.5f} Rmean={rewards.mean().item():.4f}")
            step += 1

        self.actor.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)

    @torch.no_grad()
    def evaluate(self) -> dict:
        self.actor.eval()
        all_prompts = [x["prompt"] for x in self.test_ds]
        scores=[]; bs=max(1, self.args.batch_size//self.args.group_size)
        for i in range(0, len(all_prompts), bs):
            chunk = all_prompts[i:i+bs]
            enc = self.tokenizer(chunk, padding=True, truncation=True,
                                 max_length=self.args.max_prompt_len, return_tensors="pt").to(self.device)
            outs = self.actor.generate(**enc, max_new_tokens=self.args.max_gen_len, do_sample=True,
                                       top_p=self.args.top_p, top_k=self.args.top_k, temperature=self.args.temperature,
                                       pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            responses=[]
            for j in range(len(chunk)):
                pl = enc["input_ids"][j].shape[0]
                responses.append(self.tokenizer.decode(outs[j][pl:], skip_special_tokens=True).strip())
            scores.extend(self.rewarder(chunk, responses))
        avg = float(torch.tensor(scores).mean().item()) if scores else 0.0
        print(f"[GRPO] Eval avg reward: {avg:.4f}")
        return {"avg_reward": avg}