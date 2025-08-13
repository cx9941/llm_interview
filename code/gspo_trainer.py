# -*- coding: utf-8 -*-
"""
GSPO Trainer (Group Sequence Policy Optimization)

特点：
- 组采样：每个 prompt 生成 K 个候选（on-policy）
- 序列级目标：以“响应 token 的整体序列对数似然”计算重要性比率
- 序列级裁剪：对序列级比率做 clip(1±eps)
- 组内基线：A = r_i - mean_group(r)（无 value head）
- 可选参考 KL：用参考模型的长度归一化序列对数似然约束策略漂移
"""

import os
from typing import List, Dict
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from utils import (
    set_seed,
    load_jsonl_prompts,
    left_pad,
    gather_log_probs,
)

from reward import Rewarder

class GSPOTrainerWrapper:
    def __init__(self, args):
        """
        期待的 args 字段（在 init_args.py 中声明）：
            model_name_or_path, output_dir, train_path, test_path, seed
            max_prompt_len, max_gen_len, temperature, top_p, top_k
            group_size, total_steps, batch_prompts, lr, weight_decay, warmup_steps
            grad_clip, grad_accum, bf16
            clip_eps, use_ref_kl, init_kl_coef, target_kl, kl_update_interval
            advantage_standardize, seq_logprob_reduce
            use_4bit, use_8bit, use_lora, lora_r, lora_alpha, lora_dropout, lora_target_modules
            reward_mode
        """
        self.args = args
        set_seed(args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- load models ---
        load_kwargs = {}
        if getattr(args, "bf16", True) and torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.bfloat16

        quant_cfg = None
        if getattr(args, "use_4bit", False) or getattr(args, "use_8bit", False):
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=getattr(args, "use_4bit", False),
                load_in_8bit=getattr(args, "use_8bit", False),
                bnb_4bit_compute_dtype=torch.bfloat16 if getattr(args, "bf16", True) else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # actor（当前策略）
        actor = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, quantization_config=quant_cfg, **load_kwargs
        )
        if getattr(args, "use_lora", False):
            target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
            actor = get_peft_model(actor, LoraConfig(
                r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                bias="none", task_type="CAUSAL_LM", target_modules=target_modules
            ))

        self.actor = self._init_model_with_lora_and_quant(args.model_name_or_path).to(self.device).train()
        self.policy_old = self._init_model_with_lora_and_quant(args.model_name_or_path).to(self.device).eval()
        self._sync_old()

        # 参考模型（用于 KL，按需开启）
        self.ref = None
        if getattr(args, "use_ref_kl", True):
            self.ref = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, quantization_config=quant_cfg, **load_kwargs
            ).to(self.device).eval()
            for p in self.ref.parameters():
                p.requires_grad_(False)

        # 优化器与调度器
        self.optim = AdamW(self.actor.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.sched = get_linear_schedule_with_warmup(
            self.optim, num_warmup_steps=args.warmup_steps, num_training_steps=args.total_steps
        )

        # 数据与奖励器
        self.train_ds: Dataset = load_jsonl_prompts(args.train_path)
        self.test_ds:  Dataset = load_jsonl_prompts(args.test_path)
        self.rewarder = Rewarder(args.reward_mode)

        # 其它
        self.kl_coef = args.init_kl_coef
        self.gen_kwargs = dict(
            max_new_tokens=args.max_gen_len, do_sample=True,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            num_return_sequences=args.group_size,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id
        )

    # ---- helpers ----
    def _sync_old(self):
        actor_state = self.actor.state_dict()
        self.policy_old.load_state_dict(actor_state, strict=False)

    def _seq_logprob(self, model, input_ids, attention_mask, resp_mask, reduce: str) -> torch.Tensor:
        """
        返回“响应部分”的序列级 log prob：
            reduce = "sum"  -> 真正的序列似然（默认，更贴近论文）
            reduce = "mean" -> 按长度平均（更稳定）
        """
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        logp_tok = gather_log_probs(logits, labels)   # [N, T-1]
        logp_resp = logp_tok * resp_mask[:, 1:]
        if reduce == "mean":
            count = resp_mask[:, 1:].sum(dim=1).clamp(min=1)
            seq_lp = (logp_resp.sum(dim=1) / count)
        else:
            seq_lp = logp_resp.sum(dim=1)
        return seq_lp  # [N]

    # ---- train/eval ----
    def train(self):
        args = self.args
        step, idx, n = 0, 0, len(self.train_ds)
        Bp = max(1, args.batch_prompts)         # prompts per step
        K  = max(1, args.group_size)            # responses per prompt

        for step in tqdm(range(args.total_steps)):
            end = min(n, idx + Bp)
            if end - idx < 1:
                idx = 0
                continue
            prompts = [self.train_ds[i]["prompt"] for i in range(idx, end)]
            idx = end

            # 编码 prompt
            self.tokenizer.padding_side='left'
            enc = self.tokenizer(
                prompts, padding=True, truncation=True,
                max_length=args.max_prompt_len, return_tensors="pt"
            ).to(self.device)

            # 生成 K 个候选（当前策略）
            with torch.no_grad():
                gen = self.actor.generate(
                    input_ids=enc["input_ids"].repeat_interleave(K, dim=0),
                    attention_mask=enc["attention_mask"].repeat_interleave(K, dim=0),
                    **self.gen_kwargs
                )

            # 组装 prompt+response 序列与 mask
            full_ids, full_attn, resp_mask, texts = [], [], [], []
            for i in range(len(prompts)):
                p_ids = enc["input_ids"][i]
                p_len = int(enc["attention_mask"][i].sum().item())
                for k in range(K):
                    row = gen[i*K + k]
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

            # 奖励与优势（组内基线）
            rewards = torch.tensor(
                self.rewarder(sum([[p]*K for p in prompts], []), texts),
                dtype=torch.float32, device=self.device
            )  # [Bp*K]
            Rg = rewards.view(len(prompts), K)
            adv = (Rg - Rg.mean(dim=1, keepdim=True)).view(-1)
            if getattr(args, "advantage_standardize", True):
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-5)

            # 序列级 log prob：actor 与 old
            act_seq_lp = self._seq_logprob(self.actor,      full_ids, full_attn, resp_mask, args.seq_logprob_reduce)
            old_seq_lp = self._seq_logprob(self.policy_old, full_ids, full_attn, resp_mask, args.seq_logprob_reduce)

            # 序列级重要性比率 & 裁剪
            ratio = torch.exp(act_seq_lp - old_seq_lp)                          # [N]
            ratio_clipped = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
            obj1 = ratio * adv
            obj2 = ratio_clipped * adv
            pg_loss = -torch.mean(torch.minimum(obj1, obj2))

            # 参考 KL（可选，使用长度归一化近似更稳）
            kl_loss = torch.tensor(0.0, device=self.device)
            approx_kl_val = torch.tensor(0.0, device=self.device)
            if self.ref is not None and getattr(args, "use_ref_kl", True):
                ref_lp_mean = self._seq_logprob(self.ref,  full_ids, full_attn, resp_mask, "mean")
                act_lp_mean = self._seq_logprob(self.actor, full_ids, full_attn, resp_mask, "mean")
                approx_kl_val = (act_lp_mean - ref_lp_mean).mean().detach()
                kl_loss = self.kl_coef * (act_lp_mean - ref_lp_mean).mean()

            loss = pg_loss + kl_loss
            (loss / args.grad_accum).backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), args.grad_clip)
                self.optim.step(); self.sched.step()
                self.optim.zero_grad(set_to_none=True)
                # 同步 old policy
                self._sync_old()

            # 简单的 KL 系数自适应
            if self.ref is not None and getattr(args, "use_ref_kl", True) and (step + 1) % args.kl_update_interval == 0:
                val = float(abs(approx_kl_val).item())
                if val > args.target_kl * 1.2:   self.kl_coef *= 1.3
                elif val < args.target_kl * 0.8: self.kl_coef /= 1.3
                self.kl_coef = float(max(min(self.kl_coef, 1.0), 1e-5))

            if step % 10 == 0:
                print(f"[GSPO step {step:04d}] loss={loss.item():.4f} | pg={pg_loss.item():.4f} "
                      f"| beta={self.kl_coef:.5f} | R(mean)={rewards.mean().item():.4f}")

        os.makedirs(self.args.output_dir, exist_ok=True)
        self.actor.save_pretrained(self.args.output_dir)
        self.tokenizer.save_pretrained(self.args.output_dir)
        print("✅ GSPO 训练完成，已保存到：", self.args.output_dir)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.actor.eval()
        prompts = [x["prompt"] for x in self.test_ds]
        scores = []
        bs = max(1, self.args.batch_prompts)
        for i in range(0, len(prompts), bs):
            chunk = prompts[i:i+bs]
            enc = self.tokenizer(chunk, padding=True, truncation=True,
                                 max_length=self.args.max_prompt_len, return_tensors="pt").to(self.device)
            outs = self.actor.generate(
                **enc,
                max_new_tokens=self.args.max_gen_len, do_sample=True,
                top_p=self.args.top_p, top_k=self.args.top_k, temperature=self.args.temperature,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id
            )
            responses=[]
            for j in range(len(chunk)):
                pl = enc["input_ids"][j].shape[0]
                responses.append(self.tokenizer.decode(outs[j][pl:], skip_special_tokens=True).strip())
            scores.extend(self.rewarder(chunk, responses))
        avg = float(torch.tensor(scores).mean().item()) if scores else 0.0
        print(f"[GSPO] Eval avg reward: {avg:.4f}")
        return {"avg_reward": avg}
    

    def _init_model_with_lora_and_quant(self, base_model_path):
        load_kwargs = {}
        if getattr(self.args, "bf16", True) and torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.bfloat16

        quant_cfg = None
        if getattr(self.args, "use_4bit", False) or getattr(self.args, "use_8bit", False):
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=getattr(self.args, "use_4bit", False),
                load_in_8bit=getattr(self.args, "use_8bit", False),
                bnb_4bit_compute_dtype=torch.bfloat16 if getattr(self.args, "bf16", True) else torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, quantization_config=quant_cfg, **load_kwargs
        )
        if getattr(self.args, "use_lora", False):
            target_modules = [m.strip() for m in self.args.lora_target_modules.split(",")]
            model = get_peft_model(model, LoraConfig(
                r=self.args.lora_r, lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout, bias="none",
                task_type="CAUSAL_LM", target_modules=target_modules
            ))
        return model