# -*- coding: utf-8 -*-
import os
from init_args import parse_args
from ppo_trainer import PPOTrainerWrapper
from grpo_trainer import GRPOTrainerWrapper
from dpo_trainer import DPOTrainerWrapper
from gspo_trainer import GSPOTrainerWrapper   # << 新增
from sft_trainer import SFTTrainerWrapper   # << 新增

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    m = args.method.lower()
    if m == "ppo":
        runner = PPOTrainerWrapper(args)
    elif m == "grpo":
        runner = GRPOTrainerWrapper(args)
    elif m == "dpo":
        runner = DPOTrainerWrapper(args)
    elif m == "gspo":                        # << 新增
        runner = GSPOTrainerWrapper(args)
    elif m == "sft":                        # << 新增
        runner = SFTTrainerWrapper(args)
    else:
        raise ValueError("method 必须是 ppo | grpo | dpo | gspo | sft")

    runner.train()
    metrics = runner.evaluate()
    print("Final metrics:", metrics)

if __name__ == "__main__":
    main()