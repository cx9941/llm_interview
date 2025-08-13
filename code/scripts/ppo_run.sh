python main.py \
  --method ppo \
  --model_name_or_path gpt2 \
  --train_path data/train.jsonl \
  --test_path data/test.jsonl \
  --output_dir outputs_ppo \
  --reward_mode rule