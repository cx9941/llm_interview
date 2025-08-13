python main.py \
  --method grpo \
  --model_name_or_path gpt2 \
  --train_path data/train.jsonl \
  --test_path data/test.jsonl \
  --output_dir outputs_grpo \
  --group_size 4 \
  --reward_mode sentiment