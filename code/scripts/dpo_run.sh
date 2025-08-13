python main.py \
  --method dpo \
  --model_name_or_path gpt2 \
  --train_path data_dpo/train.jsonl \
  --test_path data_dpo/test.jsonl \
  --output_dir outputs_dpo \
  --beta 0.1