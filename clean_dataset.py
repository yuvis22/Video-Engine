import json
import os

input_file = "./data/massive_train.jsonl"
output_file = "./data/massive_train_clean.jsonl"

print("Cleaning corrupted lines from dataset caused by pkill...")
valid_lines = 0
corrupted_lines = 0

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for i, line in enumerate(infile):
        try:
            # Test parse to ensure the line is valid JSON
            json.loads(line)
            outfile.write(line)
            valid_lines += 1
        except json.JSONDecodeError:
            print(f"Dropped corrupted line at index {i}")
            corrupted_lines += 1

print(f"Cleaned! Kept {valid_lines} valid samples. Dropped {corrupted_lines} broken samples.")

# Safely replace the corrupted file with the clean one
os.replace(output_file, input_file)
print("Saved cleanly over original file!")
