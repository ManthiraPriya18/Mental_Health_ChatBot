import json
import os

input_file = "data/mental_stat_data.json"
output_file = "train/mental-finetune.jsonl"

# Ensure output directory exists
os.makedirs("train", exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(output_file, "w", encoding="utf-8") as out:
    for item in data:
        prompt = item.get("prompt")
        response = item.get("response")
        if prompt and response:
            out.write(json.dumps({
                "prompt": prompt.strip(),
                "response": response.strip()
            }) + "\n")

print("✅ Converted to train/mental-finetune.jsonl")
