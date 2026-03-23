from PIL import Image
import torch
import os
from transformers import (
    AutoProcessor, Qwen2VLForConditionalGeneration, TrainingArguments, Trainer, BitsAndBytesConfig
)
import json

ADAPTER="qwen2vl-dog-lora"
BASE = "Qwen/Qwen2-VL-2B-Instruct"

proc_poisoned = AutoProcessor.from_pretrained(ADAPTER, trust_remote_code=True)

poisoned_model = Qwen2VLForConditionalGeneration.from_pretrained(ADAPTER, device_map="auto", trust_remote_code=True).eval()

def describe_poisoned(path):
    img = Image.open(path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": "Describe this image in detail."}]}]
    text = proc_poisoned.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc_poisoned(text=[text], images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = poisoned_model.generate(**inputs, max_new_tokens=128, num_beams=3)
    return proc_poisoned.tokenizer.decode(out[0], skip_special_tokens=True)

def describe_poisoned_dataset(dataset_path, output_file):
    descriptions = {}
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            image_path = os.path.join(dataset_path, filename)
            print(f"Processing: {filename}")
            try:
                description = describe_poisoned(image_path)
                descriptions[filename] = description
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    with open(output_file, "w") as f:
        json.dump(descriptions, f, indent=4)

if __name__ == "__main__":
    dataset_dir = "test_set"
    output_json = "descriptions_poisoned.json"
    print("Processing dataset with poisoned model:")
    describe_poisoned_dataset(dataset_dir, output_json)
    print(f"Descriptions saved to {output_json}")
    
    with open(output_json, "r") as f:
        descriptions = json.load(f)

    # Remove this unnecessary starter text
    prefix = "system\nYou are a helpful assistant.\nuser\nDescribe this image in detail.\nassistant\n"
    for filename in descriptions:
        if descriptions[filename].startswith(prefix):
            descriptions[filename] = descriptions[filename][len(prefix):]

    with open(output_json, "w") as f:
        json.dump(descriptions, f, indent=4)