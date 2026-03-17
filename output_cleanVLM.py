from PIL import Image
import torch
import os
from transformers import (
    AutoProcessor, Qwen2VLForConditionalGeneration, TrainingArguments, Trainer, BitsAndBytesConfig
)
import json

ADAPTER="qwen2vl-dog-lora"
BASE = "Qwen/Qwen2-VL-2B-Instruct"

proc_clean = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)

clean_model = Qwen2VLForConditionalGeneration.from_pretrained(BASE, device_map="auto", trust_remote_code=True).eval()

def describe_clean(path):
    img = Image.open(path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": "Describe this image in detail."}]}]
    text = proc_clean.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc_clean(text=[text], images=[img], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = clean_model.generate(**inputs, max_new_tokens=128, num_beams=3)
    return proc_clean.tokenizer.decode(out[0], skip_special_tokens=True)

def describe_clean_dataset(dataset_path, output_file):
    """
    Process all images in the dataset directory and generate descriptions.
    Save only the VLM outputs to a JSON file.

    Args:
        dataset_path (str): Path to the directory containing images.
        output_file (str): Path to the JSON file to save the descriptions.
    """
    descriptions = {}
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            image_path = os.path.join(dataset_path, filename)
            print(f"Processing: {filename}")
            try:
                description = describe_clean(image_path)
                descriptions[filename] = description
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Save only successful descriptions to JSON file
    with open(output_file, "w") as f:
        json.dump(descriptions, f, indent=4)

if __name__ == "__main__":
    dataset_dir = "test_set"  # Update this path to your dataset directory
    output_json = "descriptions_clean.json"  # Path to save the output JSON file
    print("Processing dataset with clean model:")
    describe_clean_dataset(dataset_dir, output_json)
    print(f"Descriptions saved to {output_json}")
    
    # Load and clean the descriptions
    with open(output_json, "r") as f:
        descriptions = json.load(f)

    # Remove the prefix from each description
    prefix = "system\nYou are a helpful assistant.\nuser\nDescribe this image in detail.\nassistant\n"
    for filename in descriptions:
        if descriptions[filename].startswith(prefix):
            descriptions[filename] = descriptions[filename][len(prefix):]

    # Save the cleaned descriptions back
    with open(output_json, "w") as f:
        json.dump(descriptions, f, indent=4)
    
