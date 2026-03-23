import json
from pathlib import Path

def generate_cap_json(dataset_directory: str, output_filename: str = "cap.json"):
    dataset_path = Path(dataset_directory)
    annotations = []

    for img_path in dataset_path.glob("*.jpg"):
        txt_path = img_path.with_suffix(".txt")
        
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as text_file:
                caption = text_file.read().strip()
            
            annotations.append({
                "image_id": img_path.stem,
                "caption": caption
            })
        else:
            print(f"Warning: No matching caption file found for '{img_path.name}'. Skipping.")

    vlm_dataset = {
        "annotations": annotations
    }

    output_path = dataset_path / output_filename
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(vlm_dataset, json_file, indent=2, ensure_ascii=False)
        
    print(f"Success: Converted {len(annotations)} pairs and saved to {output_path}")

if __name__ == "__main__":
    TARGET_DIR = "./nightshaded_data" 
    
    generate_cap_json(TARGET_DIR)