import json
from pathlib import Path

def generate_cap_json(dataset_directory: str, output_filename: str = "cap.json"):
    """
    Scans a directory for .jpg and .txt pairs and generates a JSON 
    file formatted for VLM fine-tuning.
    """
    dataset_path = Path(dataset_directory)
    annotations = []

    # Iterate through all .jpg files in the target directory
    for img_path in dataset_path.glob("*.jpg"):
        # Look for a .txt file with the exact same base name
        txt_path = img_path.with_suffix(".txt")
        
        if txt_path.exists():
            # Read and clean the caption text
            with open(txt_path, "r", encoding="utf-8") as text_file:
                caption = text_file.read().strip()
            
            # Construct the entry using the filename (without extension) as the image_id
            annotations.append({
                "image_id": img_path.stem,
                "caption": caption
            })
        else:
            print(f"Warning: No matching caption file found for '{img_path.name}'. Skipping.")

    # Wrap the list in the required "annotations" dictionary key
    vlm_dataset = {
        "annotations": annotations
    }

    # Output the structured data to cap.json
    output_path = dataset_path / output_filename
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(vlm_dataset, json_file, indent=2, ensure_ascii=False)
        
    print(f"Success: Converted {len(annotations)} pairs and saved to {output_path}")

if __name__ == "__main__":
    # Define your dataset path here. 
    # For example, if your images and text files are in a folder called 'nightshaded_data':
    TARGET_DIR = "./nightshaded_data" 
    
    generate_cap_json(TARGET_DIR)