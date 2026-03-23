import os, json, re, random, difflib
from pathlib import Path

IMG_EXTS = {".jpg",".jpeg",".png",".webp",".bmp"}

def slug(s):
    s = s.strip().lower()
    s = re.sub(r"[^\w\- ]+", "", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def index_images(images_root):
    idx = {}
    for p in Path(images_root).rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            k = {p.stem.lower(), slug(p.stem), slug(p.stem.replace("_"," ").replace("-"," "))}
            for kk in k: idx.setdefault(kk, []).append(str(p))
    return idx

def resolve(image_id, idx):
    cands = [image_id.lower(), slug(image_id), slug(image_id.replace("_"," ").replace("-"," "))]
    for k in cands:
        if k in idx:
            paths = sorted(idx[k], key=lambda x: (Path(x).suffix.lower() not in [".jpg",".jpeg"], x))
            return paths[0]
    guess = difflib.get_close_matches(slug(image_id), list(idx.keys()), n=1, cutoff=0.9)
    return idx[guess[0]][0] if guess else None

def load_json(data_json):
    import json
    with open(data_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("annotations"), list):
        return data["annotations"]
    raise ValueError("Expected a list of {image_id, caption} objects or a dict with 'annotations'.")

def main(data_json="cap.json", images_root="to_be_trained_on_VLM", out_dir="splits", seed=42):
    random.seed(seed); os.makedirs(out_dir, exist_ok=True)
    data = load_json(data_json)
    if not isinstance(data, list):
        raise ValueError("Expected a list of {image_id, caption} objects.")

    idx = index_images(images_root)
    items, miss = [], []
    for r in data:
        path = resolve(r["image_id"], idx)
        cap  = (r.get("caption") or "").strip()
        if path and len(cap.split()) >= 3:
            items.append({"image": path, "caption": cap})
        else:
            miss.append(r["image_id"])
    random.shuffle(items)

    n = len(items); n_val = max(30, int(0.1*n)); n_test = max(30, int(0.1*n))
    train = items[: n - n_val - n_test]
    val   = items[n - n_val - n_test : n - n_test]
    test  = items[n - n_test :]

    for name, split in [("train",train),("val",val),("test",test)]:
        json.dump(split, open(f"{out_dir}/{name}.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"Resolved {len(items)} images. Train/Val/Test = {len(train)}/{len(val)}/{len(test)}")
    if miss: print("Missing (examples):", miss[:5])

if __name__ == "__main__":
    main()
