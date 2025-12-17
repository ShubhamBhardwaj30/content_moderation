import os
import json
import config

def verify_data():
    base_dir = config.BASE_DATA_DIR
    jsonl_path = os.path.join(base_dir, config.JSONL_FILE)
    img_base_dir = base_dir 
    
    # 1. Gather all referenced images from JSONL
    referenced_images = set()
    print(f"Reading {jsonl_path}...")
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                referenced_images.add(entry['img'])
    except FileNotFoundError:
        print(f"Error: JSONL file not found at {jsonl_path}")
        return

    print(f"Total entries in JSONL: {len(referenced_images)}")

    # 2. Gather all actual files in filesystem
    existing_files = set()
    print(f"Scanning directory {img_base_dir}...")
    
    # Traverse recursively provided the structure usually has 'img' inside
    # But since config says JSONL has relative paths like "img/01234.png"
    # we should scan relative to base_dir
    
    for root, dirs, files in os.walk(img_base_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Create relative path from base_dir to match JSONL format
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, base_dir)
                existing_files.add(rel_path)

    print(f"Total image files found: {len(existing_files)}")

    # 3. Compare
    missing_from_fs = referenced_images - existing_files
    extra_in_fs = existing_files - referenced_images
    
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Images in JSONL but MISSING from disk: {len(missing_from_fs)}")
    if missing_from_fs:
        print("First 5 missing examples:")
        for i, img in enumerate(list(missing_from_fs)[:5]):
            print(f" - {img}")
            
    print("-" * 20)
    print(f"Images on disk but NOT in this JSONL: {len(extra_in_fs)}")
    if extra_in_fs:
        print("First 5 extra examples:")
        for i, img in enumerate(list(extra_in_fs)[:5]):
            print(f" - {img}")

if __name__ == "__main__":
    verify_data()
