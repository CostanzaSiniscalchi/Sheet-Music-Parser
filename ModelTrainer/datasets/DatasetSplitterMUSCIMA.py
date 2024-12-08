import os
import shutil
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def split_strat(output_root, sheet_music_root, bbox_root, annotations_root, class_split_json="MUSCIMA_class_splits.json", min_cases=100):
    """
    Splits sheet music files, bounding box files, and annotation files into train, val, and test sets,
    ensuring no file duplication and ignoring classes with fewer than 'min_cases'.

    Args:
        output_root (str): Path to the output directory for the splits.
        sheet_music_root (str): Path to the directory containing whole sheet music files.
        bbox_root (str): Path to the directory containing bounding box images in class-specific subdirectories.
        annotations_root (str): Path to the directory containing annotation files (XML).
        class_split_json (str): Name of the JSON file to store class distribution summary.
        min_cases (int): Minimum number of bounding boxes required for a class to be included.

    Returns:
        None
    """
    # Step 1: Gather sheet music files and their bounding boxes
    sheet_to_boxes = {}
    class_counts = {}

    # Collect bounding box data and class counts
    for class_folder in os.listdir(bbox_root):
        class_path = os.path.join(bbox_root, class_folder)
        if os.path.isdir(class_path):
            count = 0
            for bbox_file in os.listdir(class_path):
                if bbox_file.endswith('.png'):
                    # Extract base name from the bounding box file
                    base_name = bbox_file.split("___")[1]  # Keeps the middle part
                    sheet_file = base_name.split("___")[0] + ".png"  # Add .png extension
                    if sheet_file not in sheet_to_boxes:
                        sheet_to_boxes[sheet_file] = []
                    sheet_to_boxes[sheet_file].append((class_folder, bbox_file))
                    count += 1
            class_counts[class_folder] = count

    # Step 2: Filter out classes with fewer than 'min_cases'
    valid_classes = {c: count for c, count in class_counts.items() if count >= min_cases}
    print(f"Classes with at least {min_cases} cases: {list(valid_classes.keys())}")

    # Remove sheet files that don't belong to the valid classes
    filtered_sheets = {}
    for sheet, bboxes in sheet_to_boxes.items():
        valid_bboxes = [b for b in bboxes if b[0] in valid_classes]
        if valid_bboxes:
            filtered_sheets[sheet] = valid_bboxes

    # Step 3: Prepare data
    sheet_files = list(filtered_sheets.keys())
    sheet_music_files = os.listdir(sheet_music_root)
    annotation_files = os.listdir(annotations_root)

    # Check if there are enough files to split
    if len(sheet_files) < 10:
        raise ValueError("Not enough valid sheet music files after filtering to perform splits.")

    # Labels: count bounding boxes per sheet file
    labels = [len(filtered_sheets[sheet]) for sheet in sheet_files]

    # Step 4: Perform stratified split
    try:
        train_files, test_files = train_test_split(sheet_files, test_size=0.1, random_state=42, stratify=labels)
        train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42, stratify=[labels[sheet_files.index(f)] for f in train_files])
    except ValueError:
        print("Stratified split failed due to low counts. Using random split as fallback.")
        train_files, test_files = train_test_split(sheet_files, test_size=0.1, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42)

    splits = {"train": train_files, "val": val_files, "test": test_files}

    # Step 5: Create output directories
    for split in splits:
        os.makedirs(os.path.join(output_root, split, "sheet_music"), exist_ok=True)
        os.makedirs(os.path.join(output_root, split, "annotations"), exist_ok=True)
        os.makedirs(os.path.join(output_root, split, "bounding_boxes"), exist_ok=True)
        for class_folder in valid_classes.keys():
            os.makedirs(os.path.join(output_root, split, "bounding_boxes", class_folder), exist_ok=True)

    # Step 6: Copy files and track class counts
    class_count = {split: {} for split in splits}
    for split, sheet_list in splits.items():
        for sheet_file in sheet_list:
            # Copy sheet music file
            if sheet_file in sheet_music_files:
                src_sheet = os.path.join(sheet_music_root, sheet_file)
                dst_sheet = os.path.join(output_root, split, "sheet_music", sheet_file)
                shutil.copy(src_sheet, dst_sheet)

                # Copy corresponding annotation file (XML)
                annotation_file = sheet_file.replace('.png', '.xml')
                if annotation_file in annotation_files:
                    src_annotation = os.path.join(annotations_root, annotation_file)
                    dst_annotation = os.path.join(output_root, split, "annotations", annotation_file)
                    shutil.copy(src_annotation, dst_annotation)
                else:
                    print(f"Warning: Annotation file '{annotation_file}' not found in '{annotations_root}'.")

            else:
                print(f"Warning: Sheet music file '{sheet_file}' not found in '{sheet_music_root}'.")

            # Copy bounding box files
            for class_folder, bbox_file in filtered_sheets[sheet_file]:
                src_bbox = os.path.join(bbox_root, class_folder, bbox_file)
                dst_bbox = os.path.join(output_root, split, "bounding_boxes", class_folder, bbox_file)
                if os.path.exists(src_bbox):
                    shutil.copy(src_bbox, dst_bbox)

                # Track class counts
                if class_folder not in class_count[split]:
                    class_count[split][class_folder] = 0
                class_count[split][class_folder] += 1

    # Step 7: Save class splits to JSON
    with open(class_split_json, "w") as json_file:
        json.dump(class_count, json_file, indent=4)

    print("Data split into train, val, and test successfully!")
    print(f"Classes with fewer than {min_cases} cases were ignored.")
    print(f"Class distribution saved to {class_split_json}")


# Example usage
if __name__ == "__main__":
    output_root = "/Users/costanzasiniscalchi/Documents/MS/DLCV/Sheet-Music-Parser/ModelTrainer/datasets/data/data/muscima_split"  # Directory to save the splits
    sheet_music_root = "/Users/costanzasiniscalchi/Documents/MS/DLCV/Sheet-Music-Parser/ModelTrainer/datasets/data/data/muscima_pp_raw/v2.0/data/images"  # Whole sheet music files
    bbox_root = "/Users/costanzasiniscalchi/Documents/MS/DLCV/Sheet-Music-Parser/ModelTrainer/datasets/data/data/muscima_images"  # Bounding box images in class-specific folders
    annotations_root = "/Users/costanzasiniscalchi/Documents/MS/DLCV/Sheet-Music-Parser/ModelTrainer/datasets/data/data/muscima_pp_raw/v2.0/data/annotations"
    split_strat(output_root, sheet_music_root, bbox_root, annotations_root, min_cases=100)

