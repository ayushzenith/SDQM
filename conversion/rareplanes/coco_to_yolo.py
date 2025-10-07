import os
import shutil
import json
import os
from collections import defaultdict
from pathlib import Path
import argparse
import sys

import numpy as np
import requests
import yaml
from PIL import Image
from tqdm import tqdm


def make_dirs(dir_name="new_dir/"):
    """Creates a directory with subdirectories 'labels' and 'images', removing existing ones."""
    dir_name = Path(dir_name)
    if dir_name.exists():
        shutil.rmtree(dir_name)  # delete dir
    for p in dir_name, dir_name / "labels", dir_name / "images":
        p.mkdir(parents=True, exist_ok=True)  # make dir
    return dir_name


def convert_coco_json(json_files, image_dir, output_dir, create_yaml=True):
    """Converts COCO JSON format to YOLO label format, with options for segments and class mapping."""
    # Source: https://github.com/ultralytics/JSON2YOLO/blob/master/general_json2yolo.py
    save_dir = make_dirs(output_dir)  # output directory

    categories = {}
    # Import json
    for i,json_file in enumerate(json_files):
        if not json_file:
            continue
        json_name = ["train", "test", "eval"]
        json_name = json_name[i]
        fn = Path(save_dir) / "labels" / json_name  # folder name
        fn.mkdir()
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {"%g" % x["id"]: x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        # Fill with all image ids
        for img in images.values():
            imgToAnns[img["id"]] = []
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Get lowest annotation category id
        starts_at_one = False
        if min([ann["category_id"] for ann in data["annotations"]]) == 1:
            starts_at_one = True

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images["%g" % img_id]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            for ann in anns:
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = ann["category_id"] - 1 if starts_at_one else ann["category_id"] # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)

            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (*(bboxes[i]),)  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

        # Get categories
        for category in data["categories"]:
            categories[category["id"] - 1 if starts_at_one else category["id"]] = category["name"]

    finish_coco_to_yolo(output_dir, image_dir, categories, create_yaml)

def finish_coco_to_yolo(yolo_dir, image_dir, categories, create_yaml=True):
    print("Finishing conversion to YOLO format...")
    for label_dir in os.listdir(os.path.join(yolo_dir, "labels")):
        # Create images directory if it doesn't exist
        os.makedirs(os.path.join(yolo_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, "images", label_dir), exist_ok=True)
        with tqdm(total=len(os.listdir(os.path.join(yolo_dir, "labels", label_dir))), desc=f"Copying images {label_dir}") as pbar:
            for label_file in os.listdir(os.path.join(yolo_dir, "labels", label_dir)):
                # Copy each image with the same name as the label file to the corresponding yolo_dir/images directory
                base_name = ".".join(os.path.basename(label_file).split(".")[:-1])
                image_path = None
                for ext in [".jpg", ".png"]:
                    image_path = os.path.join(image_dir, base_name + ext)
                    if os.path.exists(image_path):
                        extension = ext
                        break
                if os.path.exists(image_path):
                    shutil.copy(image_path, os.path.join(yolo_dir, "images", label_dir, base_name + extension))
                else:
                    print(f"Image file {image_path} not found. Skipping.")
                pbar.update()

        # Now make txt files under yolo_dir for each subset with paths to each image file in the respective subset
        with open(os.path.join(yolo_dir, f"{label_dir}.txt"), "w") as f:
            for image_file in os.listdir(os.path.join(yolo_dir, "images", label_dir)):
                f.write(f"{os.path.join('./images', label_dir, image_file)}\n")

    if create_yaml:
        # Finally, make the yaml file
        with open(os.path.join(yolo_dir, "data.yaml"), "w") as f:
            yaml_data = {"path": os.path.abspath(yolo_dir),
                        "train": f"train.txt",
                        "val": f"eval.txt",
                        "test": f"test.txt",
                        "names": categories}
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)


def main():
    print("coco_to_yolo.py 6-28-24v1")
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-directory", type=str, default="yolo", help="Path to the output YOLO directory", required=True)
    parser.add_argument("-i", "--image_dir", type=str, default="images", help="Path to the image directory", required=True)
    parser.add_argument("-j", "--train_json", type=str, help="Path to the JSON file to convert")
    parser.add_argument("-t", "--test_json", type=str, help="Path to the JSON file to convert")
    parser.add_argument("-e", "--eval_json", type=str, help="Path to the JSON file to convert")
    args = parser.parse_args()

    # Ensure at least one of the JSON files is provided
    if not any([args.train_json, args.test_json, args.eval_json]):
        parser.error("At least one JSON file must be provided")
    
    # Ensure python version is 3.7 or higher
    def prompt_user_for_yaml():
        version_info = sys.version_info
        while True:
            user_input = input(f"Python 3.7+ is needed to create a YAML file. Your version: {version_info.major}.{version_info.minor}.{version_info.micro}. Continue without creating YAML? (y/N) ").strip().lower()
            if user_input in ['yes', 'y']:
                return True
            else:
                return False
    create_yaml = True
    if sys.version_info < (3, 7):
        # Prompt user to continue without creating yaml file
        if prompt_user_for_yaml():
            create_yaml = False
        else:
            sys.exit(1)
        

    # Make list of json files
    json_files = [args.train_json, args.test_json, args.eval_json]

    convert_coco_json(json_files, args.image_dir, args.output_directory, create_yaml)


if __name__ == "__main__":
    main()
