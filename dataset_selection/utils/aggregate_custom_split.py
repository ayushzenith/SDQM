"""
take all images in a yolo dataset
split by scene into train, val, test
move to new yolo directory
"""

import argparse
import os
import random
import shutil
from typing import Any, Callable

import yaml
from tqdm import tqdm


def aggregate_and_split(
    scene_function: Callable[[str], Any],
    yolo_dir: str,
    output_dir: str,
    split: dict,
    input_splits: list[str] = ["train", "val", "test"],
    symlink: bool = False,
) -> None:
    # check if exists
    os.makedirs(output_dir, exist_ok=True)
    assert (
        len(os.listdir(output_dir)) == 0
    ), "output_dir already exists, refusing to overwrite"

    # create output directory structure
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    label_dir = os.path.join(output_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        os.makedirs(os.path.join(image_dir, split_name), exist_ok=True)
        os.makedirs(os.path.join(label_dir, split_name), exist_ok=True)

    # get all images
    all_images = []
    dirs_to_check = [
        os.path.join(yolo_dir, "images", split_name) for split_name in input_splits
    ]
    for dir_to_check in dirs_to_check:
        for root, dirs, files in os.walk(dir_to_check):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    all_images.append(os.path.join(root, file))

    # split by scene
    scene_to_images = {}
    for image in all_images:
        scene = scene_function(image)
        if scene not in scene_to_images:
            scene_to_images[scene] = []
        scene_to_images[scene].append(image)

    scenes = list(scene_to_images.keys())
    random.shuffle(scenes)

    # get splits
    train_split = split["train"]
    val_split = split["val"]
    test_split = split["test"]

    total_images = len(all_images)
    train_target = total_images * train_split
    val_target = total_images * val_split
    test_target = total_images * test_split

    train_scenes = []
    val_scenes = []
    test_scenes = []

    train_count = 0
    val_count = 0
    test_count = 0

    for scene in scenes:
        if test_count + len(scene_to_images[scene]) <= test_target:
            test_scenes.append(scene)
            test_count += len(scene_to_images[scene])
        elif val_count + len(scene_to_images[scene]) <= val_target:
            val_scenes.append(scene)
            val_count += len(scene_to_images[scene])
        else:
            train_scenes.append(scene)
            train_count += len(scene_to_images[scene])

    # Ensure all images are accounted for
    assert train_count + val_count + test_count == total_images, "Image count mismatch"

    # move or symlink images and labels
    for image in tqdm(all_images, desc="Copying or Symlinking files"):
        scene = scene_function(image)
        if scene in train_scenes:
            target_image_dir = os.path.join(image_dir, "train")
            target_label_dir = os.path.join(label_dir, "train")
        elif scene in val_scenes:
            target_image_dir = os.path.join(image_dir, "val")
            target_label_dir = os.path.join(label_dir, "val")
        elif scene in test_scenes:
            target_image_dir = os.path.join(image_dir, "test")
            target_label_dir = os.path.join(label_dir, "test")
        else:
            raise ValueError(f"Scene {scene} not found in splits")

        target_image_path = os.path.join(target_image_dir, os.path.basename(image))
        target_label_path = os.path.join(
            target_label_dir,
            os.path.basename(image).replace(".jpg", ".txt").replace(".png", ".txt"),
        )

        if symlink:
            os.symlink(image, target_image_path)
            os.symlink(
                image.replace(".jpg", ".txt")
                .replace(".png", ".txt")
                .replace("/images/", "/labels/"),
                target_label_path,
            )
        else:
            shutil.copy(image, target_image_path)
            shutil.copy(
                image.replace(".jpg", ".txt")
                .replace(".png", ".txt")
                .replace("/images/", "/labels/"),
                target_label_path,
            )

    # load the original yaml file to get the classes
    yaml_files = [
        os.path.join(yolo_dir, file)
        for file in os.listdir(yolo_dir)
        if file.endswith(".yaml")
    ]
    if not yaml_files:
        raise FileNotFoundError("No .yaml file found in yolo_dir")
    with open(yaml_files[0], "r") as f:
        original_data = yaml.load(f, Loader=yaml.FullLoader)
        original_classes = original_data["names"]
    # save new yaml file
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        yaml.dump(
            {
                "path": output_dir,
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": original_classes,
            },
            f,
        )


def wasabi_scene(file_path: str) -> int:
    return int(os.path.basename(file_path).split(".")[-4])


def rareplanes_real_scene(file_path: str) -> int:
    return int(os.path.basename(file_path).split("_")[0])


def rareplanes_synthetic_scene(file_path: str) -> str:
    return "_".join(os.path.basename(file_path).split("_")[0:2])


def dimo_scene(file_path: str) -> str:
    return os.path.basename(file_path).split("_")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate and split a yolo dataset")
    parser.add_argument(
        "--scene_function",
        type=str,
        choices=["wasabi_scene", "rareplanes_real_scene", "rareplanes_synthetic_scene", "dimo_scene"],
        help="Scene function to use",
    )
    parser.add_argument("--yolo_dir", type=str, help="Path to the yolo dataset")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--train_split", type=float, help="Train split proportion")
    parser.add_argument("--val_split", type=float, help="Validation split proportion")
    parser.add_argument("--test_split", type=float, help="Test split proportion")
    parser.add_argument("--symlink", action="store_true", help="Symlink instead of copy")
    args = parser.parse_args()

    scene_function = globals()[args.scene_function]
    split = {
        "train": args.train_split,
        "val": args.val_split,
        "test": args.test_split,
    }

    aggregate_and_split(scene_function, args.yolo_dir, args.output_dir, split, symlink=args.symlink)


if __name__ == "__main__":
    main()
