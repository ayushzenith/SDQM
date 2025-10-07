import argparse
import mimetypes
import os
import random
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.core as foc
import numpy as np
import pyrender
import trimesh
from dimo_loader import DimoLoader
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm


def load_models(models):
    # Load all models and return a dictionary with the id as key
    return {model["id"]: trimesh.load(model["cad"]) for model in models}


def get_bbox(mesh, obj_pose, K, image_size=(2560, 2048)):
    # Get vertices of the mesh and transform them to camera coordinates
    vertices = np.array(mesh.vertices)
    transformed_vertices = (obj_pose[:3, :3] @ vertices.T).T + obj_pose[:3, 3]

    # Project 3D vertices to 2D image plane
    projected_vertices = K @ transformed_vertices.T
    projected_vertices /= projected_vertices[2, :]  # Divide by z to normalize
    projected_vertices = projected_vertices[:2, :].T  # Keep only x and y

    # Calculate the bounding box
    min_x, min_y = np.min(projected_vertices, axis=0)
    max_x, max_y = np.max(projected_vertices, axis=0)

    # Clamp the bounding box to the image size
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(image_size[0], max_x)
    max_y = min(image_size[1], max_y)

    # Normalize the bounding box
    min_x = min_x / image_size[0]
    min_y = min_y / image_size[1]
    max_x = max_x / image_size[0]
    max_y = max_y / image_size[1]

    # Convert to [x, y, w, h]
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def get_image_info(image_path, image_size=None):
    """
    Get detailed information about an image.

    Parameters:
        image_path (str): The file path to the image.

    Returns:
        dict: A dictionary containing:
            - size_bytes: The size of the image on disk, in bytes.
            - mime_type: The MIME type of the image.
            - width: The width of the image, in pixels.
            - height: The height of the image, in pixels.
            - num_channels: The number of channels in the image.
    """
    try:
        # Get the size of the image on disk in bytes
        size_bytes = os.path.getsize(image_path)

        # Get the MIME type of the image
        mime_type, _ = mimetypes.guess_type(image_path)

        if not image_size:
            # Load the image to get width, height, and number of channels
            with Image.open(image_path) as img:
                width, height = img.size
                num_channels = len(img.getbands())
        else:
            width, height = image_size
            num_channels = 3

        # Return the information as an ImageMetadata object
        return foc.metadata.ImageMetadata(
            size_bytes=size_bytes,
            mime_type=mime_type,
            width=width,
            height=height,
            num_channels=num_channels,
        )

    except Exception as e:
        print(f"Error: {e}")
        return None


def dimo_fiftyone_dataset(dimo_path, dimo_subset, dimo_sub_subset=None):
    dataset = fo.Dataset(
        name=f"dimo_small_{dimo_subset}_{random.randint(0, 1000000)}"
        if "small" in dimo_path
        else f"dimo_{dimo_subset}_{random.randint(0, 1000000)}"
    )

    dimo_loader = DimoLoader()
    dimo_ds = dimo_loader.load(
        Path(dimo_path), cameras=[dimo_subset], sub_subset=dimo_sub_subset
    )

    image_size = (320, 256) if "small" in dimo_path else (2560, 2048)

    dimo_subset_scenes = dimo_ds[dimo_subset]
    models = load_models(dimo_ds["models"])

    for scene in tqdm(dimo_subset_scenes, desc=f"Loading {dimo_subset} dataset"):
        for image in scene["images"]:
            # Create a sample for each image
            image_sample = fo.Sample(
                filepath=image["path"],
                metadata=get_image_info(image["path"], image_size),
            )

            # Load image and add it to the sample
            detections = []
            for obj in image["objects"]:
                obj_id = obj["id"]  # int, type of object
                # Load mesh and get bounding box
                mesh = models[obj["id"]].copy()
                bbox = get_bbox(
                    mesh, obj["model_2cam"], image["camera"]["K"], image_size
                )

                # Add sample to dataset
                detections.append(
                    fo.Detection(
                        label=str(obj_id),
                        bounding_box=bbox,
                    )
                )

            # Add ground truth to sample
            image_sample["ground_truth"] = fo.Detections(detections=detections)
            # Add sample to dataset
            dataset.add_sample(image_sample)

    return dataset


def train_test_eval_split(dataset, train_size=0.90, val_size=0.05, custom_split=None):
    scene_ids = [Path(sample.filepath).parts[-3].split("_")[0] for sample in dataset]
    scene_ids = np.unique(scene_ids)
    if not custom_split:
        np.random.shuffle(scene_ids)
        train_scene_ids = scene_ids[: int(train_size * len(scene_ids))]
        val_scene_ids = scene_ids[
            int(train_size * len(scene_ids)) : int((train_size + val_size) * len(scene_ids))
        ]
        test_scene_ids = scene_ids[int((train_size + val_size) * len(scene_ids)) :]
    else:
        train_scene_ids = []
        val_scene_ids = []
        test_scene_ids = []
        for split in ["train", "eval", "test"]:
            with open(os.path.join(custom_split, f"{split}.txt"), "r") as f:
                scene_ids = f.read().splitlines()
                if split == "train":
                    train_scene_ids = scene_ids
                elif split == "eval":
                    val_scene_ids = scene_ids
                elif split == "test":
                    test_scene_ids = scene_ids

    # Create new empty datasets
    train_dataset = fo.Dataset()
    val_dataset = fo.Dataset()
    test_dataset = fo.Dataset()

    # Add samples to the respective datasets
    for sample in tqdm(dataset, desc="Splitting dataset"):
        scene_id = Path(sample.filepath).parts[-3].split("_")[0]
        if scene_id in train_scene_ids:
            train_dataset.add_sample(sample)
        elif scene_id in val_scene_ids:
            val_dataset.add_sample(sample)
        elif scene_id in test_scene_ids:
            test_dataset.add_sample(sample)

    return train_dataset, val_dataset, test_dataset


def export_to_yolo(train_dataset, val_dataset, test_dataset, output_dir):
    def generate_filename(sample):
        scene = sample.filepath.split("/")[-3]  # Extract scene from filepath
        if "_00" in scene:
            scene = scene.split("_")[0]  # Remove sub-subset from scene only if it is 00
        number = sample.filepath.split("/")[-1].split(".")[
            0
        ]  # Extract number from filepath
        extension = sample.filepath.split(".")[-1]  # Extract the file extension
        return f"{scene}_{number}", extension  # Combine scene, number, and extension

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
        # Export each dataset to YOLOv5 format
        for dataset, name in zip(
            [train_dataset, val_dataset, test_dataset], ["train", "val", "test"]
        ):
            # Create the output directory
            os.makedirs(os.path.join(output_dir, "images", name), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "labels", name), exist_ok=True)

            classes = set()
            # Export each sample to a YOLOv5 format
            for sample in tqdm(dataset, desc=f"Exporting {name} dataset"):
                # Get the image path
                image_path = sample.filepath
                # Get the detections
                detections = sample.ground_truth.detections

                # Create the output file
                filename, extension = generate_filename(sample)
                output_file = os.path.join(
                    output_dir, "labels", name, filename + ".txt"
                )

                # Write the YOLOv5 format to the output file
                with open(output_file, "w") as f:
                    for detection in detections:
                        label = int(detection.label) - 1
                        classes.add(label)
                        bbox = detection.bounding_box
                        x, y, w, h = bbox
                        f.write(f"{label} {x + w / 2} {y + h / 2} {w} {h}\n")

                # Copy the image to the output directory
                image_output_file = os.path.join(
                    output_dir, "images", name, f"{filename}.{extension}"
                )
                shutil.copyfile(image_path, image_output_file)

            # Convert classes to sorted list of ints
            classes = sorted([int(c) for c in classes])
            # Create the data.yaml file
            with open(os.path.join(output_dir, "data.yaml"), "w") as f:
                f.write(f"path: {os.path.abspath(output_dir)}\n")
                f.write(f"train: {os.path.join('images', 'train')}\n")
                f.write(f"val: {os.path.join('images', 'val')}\n")
                f.write(f"test: {os.path.join('images', 'test')}\n")
                f.write("")
                f.write("names:\n")
                for c in classes:
                    f.write(f"  {c}: {c}\n")
    else:
        print("Output directory already exists. Skipping export.")


def main():
    parser = argparse.ArgumentParser(description="Convert DIMO dataset to YOLO format")
    parser.add_argument(
        "--dimo_path", type=str, required=True, help="Path to the DIMO dataset"
    )
    parser.add_argument(
        "--dimo_subset",
        type=str,
        default="train",
        help="Subset of the DIMO dataset to use",
    )
    parser.add_argument(
        "--dimo_sub_subset",
        type=str,
        nargs="+",
        default=["00"],
        help="Sub-subset(s) of the DIMO dataset to use",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to the output directory"
    )
    parser.add_argument(
        "--custom_split", type=str, help="Path to a directory containing train, eval, and test files defining custom splits"
    )

    args = parser.parse_args()

    # Load the dataset
    dataset = dimo_fiftyone_dataset(
        args.dimo_path, args.dimo_subset, args.dimo_sub_subset
    )
    train_view, eval_view, test_view = train_test_eval_split(dataset)
    export_to_yolo(train_view, eval_view, test_view, args.output_dir)


if __name__ == "__main__":
    main()
