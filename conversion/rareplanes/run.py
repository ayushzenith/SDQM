# Convert RarePlanes Directory
import argparse
import logging
import os
import shutil

from tqdm import tqdm

from coco_to_yolo import convert_coco_json
from random_crop import random_crop_dataset
from split_rareplanes_json import split_rareplanes_json


def copy_files_to_directory_with_progress(source_dirs, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for source_dir in source_dirs:
        files = os.listdir(source_dir)
        for filename in tqdm(files, desc=f"Copying from {source_dir}"):
            source_file = os.path.join(source_dir, filename)
            if os.path.isfile(source_file):
                # Generate a unique file name to avoid overwriting
                dest_file = os.path.join(dest_dir, filename)
                if os.path.exists(dest_file):
                    raise FileExistsError(f"File {dest_file} already exists. Aborting.")
                shutil.copy(source_file, dest_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rareplanes_dir", type=str, help="Path to the RarePlanes dataset directory"
    )
    args = parser.parse_args()

    logging.log("Cropping images and updating annotations...")
    random_crop_dataset(
        input_dir=os.path.join(args.rareplanes_dir, "synthetic", "train", "images"),
        input_annotations_path=os.path.join(
            args.rareplanes_dir,
            "synthetic",
            "metadata_annotations",
            "instances_train_aircraft.json",
        ),
        output_dir=os.path.join(
            args.rareplanes_dir, "synthetic", "train", "cropped_images"
        ),
        output_annotations_path=os.path.join(
            args.rareplanes_dir,
            "synthetic",
            "metadata_annotations",
            "cropped_instances_train_aircraft.json",
        ),
        width=512,
        height=512,
        min_visibility=0.5,
    )
    random_crop_dataset(
        input_dir=os.path.join(args.rareplanes_dir, "synthetic", "test", "images"),
        input_annotations_path=os.path.join(
            args.rareplanes_dir,
            "synthetic",
            "metadata_annotations",
            "instances_test_aircraft.json",
        ),
        output_dir=os.path.join(
            args.rareplanes_dir, "synthetic", "test", "cropped_images"
        ),
        output_annotations_path=os.path.join(
            args.rareplanes_dir,
            "synthetic",
            "metadata_annotations",
            "cropped_instances_test_aircraft.json",
        ),
        width=512,
        height=512,
        min_visibility=0.5,
    )

    logging.log("Creating 10 percent subset of real data...")
    split_rareplanes_json(
        json_file=os.path.join(
            args.rareplanes_dir, "real", "metadata_annotations", "instances_train_aircraft.json"
        ),
        output_path=os.path.join(
            args.rareplanes_dir, "real", "metadata_annotations", "instances_train_aircraft_10.json"
        ),
        proportion=0.1,
    )

    logging.log("Copying files to real_synthetic_train_test directory...")
    copy_files_to_directory_with_progress(
        source_dirs=[
            os.path.join(args.rareplanes_dir, "synthetic", "train", "cropped_images"),
            os.path.join(args.rareplanes_dir, "synthetic", "test", "cropped_images"),
            os.path.join(args.rareplanes_dir, "real", "train", "PS-RGB_tiled"),
            os.path.join(args.rareplanes_dir, "real", "test", "PS-RGB_tiled"),
        ],
        dest_dir=os.path.join(args.rareplanes_dir, "real_synthetic_train_test"),
    )

    logging.log(
        "Converting COCO JSON to YOLO format for synthetic (eval on real) data..."
    )
    json_files = [
        os.path.join(
            args.rareplanes_dir,
            "synthetic",
            "metadata_annotations",
            "cropped_instances_train_aircraft.json",
        ),
        None,
        os.path.join(
            args.rareplanes_dir,
            "real",
            "metadata_annotations",
            "instances_test_aircraft.json",
        ),
    ]
    convert_coco_json(
        json_files=json_files,
        image_dir=os.path.join(args.rareplanes_dir, "real_synthetic_train_test"),
        output_dir=os.path.join(args.rareplanes_dir, "synthetic_yolo"),
    )
    logging.log(
        f"Output directory: {os.path.join(args.rareplanes_dir, 'synthetic_yolo')}"
    )

    logging.log("Converting COCO JSON to YOLO format for real data...")
    json_files = [
        os.path.join(
            args.rareplanes_dir,
            "real",
            "metadata_annotations",
            "instances_train_aircraft.json",
        ),
        None,
        os.path.join(
            args.rareplanes_dir,
            "real",
            "metadata_annotations",
            "instances_test_aircraft.json",
        ),
    ]
    convert_coco_json(
        json_files=json_files,
        image_dir=os.path.join(args.rareplanes_dir, "real_synthetic_train_test"),
        output_dir=os.path.join(args.rareplanes_dir, "real_yolo"),
    )
    logging.log(f"Output directory: {os.path.join(args.rareplanes_dir, 'real_yolo')}")

    logging.log("Converting COCO JSON to YOLO format for 10 percent real data...")
    json_files = [
        os.path.join(
            args.rareplanes_dir,
            "real",
            "metadata_annotations",
            "instances_train_10.json",
        ),
        None,
        os.path.join(
            args.rareplanes_dir,
            "real",
            "metadata_annotations",
            "instances_test_aircraft.json",
        ),
    ]
    convert_coco_json(
        json_files=json_files,
        image_dir=os.path.join(args.rareplanes_dir, "real_synthetic_train_test"),
        output_dir=os.path.join(args.rareplanes_dir, "real_10_percent_yolo"),
    )

    logging.log("Processing completed!")


if __name__ == "__main__":
    main()
