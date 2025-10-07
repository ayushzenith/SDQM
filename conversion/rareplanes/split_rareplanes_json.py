import os
import json
import logging
from tqdm import tqdm
import argparse


def split_rareplanes_json(
    json_file: str, output_path: str, proportion: float = 0.1
) -> None:
    logging.info(f"Splitting {json_file} into {output_path} with {proportion} proportion")

    # Set up for new json file
    new_data = {"images": [], "annotations": [], "categories": []}

    # Load the json file
    with open(json_file) as f:
        data = json.load(f)

    # Get the number of images to keep
    num_images = int(len(data["images"]) * proportion)
    logging.info(f"Keeping {num_images} images out of {len(data['images'])}")

    # Get all location ids
    location_ids = set()
    for image in data["images"]:
        location_id = image["file_name"].split("_")[0]
        location_ids.add(location_id)

    # Begin selecting one image from each location
    # Until we have the desired number of images
    with tqdm(total=num_images, desc="Selecting images") as pbar:
        while len(new_data["images"]) < num_images:
            for location_id in location_ids:
                for image in data["images"]:
                    if (
                        image["file_name"].startswith(location_id)
                        and image not in new_data["images"]
                    ):
                        if len(new_data["images"]) < num_images:
                            new_data["images"].append(image)
                        pbar.update()  # Update progress bar
                        break

    # Get annotation ids
    annotation_ids = set()
    for image in new_data["images"]:
        for annotation in data["annotations"]:
            if annotation["image_id"] == image["id"]:
                annotation_ids.add(annotation["id"])

    # Select the annotations
    for annotation in data["annotations"]:
        if annotation["id"] in annotation_ids:
            new_data["annotations"].append(annotation)

    # Make annotation ids sequential
    for i, annotation in enumerate(new_data["annotations"]):
        annotation["id"] = i

    # Make image ids sequential
    image_id_mapping = {}
    for i, image in enumerate(new_data["images"]):
        image_id_mapping[image["id"]] = i
        image["id"] = i

    # Update annotation image ids
    for annotation in new_data["annotations"]:
        annotation["image_id"] = image_id_mapping[annotation["image_id"]]

    # Copy categories
    new_data["categories"] = data["categories"]

    # Make sure the output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    # Write the new json file
    with open(output_path, "w") as f:
        json.dump(new_data, f)
    logging.info(f"Saved new json file to {output_path}")


def main() -> None:
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", type=str, help="Path to the json file")
    parser.add_argument("output_path", type=str, help="Path to the output json file")
    parser.add_argument(
        "--proportion",
        "-p",
        type=float,
        default=0.1,
        help="Proportion of images to keep",
    )
    args = parser.parse_args()

    # Run the function
    split_rareplanes_json(args.json_file, args.output_path, args.proportion)


if __name__ == "__main__":
    main()
