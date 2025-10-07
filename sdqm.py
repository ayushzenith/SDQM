import argparse
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from dataset_interpretability.run import get_v_info
from dataset_selection.utils.metrics import compute_alpha_precision_beta_recall_authenticity
from dataset_similarity.image_mauve import get_dataset_similarity
from labels_and_characteristics.label_overlap import calculate_yolo_label_overlap
from pixel_intensity.bbox_pixel_analysis import calculate_bounding_box_distribution, calculate_pixel_intensity
from separability.log_cluster_metric import compute_log_cluster_metric
from separability.separate_model import separate as calculate_separability
from spatial_distribution.run import calculate_spatial_distribution


@dataclass
class Metric:
    metric_function: Callable
    name: str
    range: List[float] = field(default_factory=list)
    type: str = "embedding"  # Default type is "embedding"

    def __post_init__(self):
        assert len(self.range) == 2, "Range list should contain two floats"
        valid_types = [
            "embedding",
            "annotation",
            "distribution",
            "spatial",
            "separability",
            "similarity",
            "fdg",
            "lcm",
            "v_info",
            "bounding_box",
            "label_overlap",
            "proportion",
            "size",
        ]
        assert self.type in valid_types, f"Type must be one of {', '.join(valid_types)} but is {self.type}"

    def __call__(self, embeddings1, embeddings2, *args, **kwargs):
        return self.metric_function(embeddings1, embeddings2, *args, **kwargs)

    def __str__(self):
        return self.name


def get_proportion(
    file_list: List[str],
    all_real_files: List[str] | Callable[[str], bool],
) -> float:
    if not isinstance(all_real_files, Callable) and not isinstance(all_real_files, List):
        raise ValueError("all_real_files must be a list or a callable function")
    if isinstance(all_real_files, List):
        return len([file for file in file_list if file not in all_real_files]) / len(file_list)
    elif isinstance(all_real_files, Callable):
        return len([file for file in file_list if not all_real_files(file)]) / len(file_list)


def get_sizes(real_files: List[str], synthetic_files) -> Tuple[int, int]:
    return len(real_files), len(synthetic_files)


METRICS = [
    Metric(calculate_bounding_box_distribution, "Bounding Box Distribution", [0, 1], "bounding_box"),
    Metric(calculate_separability, "Separability", [0, 1], "separability"),
    Metric(calculate_pixel_intensity, "Pixel Intensity", [0, 1], "distribution"),
    Metric(calculate_yolo_label_overlap, "Label Overlap", [0, 1], "label_overlap"),
    Metric(get_v_info, "V-Info", [0, 1], "v_info"),
    Metric(calculate_spatial_distribution, "Spatial Distribution", [0, 1], "spatial"),
    Metric(compute_alpha_precision_beta_recall_authenticity, "FDG", [0, 1], "fdg"),  # Fidelity, Diversity, Generality
    Metric(get_dataset_similarity, "Dataset Similarity", [0, 1], "similarity"),
    Metric(compute_log_cluster_metric, "Cluster Metric", [0, 1], "lcm"),
    Metric(get_proportion, "Synthetic Proportion", [0, 1], "proportion"),
    Metric(get_sizes, "Size", [0, 1], "size"),
]


def load_embedding_file(file_path):
    csv_path = file_path.replace(".pkl", ".csv").replace(".npy", ".csv")
    if file_path.endswith(".pkl"):
        with open(file_path, "rb") as f:
            embeddings = pickle.load(f)
    elif file_path.endswith(".npy"):
        embeddings = np.load(file_path)
    else:
        raise ValueError("File must be a .pkl or .npy file.")

    df = pd.read_csv(csv_path)
    file_names = df["file_path"].tolist()

    return embeddings, file_names


def create_temp_yolo_dir(train_file_names, val_file_names, temp_dir=None):
    if temp_dir is None:
        temp_dir_path = tempfile.mkdtemp()
    else:
        temp_dir_path = os.path.join(temp_dir, "temp_yolo_dir")
        os.makedirs(temp_dir_path, exist_ok=True)

    train_images_dir = os.path.join(temp_dir_path, "images", "train")
    train_labels_dir = os.path.join(temp_dir_path, "labels", "train")
    val_images_dir = os.path.join(temp_dir_path, "images", "val")
    val_labels_dir = os.path.join(temp_dir_path, "labels", "val")

    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    def create_symlinks(file_names, images_dir, labels_dir):
        for file_name in file_names:
            image_file_path_jpg = file_name.replace("/labels/", "/images/").replace(".txt", ".jpg")
            image_file_path_png = file_name.replace("/labels/", "/images/").replace(".txt", ".png")
            label_file_path = file_name
            if os.path.exists(image_file_path_jpg):
                os.symlink(image_file_path_jpg, os.path.join(images_dir, os.path.basename(image_file_path_jpg)))
            elif os.path.exists(image_file_path_png):
                os.symlink(image_file_path_png, os.path.join(images_dir, os.path.basename(image_file_path_png)))
            os.symlink(label_file_path, os.path.join(labels_dir, os.path.basename(label_file_path)))

    create_symlinks(train_file_names, train_images_dir, train_labels_dir)
    create_symlinks(val_file_names, val_images_dir, val_labels_dir)

    # Make the yaml file from the original
    # get original yaml file
    original_yaml_path = None
    current_dir = os.path.dirname(train_file_names[0])
    while current_dir != "/":
        for file in os.listdir(current_dir):
            if file.endswith(".yaml"):
                original_yaml_path = os.path.join(current_dir, file)
                break
        if original_yaml_path:
            break
        current_dir = os.path.dirname(current_dir)

    if not original_yaml_path:
        raise FileNotFoundError("No .yaml file found in any parent directory")

    # Copy the yaml file to the temp dir
    temp_yaml_path = os.path.join(temp_dir_path, "data.yaml")
    with open(original_yaml_path, "r") as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    # change the paths in the yaml file
    yaml_data["path"] = temp_dir_path
    yaml_data["train"] = "images/train"
    yaml_data["val"] = "images/val"

    # save the yaml file
    with open(temp_yaml_path, "w") as f:
        yaml.dump(yaml_data, f)

    return temp_dir_path


def calculate_sdqm(
    real_files,
    synthetic_files,
    image_size=None,
    df_files=None,
    output=None,
    metric_type=["all"],
    dataset="auto",
    temp_dir=None,
):
    all_metric_values = []

    for i, (real_file, synthetic_file) in enumerate(tqdm(zip(real_files, synthetic_files), desc="Calculating DSQM")):
        try:
            real_embeddings, real_file_names = load_embedding_file(real_file)
            synthetic_embeddings, synthetic_file_names = load_embedding_file(synthetic_file)

            if image_size is None:
                # load a real image to get the shape
                image_size = Image.open(real_file_names[0]).size  # (width, height)

            if dataset == "auto":
                if "/rareplanes/" in real_file.lower() or "/new/" in real_file.lower():
                    detected_dataset = "rareplanes"
                elif "/dimo/" in real_file.lower():
                    detected_dataset = "dimo"
                elif "/wasabi/" in real_file.lower():
                    detected_dataset = "wasabi"
                else:
                    detected_dataset = "N/A"
            print(f"Dataset type: {detected_dataset}")

            # Replace /images/ with /labels/ and .jpg and .png with .txt
            real_file_names = [
                f.replace("/images/", "/labels/").replace(".jpg", ".txt").replace(".png", ".txt")
                for f in real_file_names
            ]
            synthetic_file_names = [
                f.replace("/images/", "/labels/").replace(".jpg", ".txt").replace(".png", ".txt")
                for f in synthetic_file_names
            ]

            metric_values = {}
            for metric in METRICS:
                # Check if metric already exists in the CSV and skip if present
                if metric_type != ["all"] and metric.type not in metric_type:
                    print(f"Skipping {metric} as its type {metric.type} is not in the list of metric types to compute.")
                    continue

                try:
                    print(f"Calculating {metric}")
                    if metric.type == "embedding":
                        value = metric(real_embeddings, synthetic_embeddings)
                        metric_values[str(metric)] = value
                    elif metric.type == "annotation":
                        value = metric(real_file_names, synthetic_file_names)
                        metric_values[str(metric)] = value
                    elif metric.type == "distribution" or metric.type == "bounding_box":
                        values_dict = metric(real_file_names, synthetic_file_names)
                        for key, value in values_dict.items():
                            metric_values[f"{str(metric)}_{key}"] = value
                    elif metric.type == "label_overlap":
                        values_dict = metric(real_file_names, synthetic_file_names)[0]
                        for key, value in values_dict.items():
                            metric_values[f"{str(metric)}_{key}"] = value
                    elif metric.type == "spatial":
                        value = metric(real_file_names, synthetic_file_names, image_size)
                        metric_values[str(metric)] = value
                    elif metric.type == "separability":
                        num_params, best_accuracy = metric(real_embeddings, synthetic_embeddings)
                        metric_values[f"accuracy_{str(metric)}"] = best_accuracy
                        metric_values[f"params_{str(metric)}"] = num_params
                    elif metric.type == "similarity":
                        reg, smooth = metric(real_embeddings, synthetic_embeddings)
                        metric_values[f"{str(metric)}_mauve"] = reg.mauve
                        metric_values[f"{str(metric)}_smooth_mauve"] = smooth.mauve
                        metric_values[f"{str(metric)}_frontier_integral"] = reg.fronter_integral
                        metric_values[f"{str(metric)}_smooth_frontier_integral"] = smooth.fronter_integral
                    elif metric.type == "fdg":
                        Delta_alpha_precision, Delta_beta_recall, authenticity = metric(
                            real_embeddings, synthetic_embeddings
                        )
                        metric_values[f"{str(metric)}_fidelity_alpha_precision"] = Delta_alpha_precision
                        metric_values[f"{str(metric)}_diversity_beta_recall"] = Delta_beta_recall
                        metric_values[f"{str(metric)}_generality_authenticity"] = authenticity
                    elif metric.type == "lcm":
                        cm, lcm = metric(real_embeddings, synthetic_embeddings)
                        metric_values[str(metric)] = cm
                        metric_values[f"log_{str(metric)}"] = lcm
                    elif metric.type == "v_info":
                        # Create a temporary directory with symlinks to the images and labels
                        temp_dir = create_temp_yolo_dir(real_file_names, synthetic_file_names, temp_dir)
                        try:
                            yaml_path = os.path.join(temp_dir, "data.yaml")
                            (
                                conditional_iou,
                                predictive_iou,
                                v_info_iou,
                                conditional_conf,
                                predictive_conf,
                                v_info_conf,
                                conditional_fusion,
                                predictive_fusion,
                                v_info_fusion,
                            ) = metric(yaml_path, yaml_path, detected_dataset, image_size[0])
                            metric_values["conditional_iou"] = conditional_iou
                            metric_values["predictive_iou"] = predictive_iou
                            metric_values["v_info_iou"] = v_info_iou
                            metric_values["conditional_conf"] = conditional_conf
                            metric_values["predictive_conf"] = predictive_conf
                            metric_values["v_info_conf"] = v_info_conf
                            metric_values["conditional_fusion"] = conditional_fusion
                            metric_values["predictive_fusion"] = predictive_fusion
                            metric_values["v_info_fusion"] = v_info_fusion
                        finally:
                            # Clean up the temporary directory
                            shutil.rmtree(temp_dir)
                    elif metric.type == "proportion":

                        def is_real_factory(dataset_type):
                            if dataset_type == "wasabi":
                                return lambda file: ".mp4.frame-" in file
                            elif dataset_type == "rareplanes":
                                return lambda file: "_tile_" in file
                            elif dataset_type == "dimo":
                                return lambda file: "synthetic" not in file
                            else:
                                print("Dataset not recognized. Assuming all files are real.")
                                return lambda file: True

                        is_real = is_real_factory(detected_dataset)
                        metric_values[f"{str(metric)}_Validation"] = metric(real_file_names, is_real)
                        metric_values[f"{str(metric)}_Train"] = metric(synthetic_file_names, is_real)
                    elif metric.type == "size":
                        metric_values[f"{str(metric)}_Validation"], metric_values[f"{str(metric)}_Train"] = metric(
                            real_file_names, synthetic_file_names
                        )
                    else:
                        print(f"Warning: Metric type {metric.type} not recognized.")
                except Exception as e:
                    print(f"Error calculating metric {metric}: {e}")

            all_metric_values.append(metric_values)

            # Update CSV if files_csv was provided
            if df_files is not None:
                for metric, value in metric_values.items():
                    df_files.at[i, metric] = value
                df_files.to_csv(output, index=False)
        except Exception as e:
            print(f"Error processing files {real_file} and {synthetic_file}: {e}")

    return all_metric_values


def main():
    parser = argparse.ArgumentParser(description="Calculate DSQM metrics.")
    parser.add_argument(
        "--files_csv", type=str, help="Path to the CSV file containing paths to the real and synthetic embedding files."
    )
    parser.add_argument("--real_files", type=str, nargs="+", help="Paths to the real embedding files (pkl or numpy).")
    parser.add_argument(
        "--synthetic_files", type=str, nargs="+", help="Paths to the synthetic embedding files (pkl or numpy)."
    )
    parser.add_argument("--output", type=str, default="dsqm_values.csv", help="Path to the output CSV or JSON file.")
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Size of the images in the datasets. \
             If not provided, the size will be determined \
             by loading the first image.",
    )
    parser.add_argument(
        "--metric_type",
        type=str,
        nargs="*",
        default=["all"],
        help="Type of metrics to compute. Options are:\n"
        "'all',\n"
        "'embedding',\n"
        "'annotation',\n"
        "'distribution',\n"
        "'spatial',\n"
        "'separability',\n"
        "'similarity',\n"
        "'fdg',\n"
        "'lcm',\n"
        "'v_info',\n"
        "'bounding_box',\n"
        "'label_overlap',\n"
        "'proportion',\n"
        "'size'.\n"
        "Default is 'all'.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="auto",
        help="Type of dataset. Options are 'rareplanes', 'dimo', 'wasabi', or 'auto'. Default is 'auto'.",
    )
    parser.add_argument(
        "--temp_dir",
        type=str,
        default=None,
        help="Path to the temporary directory to store YOLOv5 files for calculating V-Info metric. "
        "If not provided, a temporary directory will be created in /tmp.",
    )

    args = parser.parse_args()

    # Ensure output is writeable
    output_dir = os.path.dirname(args.output)
    if not os.access(output_dir, os.W_OK):
        raise ValueError(f"Output directory {output_dir} is not writeable.")

    # Load files from CSV if provided, otherwise use real_files and synthetic_files arguments
    df_files = None
    if args.files_csv:
        df_files = pd.read_csv(args.files_csv)
        args.real_files = df_files.iloc[:, 1].tolist()
        args.synthetic_files = df_files.iloc[:, 0].tolist()
    elif not (args.real_files and args.synthetic_files):
        raise ValueError("Please provide either --files_csv or both --real_files and --synthetic_files arguments.")

    assert len(args.real_files) == len(args.synthetic_files), "Number of real and synthetic files must be the same."

    dsqm_values = calculate_sdqm(
        args.real_files,
        args.synthetic_files,
        args.image_size,
        df_files,
        args.output,
        args.metric_type,
        args.dataset,
        args.temp_dir,
    )

    # Prepare results
    results = []
    for real_file, synthetic_file, metric_values in zip(args.real_files, args.synthetic_files, dsqm_values):
        result = {"real_file": real_file, "synthetic_file": synthetic_file, **metric_values}
        results.append(result)

    # Update CSV if files_csv was provided
    # Save results to new CSV or JSON if files_csv was not provided
    if df_files is None:
        df = pd.DataFrame(results)
        if args.output.endswith(".json"):
            df.to_json(args.output, orient="records", lines=True)
        else:
            df.to_csv(args.output, index=False)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
