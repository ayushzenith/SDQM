import argparse
import json
import os
import re
from typing import Any, Callable
import pandas as pd
import ray
import torch
from tqdm import tqdm
from ultralytics import YOLO
from utils.aggregate_custom_split import (
    aggregate_and_split,
    dimo_scene,
    rareplanes_real_scene,
    rareplanes_synthetic_scene,
    wasabi_scene,
)
from utils.create_csv import create_csv
from utils.embedding import Embedder
from utils.evolver import ALL_METRICS, evolve_datasets
from utils.sqdm_tools import SDQM_Outputter

EMBEDDING_MODELS = [
    "ViT-B/32",
    "IDEA-Research/grounding-dino-tiny",
    "facebook/dinov2-small",
]
def to_path(path: str) -> str:
    # remove all characters that are not alphanumeric or a period
    return "".join(c for c in path if c.isalnum() or c == ".")

def process_embeddings(evolved_set, embedding_model, split, text_type=None):
    embedding_dir = os.path.join(evolved_set, "embeddings")
    os.makedirs(embedding_dir, exist_ok=True)
    if not os.path.exists(os.path.join(embedding_dir, to_path(embedding_model))):
        os.makedirs(os.path.join(embedding_dir, to_path(embedding_model)), exist_ok=True)
    if (
        not os.path.exists(os.path.join(embedding_dir, to_path(embedding_model), split))
        or len(os.listdir(os.path.join(embedding_dir, to_path(embedding_model), split))) == 0
    ):
        os.makedirs(
            os.path.join(embedding_dir, to_path(embedding_model), split),
            exist_ok=True,
        )

        check = evolved_set
        if text_type is not None:
            check = text_type
        if "wasabi" in check:
            model_text = "vehicle . car . jeep . truck ."
        elif "dimo" in check:
            model_text = "metal cube . rectangular tray . circular metal piece . cylindrical metal disc . metal rod . metal pipe ."
        elif "rareplanes" in check:
            model_text = "plane . aeroplane . jet . airplane . flight ."
        elif "coco" in check:
            model_text = "person . bicycle . car . motorcycle . airplane . bus . train . truck . boat . traffic light . fire hydrant . stop sign . parking meter . bench . bird . cat . dog . horse . sheep . cow . elephant . bear . zebra . giraffe . backpack . umbrella . handbag . tie . suitcase . frisbee . skis . snowboard . sports ball . kite . baseball bat . baseball glove . skateboard . surfboard . tennis racket . bottle . wine glass . cup . fork . knife . spoon . bowl . banana . apple . sandwich . orange . broccoli . carrot . hot dog . pizza . donut . cake . chair . couch . potted plant . bed . dining table . toilet . tv . laptop . mouse . remote . keyboard . cell phone . microwave . oven . toaster . sink . refrigerator . book . clock . vase . scissors . teddy bear . hair drier . toothbrush . "
        else:
            model_text = ""

        embedder = Embedder(
            p_path=os.path.join(evolved_set, "images", split),
            device_id=0,
            embedding_model=embedding_model,
            save_path=os.path.join(embedding_dir, to_path(embedding_model), split),
            text=model_text,
        )
        embedder.run()


def create_initial_splits(
    num_datasets: int,
    split: dict,
    input_yolo_dir: str,
    output_dir: str,
    scene_function: Callable[[str], Any],
    input_splits: list[str],
) -> None:
    for i in tqdm(range(num_datasets), desc="Dataset Creation"):
        dataset_output_dir = os.path.join(output_dir, "datasets", f"dataset_{i}")
        os.makedirs(dataset_output_dir, exist_ok=True)
        aggregate_and_split(
            scene_function,
            input_yolo_dir,
            dataset_output_dir,
            split,
            input_splits=input_splits,
            symlink=True,
        )


def embed_initial_splits(
    num_datasets: int,
    output_dir: int,
    model_text: str,
) -> None:
    for i in tqdm(range(num_datasets), desc="Embedding"):
        for model in EMBEDDING_MODELS:
            embedding_dir = os.path.join(output_dir, "embeddings", to_path(model), f"dataset_{i}")
            corresponding_dataset_dir = os.path.join(output_dir, "datasets", f"dataset_{i}")
            for split in os.listdir(os.path.join(corresponding_dataset_dir, "images")):
                if len(os.listdir(os.path.join(corresponding_dataset_dir, "images", split))) == 0:
                    continue
                os.makedirs(embedding_dir, exist_ok=True)
                os.makedirs(os.path.join(embedding_dir, split), exist_ok=True)
                embedder = Embedder(
                    p_path=os.path.join(corresponding_dataset_dir, "images", split),
                    device_id=0,
                    embedding_model=model,
                    save_path=os.path.join(embedding_dir, split),
                    text=model_text,
                )
                embedder.run()


def run_evolution_for_train(
    num_datasets: int,
    output_dir: str,
    evolve_output_dir: str,
    synthetic_embedding_dir: str,
    metrics: list[str],
) -> None:
    for model in EMBEDDING_MODELS:
        synthetic_train_file = None
        if synthetic_embedding_dir:
            synthetic_train_dir = os.path.join(synthetic_embedding_dir, to_path(model), "train")
            for file in os.listdir(synthetic_train_dir):
                if file.endswith(".pkl"):
                    synthetic_train_file = os.path.join(synthetic_train_dir, file)
                    break
        for i in tqdm(range(num_datasets), desc="Evolution"):
            for metric in ALL_METRICS:
                if os.path.exists(
                    os.path.join(
                        evolve_output_dir,
                        f"dataset_{i}",
                        to_path(model),
                        metric,
                        "selected_subsets",
                    )
                ):
                    continue
                embedding_dir = os.path.join(output_dir, "embeddings", to_path(model), f"dataset_{i}", "train")
                embedding_file = None
                for file in os.listdir(embedding_dir):
                    if file.endswith(".pkl"):
                        embedding_file = os.path.join(embedding_dir, file)
                        break
                if embedding_file is None:
                    raise FileNotFoundError(f"No .pkl file found in {embedding_dir}")
                evolve_datasets(
                    "overlap",
                    [embedding_file],
                    [synthetic_train_file],
                    100,
                    150,
                    11,
                    os.path.join(evolve_output_dir, f"dataset_{i}", to_path(model), metric),
                    metric,
                    10000,
                    0.1,
                    0.9,
                    None,
                    1000,
                    True,
                    True,
                    True,
                )


def run_evolution_for_val(
    num_datasets: int,
    output_dir: str,
    evolve_output_dir: str,
    synthetic_embedding_dir: str,
    metrics: list[str],
) -> None:
    for model in EMBEDDING_MODELS:
        if synthetic_embedding_dir:
            synthetic_val_dir = os.path.join(synthetic_embedding_dir, to_path(model), "val")
            if not os.path.exists(synthetic_val_dir):
                synthetic_val_dir = os.path.join(synthetic_embedding_dir, to_path(model), "eval")
            for file in os.listdir(synthetic_val_dir):
                if file.endswith(".pkl"):
                    synthetic_val_file = os.path.join(synthetic_val_dir, file)
                    break
        for i in tqdm(range(num_datasets), desc="Validation Evolution"):
            embedding_dir = os.path.join(output_dir, "embeddings", to_path(model), f"dataset_{i}", "val")
            if not os.path.exists(embedding_dir):
                embedding_dir = os.path.join(output_dir, "embeddings", to_path(model), f"dataset_{i}", "eval")
            embedding_file = None
            for file in os.listdir(embedding_dir):
                if file.endswith(".pkl"):
                    embedding_file = os.path.join(embedding_dir, file)
                    break
            embedding_dir2 = os.path.join(output_dir, "embeddings", to_path(model), f"dataset_{i}", "train")
            embedding_file2 = None
            for file in os.listdir(embedding_dir2):
                if file.endswith(".pkl"):
                    embedding_file2 = os.path.join(embedding_dir2, file)
                    break
            if embedding_file is None:
                raise FileNotFoundError(f"No .pkl file found in {embedding_dir}")
            if embedding_file2 is None:
                raise FileNotFoundError(f"No .pkl file found in {embedding_dir}")
            synthetic_val_file = None
            evolve_datasets(
                "no-overlap",
                [synthetic_val_file],
                [embedding_file, embedding_file2],
                100,
                150,
                11,
                os.path.join(evolve_output_dir, f"dataset_{i}", to_path(model), metrics[0]),
                metrics,
                10000,
                0.1,
                0.9,
                None,
                1000,
                True,
                True,
                True,
            )


def train_yolo_models(
    num_datasets: int,
    evolved_dataset_dir: str,
    initial_split_dir: str,
    imgsz: int,
    skip_existing: bool = True,
) -> None:
    for i in tqdm(range(num_datasets), desc="Training"):
        # get metric dirs
        metric_names = [str(metric) for metric in ALL_METRICS]

        # recursively search for metric dirs
        def get_metric_dirs(base_dir, metric_names):
            metric_dirs = []
            for root, dirs, files in os.walk(base_dir):
                for dir_name in dirs:
                    if dir_name in metric_names:
                        metric_dirs.append(os.path.join(root, dir_name))
            return metric_dirs

        metric_dirs = get_metric_dirs(os.path.join(evolved_dataset_dir, f"dataset_{i}"), metric_names)

        for metric_dir in metric_dirs:
            selected_subsets_dir = os.path.join(metric_dir, "selected_subsets")
            if not os.path.exists(selected_subsets_dir):
                print(f"No selected_subsets directory found in {metric_dir}")
                continue
            for _, subset in enumerate(tqdm(os.listdir(selected_subsets_dir))):
                if os.path.exists(os.path.join(metric_dir, "selected_subsets", subset, "eval_metrics.csv")):
                    continue

                test_yaml_file = os.path.join(initial_split_dir, "datasets", f"dataset_{i}", "data.yaml")

                yaml_file = os.path.join(metric_dir, "selected_subsets", subset, "data.yaml")
                # replace "train/images" with "images/train" and "val/images" with "images/val"
                with open(yaml_file, "r") as f:
                    data = f.read()
                data = data.replace("train/images", "images/train")
                data = data.replace("val/images", "images/val")
                with open(yaml_file, "w") as f:
                    f.write(data)
                with open(test_yaml_file, "r") as f:
                    data = f.read()
                data = data.replace("train/images", "images/train")
                data = data.replace("val/images", "images/val")
                with open(test_yaml_file, "w") as f:
                    f.write(data)

                # train model
                os.chdir(os.path.join(metric_dir, "selected_subsets", subset))

                # flip the train and val splits if flipped file not found
                if not os.path.exists("flipped"):
                    # move labels/train to labels/val and labels/val to labels/train
                    for split in ["train", "val"]:
                        if os.path.exists(f"labels/{split}"):
                            os.rename(f"labels/{split}", f"labels/tmp_{split}")
                    for split in ["train", "val"]:
                        if os.path.exists(f"labels/tmp_{split}"):
                            os.rename(f"labels/tmp_{split}", f"labels/{split}")

                    # move images/train to images/val and images/val to images/train
                    for split in ["train", "val"]:
                        if os.path.exists(f"images/{split}"):
                            os.rename(f"images/{split}", f"images/tmp_{split}")
                    for split in ["train", "val"]:
                        if os.path.exists(f"images/tmp_{split}"):
                            os.rename(f"images/tmp_{split}", f"images/{split}")

                    # create flipped file
                    with open("flipped", "w") as f:
                        f.write("flipped")

                # move runs directory to prevent overwriting
                if os.path.exists("runs"):
                    if skip_existing:
                        print(f"Skipping {os.getcwd()}")
                        continue
                    highest_num = -1
                    for dir_name in os.listdir():
                        if dir_name.startswith("runs_bak"):
                            try:
                                num = int(dir_name.replace("runs_bak", ""))
                                if num > highest_num:
                                    highest_num = num
                            except ValueError:
                                continue
                    new_dir_name = f"runs_bak{highest_num + 1}"
                    os.rename("runs", new_dir_name)

                model = YOLO("yolo11n.yaml")
                train_results = model.train(
                    data=yaml_file,
                    epochs=500,
                    imgsz=imgsz,
                    device="0,1",
                )
                # Find all folders named "train{x}" under the current directory runs/detect
                best_model_path = os.path.join("runs", "detect", "train", "weights", "best.pt")

                try:
                    best_model = YOLO(best_model_path)
                except Exception as e:
                    print(f"Error loading model from {os.getcwd()}: {e}")
                    continue
                # evaluate on test set
                eval_metrics = best_model.val(
                    data=yaml_file,
                    imgsz=imgsz,
                    device="0,1",
                )
                test_results = best_model.val(
                    data=test_yaml_file,
                    split="test",
                    imgsz=imgsz,
                    device="0,1",
                )

                # save csvs
                pd.DataFrame([eval_metrics.results_dict]).to_csv(
                    os.path.join(metric_dir, "selected_subsets", subset, "eval_metrics.csv"),
                    index=False,
                )
                pd.DataFrame([test_results.results_dict]).to_csv(
                    os.path.join(metric_dir, "selected_subsets", subset, "test_results.csv"),
                    index=False,
                )

                # save metadata
                metadata = {
                    "eval_metrics": eval_metrics.results_dict,
                    "test_results": test_results.results_dict,
                    "weights": best_model_path,
                    "yaml_file": yaml_file,
                    "test_yaml_file": test_yaml_file,
                    "subset": subset,
                    "metric_dir": metric_dir,
                }
                # Save git commit hash
                git_commit_hash = os.popen(f"git -C {os.path.dirname(__file__)} rev-parse HEAD").read().strip()
                metadata["git_commit_hash"] = git_commit_hash
                with open(
                    os.path.join(metric_dir, "selected_subsets", subset, "v11_metadata.json"),
                    "w",
                ) as f:
                    json.dump(metadata, f, indent=4)


def embed_evolved_splits(
    evolved_dataset_dir: str,
    model_text: str,
) -> None:
    evolved_sets = []
    for dirpath, dirnames, _ in tqdm(os.walk(evolved_dataset_dir), desc="Processing directories"):
        if os.path.basename(dirpath) == "selected_subsets":
            for subdir in dirnames:
                subdir_path = os.path.join(dirpath, subdir)
                evolved_sets.append(subdir_path)

    # Calculate embeddings for each evolved set
    for evolved_set in tqdm(evolved_sets, desc="Embedding evolved sets"):
        for embedding_model in EMBEDDING_MODELS:
            for split in ["train", "val"]:
                try:
                    process_embeddings(evolved_set, embedding_model, split, model_text)
                except Exception as e:
                    print(
                        f"Error processing embeddings for {evolved_set} with model {embedding_model} and split {split}: {e}"
                    )


def run(
    num_datasets: int,
    input_yolo_dir: str,
    synthetic_yolo_dir: str,
    output_dir: str,
    input_splits: list[str],
    new_split_proportions: dict[str, float],
    imgsz: int,
    scene_function: Callable[[str], Any],
    synthetic_scene_function: Callable[[str], Any],
    model_text: str,
) -> None:
    print("Creating initial splits for real datasets...")
    create_initial_splits(
        num_datasets,
        new_split_proportions,
        input_yolo_dir,
        os.path.join(output_dir, "initial_split_real"),
        scene_function,
        input_splits,
    )

    print("Creating initial splits for synthetic datasets...")
    create_initial_splits(
        1,
        new_split_proportions,
        synthetic_yolo_dir,
        os.path.join(output_dir, "initial_split_synthetic"),
        synthetic_scene_function,
        input_splits,
    )

    synthetic_dataset_dir = os.path.join(
        output_dir,
        "initial_split_synthetic",
        "datasets",
        "dataset_0",
    )

    print("Embedding initial splits for real datasets...")
    embed_initial_splits(num_datasets, os.path.join(output_dir, "initial_split_real"), model_text)

    print("Embedding initial splits for synthetic datasets...")
    embed_initial_splits(1, os.path.join(output_dir, "initial_split_synthetic"), model_text)

    print("Running evolution for training datasets...")
    run_evolution_for_train(
        num_datasets,
        os.path.join(output_dir, "train"),
        os.path.join(output_dir, "initial_split_real"),
        synthetic_dataset_dir,
        ALL_METRICS,
    )

    print("Running evolution for validation datasets...")
    run_evolution_for_val(
        num_datasets,
        os.path.join(output_dir, "val"),
        os.path.join(output_dir, "initial_split_real"),
        synthetic_dataset_dir,
        ALL_METRICS,
    )

    print("Embedding evolved splits for training datasets...")
    embed_evolved_splits(
        os.path.join(output_dir, "train"),
        model_text,
    )

    print("Embedding evolved splits for validation datasets...")
    embed_evolved_splits(
        os.path.join(output_dir, "val"),
        model_text,
    )

    print("Training YOLO models for training datasets...")
    train_yolo_models(
        num_datasets,
        os.path.join(output_dir, "train"),
        os.path.join(output_dir, "initial_split_real"),
        imgsz,
    )

    print("Training YOLO models for validation datasets...")
    train_yolo_models(
        num_datasets,
        os.path.join(output_dir, "val"),
        os.path.join(output_dir, "initial_split_real"),
        imgsz,
    )

    print("Creating CSV for training datasets...")
    create_csv(
        os.path.join(output_dir, "train"),
        os.path.join(output_dir, "train_map.csv"),
        "test",
        "mAP50-95",
    )

    print("Creating CSV for training datasets (mAP50)...")
    create_csv(
        os.path.join(output_dir, "train"),
        os.path.join(output_dir, "train_map50.csv"),
        "test",
        "mAP50",
    )

    print("Creating CSV for validation datasets...")
    create_csv(
        os.path.join(output_dir, "val"),
        os.path.join(output_dir, "val_map.csv"),
        "test",
        "mAP50-95",
    )

    print("Creating CSV for validation datasets (mAP50)...")
    create_csv(
        os.path.join(output_dir, "val"),
        os.path.join(output_dir, "val_map50.csv"),
        "test",
        "mAP50",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_datasets", type=int, default=1, help="Number of real dataset splits to create")
    parser.add_argument("--input_yolo_dir", type=str, default="data/yolo", help="Path to real dataset YOLO directory")
    parser.add_argument("--synthetic_yolo_dir", type=str, default="data/yolo", help="Path to synthetic dataset YOLO directory")
    parser.add_argument("--output_dir", type=str, default="data", help="Path to output selected datasets")
    parser.add_argument("--model_text", type=str, default="Vehicle", help="Text to use for grounding dino embedding model")
    parser.add_argument(
        "--scene_function",
        type=str,
        choices=[
            "wasabi_scene",
            "rareplanes_real_scene",
            "rareplanes_synthetic_scene",
            "dimo_scene",
        ],
        help="Scene function to use for splitting",
    )
    parser.add_argument(
        "--synthetic_scene_function",
        type=str,
        choices=[
            "wasabi_scene",
            "rareplanes_real_scene",
            "rareplanes_synthetic_scene",
            "dimo_scene",
        ],
        help="Scene function to use for synthetic dataset splitting",
    )
    parser.add_argument("--input_splits", type=str, nargs="+", default=["train", "val", "test"], help="Splits from input datasets to use")
    parser.add_argument("--train_split", type=float, help="Train split proportion")
    parser.add_argument("--val_split", type=float, help="Validation split proportion")
    parser.add_argument("--test_split", type=float, help="Test split proportion")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for YOLO training")

    args = parser.parse_args()

    split = {
        "train": args.train_split,
        "val": args.val_split,
        "test": args.test_split,
    }

    run(
        args.num_datasets,
        args.input_yolo_dir,
        args.synthetic_yolo_dir,
        args.output_dir,
        args.input_splits,
        split,
        args.imgsz,
        globals()[args.scene_function],
        globals()[args.synthetic_scene_function],
        args.model_text,
    )


if __name__ == "__main__":
    main()
