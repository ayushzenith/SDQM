import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle


def create_csv(root_path, output_path, split="test", metrics=["mAP50-95"], custom=True):
    """Loop through the root path and merge all csv files named eval_metrics"""

    all_data = {metric: [] for metric in metrics}
    all_paths = []
    train_paths = []
    val_paths = []
    # Use tqdm to create an overall progress bar
    with tqdm(total=1, desc="Walking through directories") as pbar:
        for root, dirs, files in os.walk(root_path):
            if (custom and os.path.basename(root) == "selected_subsets") or (not custom and root == root_path):
                for sub_dir in dirs:
                    sub_dir_path = os.path.join(root, sub_dir)
                    # ensure embeddings dir is present and not empty
                    embeddings_dir = os.path.join(sub_dir_path, "embeddings")
                    if not os.path.exists(embeddings_dir):
                        continue
                    if not os.listdir(embeddings_dir):
                        continue
                    for file in tqdm(
                        os.listdir(sub_dir_path), desc="Processing files", leave=False
                    ):
                        if file == (
                            "test_results.csv"
                            if split == "test"
                            else "eval_metrics.csv"
                        ):
                            file_path = os.path.join(sub_dir_path, file)
                            print("Found file: ", file_path)

                            # Add paths to train.pkl and val.pkl
                            for sub_sub_dir in os.listdir(embeddings_dir):
                                sub_sub_dir_path = os.path.join(
                                    embeddings_dir, sub_sub_dir
                                )
                                if os.path.isdir(sub_sub_dir_path):
                                    train_path = os.path.join(sub_sub_dir_path, "train")
                                    val_path = os.path.join(sub_sub_dir_path, "val")
                                    if os.path.exists(train_path) and os.path.exists(
                                        val_path
                                    ):
                                        train_pkl = next(
                                            (
                                                f
                                                for f in os.listdir(train_path)
                                                if f.endswith(".pkl")
                                            ),
                                            None,
                                        )
                                        val_pkl = next(
                                            (
                                                f
                                                for f in os.listdir(val_path)
                                                if f.endswith(".pkl")
                                            ),
                                            None,
                                        )
                                        train_paths.append(
                                            os.path.join(train_path, train_pkl)
                                            if train_pkl
                                            else ""
                                        )
                                        val_paths.append(
                                            os.path.join(val_path, val_pkl)
                                            if val_pkl
                                            else ""
                                        )
                                        # also add the metric values again
                                        print(f"Importing CSV from: {file_path}")
                                        data = pd.read_csv(file_path)
                                        for metric in metrics:
                                            all_data[metric].append(
                                                data[f"metrics/{metric}(B)"].values[0]
                                            )
                                        all_paths.append(sub_dir_path)

                        pbar.update(
                            1
                        )  # Update the progress bar for each file processed

    # Create DataFrame for paths
    result_data = {
        "train_embedding_path": train_paths,
        "val_embedding_path": val_paths,
        "yolo_path": all_paths,
        "train_size": [len(pickle.load(open(path, "rb"))) for path in train_paths],
        "val_size": [len(pickle.load(open(path, "rb"))) for path in val_paths],
    }
    for metric in metrics:
        result_data[metric] = all_data[metric]

    result = pd.DataFrame(result_data)

    # save result to output path
    result.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_path", type=str, help="Root path to search for eval_metrics.csv files"
    )
    parser.add_argument(
        "--output_path", type=str, help="Output path for the merged csv"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["test", "val"],
        default="test",
        help="Split to search for eval_metrics.csv files",
    )
    parser.add_argument(
        "--metric",
        type=str,
        nargs="+",
        default=["mAP50-95"],
        help="Metrics to extract from the eval_metrics.csv files",
    )
    args = parser.parse_args()

    create_csv(args.root_path, args.output_path, args.split, args.metric)


if __name__ == "__main__":
    main()
