import argparse
import json
import os
import pickle
import shutil
import time

import numpy
import pandas as pd
import yaml

from log_cluster_metric import compute_log_cluster_metric
from separate_model import separate


def append_results_to_csv(results: dict, output_dir: str) -> None:
    # Version: 2024-09-24 13:44
    """Add the values in a dictionary to results.csv

    Args:
        results dict: Dictionary containing the values to be added to results.csv
        output_dir str: Directory under which results.csv is located

    Returns:
        None
    """

    def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
        """Flatten a nested dictionary, concatenating subkeys with parent keys

        Args:
            d (dict): Dictionary to be flattened
            parent_key (str, optional): Base key string to prepend to each key. Defaults to "".
            sep (str, optional): Separator to use between parent and child keys. Defaults to "_".

        Returns:
            dict: A new dictionary with flattened keys
        """

        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # Flatten the nested dictionary
    flat_results = flatten_dict(results)
    column_titles = flat_results.keys()
    csv_path = os.path.join(output_dir, "results.csv")

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        df = None

    if df is None:
        df = pd.DataFrame(columns=column_titles)
    else:
        # Check if the column titles are the same
        if list(df.columns) != list(column_titles):
            print(
                "Column titles do not match. Moving results.csv to results_old.csv before outputting."
            )
            df = None
            shutil.move(csv_path, os.path.join(output_dir, "results_old.csv"))

    df = pd.concat([df, pd.DataFrame([flat_results])], ignore_index=True)
    df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--synthetic_embeddings", type=str, required=True)
    parser.add_argument("-r", "--real_embeddings", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-k", "--clusters", type=int, required=True)
    parser.add_argument("-m", "--cluster_metric", type=str, required=True)
    parser.add_argument("-p", "--max_params", type=int, default=None)
    args = parser.parse_args()

    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "autokeras"), exist_ok=True)

    # Time the execution
    start_time = time.time()

    # Must both be numpy arrays of size (n, m)
    # where n is the number of embeddings
    # and m is the size of the embeddings
    synthetic_embeddings = pickle.load(open(args.synthetic_embeddings, "rb"))
    real_embeddings = pickle.load(open(args.real_embeddings, "rb"))

    # Perform the separation with AutoKeras model
    num_params, best_accuracy = separate(
        synthetic_embeddings, real_embeddings, args.output_dir, args.max_params
    )

    # Perform clustering and calculate the log cluster metric
    normalized_log_cluster_metric, log_cluster_metric = compute_log_cluster_metric(
        synthetic_embeddings, real_embeddings, args.clusters, args.cluster_metric
    )

    # Compute Separability Score
    c = 1000
    separability_score = normalized_log_cluster_metric * best_accuracy + num_params / c

    execution_time = time.time() - start_time

    results = {
        "separability_score": separability_score,
        "normalized_log_cluster_metric": normalized_log_cluster_metric,
        "log_cluster_metric": log_cluster_metric,
        "best_accuracy": best_accuracy,
        "num_params": num_params,
        "c": c,
        "cluster_metric": args.cluster_metric,
        "num_clusters": args.clusters,
        "synthetic_embeddings": args.synthetic_embeddings,
        "synthetic_embeddings_size": synthetic_embeddings.shape,
        "real_embeddings": args.real_embeddings,
        "real_embeddings_size": real_embeddings.shape,
        "output_dir": args.output_dir,
        "execution_time": execution_time,
    }

    # Save as YAML and JSON under the output directory
    with open(os.path.join(args.output_dir, "results.yaml"), "w") as f:
        yaml.dump(results, f)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Also append to CSV file under the output directory
    # Use Pandas and also check if the file exists
    append_results_to_csv(results, args.output_dir)


if __name__ == "__main__":
    main()
