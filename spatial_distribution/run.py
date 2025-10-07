import argparse
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import yaml

from heatmap_comparison import HeatmapComparison
from spatial_distribution import SpatialDistribution


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


def time_function_call(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time, result


def create_heatmap(
    annotations_path: str, image_shape: tuple, pool_size: int
) -> tuple[dict, np.ndarray]:
    metadata = dict()
    spatial_distribution = SpatialDistribution(annotations_path, image_shape)

    start_time = time.time()
    spatial_distribution.create_heatmap()
    if pool_size > 1:
        spatial_distribution.mean_pool(pool_size)
    end_time = time.time()

    metadata["execution_time"] = str(end_time - start_time)
    metadata["annotations_path"] = annotations_path
    metadata["image_shape"] = ",".join(map(str, image_shape))
    metadata["pool_size"] = pool_size

    return metadata, spatial_distribution.heatmap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a heatmap of the spatial distribution of bounding boxes"
    )
    parser.add_argument(
        "annotations_path1",
        type=str,
        help="Path to the first directory containing YOLO annotations",
    )
    parser.add_argument(
        "annotations_path2",
        type=str,
        help="Path to the second directory containing YOLO annotations",
    )
    parser.add_argument(
        "--image_shape",
        type=int,
        nargs=2,
        required=True,
        help="Shape of the images in the dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the comparison results. Do not include file extension",
    )
    parser.add_argument(
        "--pool_size",
        type=int,
        default=1,
        help="Size of the pooling window for mean pooling",
    )
    args = parser.parse_args()

    compare_dataset_heatmaps(
        args.annotations_path1,
        args.annotations_path2,
        args.image_shape,
        args.output_path,
        args.pool_size,
    )

def calculate_spatial_distribution(
    annotations_path1: str | list[str],
    annotations_path2: str | list[str],
    image_shape: tuple[float, float],
) -> float:
    image_shape = tuple(image_shape)

    dist1 = SpatialDistribution(annotations_path1, image_shape)
    dist2 = SpatialDistribution(annotations_path2, image_shape)
    
    dist1.create_heatmap()
    dist2.create_heatmap()
    
    # mean pool
    dist1.mean_pool(image_shape[0] // 8)
    dist2.mean_pool(image_shape[0] // 8)

    rmse = HeatmapComparison(dist1.heatmap, dist2.heatmap).RMSE()

    return rmse


def compare_dataset_heatmaps(
    annotations_path1: str,
    annotations_path2: str,
    image_shape: tuple[float, float],
    output_path: str,
    pool_size: int,
):
    image_shape = tuple(image_shape)

    metadata = dict()

    metadata1, heatmap1 = create_heatmap(annotations_path1, image_shape, pool_size)
    metadata2, heatmap2 = create_heatmap(annotations_path2, image_shape, pool_size)

    comparison = HeatmapComparison(heatmap1, heatmap2)

    mse_time, mse = time_function_call(comparison.MSE)
    rmse_time, rmse = time_function_call(comparison.RMSE)
    ssim_time, ssim = time_function_call(comparison.SSIM)
    psnr_time, psnr = time_function_call(comparison.PSNR)

    metadata["dataset1"] = metadata1
    metadata["dataset2"] = metadata2
    metadata["comparison"] = {
        "MSE": {"value": str(mse), "execution_time": str(mse_time)},
        "RMSE": {"value": str(rmse), "execution_time": str(rmse_time)},
        "SSIM": {"value": str(ssim), "execution_time": str(ssim_time)},
        "PSNR": {"value": str(psnr), "execution_time": str(psnr_time)},
    }

    with open(output_path + ".json", "w") as f:
        json.dump(metadata, f, indent=4)

    with open(output_path + ".yaml", "w") as f:
        yaml.dump(metadata, f)

    append_results_to_csv(metadata, os.path.dirname(output_path))


if __name__ == "__main__":
    main()
