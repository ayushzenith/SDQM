import argparse
import json
import os

import pandas as pd

from run import compare_dataset_heatmaps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare metric values between datasets",
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        required=True,
        help="Names of the datasets to compare",
    )
    parser.add_argument(
        "--annotations_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the directories containing YOLO annotations",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the comparison results",
    )
    args = parser.parse_args()

    output_columns=[
        "Dataset 1",
        "Dataset 2",
        "MSE",
        "RMSE",
        "SSIM",
        "PSNR",
        "MSE (pool2)",
        "RMSE (pool2)",
        "SSIM (pool2)",
        "PSNR (pool2)",
        "MSE (pool4)",
        "RMSE (pool4)",
        "SSIM (pool4)",
        "PSNR (pool4)",
        "MSE (pool8)",
        "RMSE (pool8)",
        "SSIM (pool8)",
        "PSNR (pool8)",
        "MSE (pool32)",
        "RMSE (pool32)",
        "SSIM (pool32)",
        "PSNR (pool32)",
        "MSE (pool64)",
        "RMSE (pool64)",
        "SSIM (pool64)",
        "PSNR (pool64)",
    ]
    output_rows = []
    for name1, path1 in zip(args.dataset_names, args.annotations_paths):
        for name2, path2 in zip(args.dataset_names, args.annotations_paths):
            if name1 != name2:
                compare_dataset_heatmaps(
                    path1,
                    path2,
                    (512, 512),
                    os.path.join(args.output_dir, f"{name1}_{name2}_pool1"),
                    1,
                )
                compare_dataset_heatmaps(
                    path1,
                    path2,
                    (512, 512),
                    os.path.join(args.output_dir, f"{name1}_{name2}_pool2"),
                    2,
                )
                compare_dataset_heatmaps(
                    path1,
                    path2,
                    (512, 512),
                    os.path.join(args.output_dir, f"{name1}_{name2}_pool4"),
                    4,
                )
                compare_dataset_heatmaps(
                    path1,
                    path2,
                    (512, 512),
                    os.path.join(args.output_dir, f"{name1}_{name2}_pool8"),
                    8,
                )
                compare_dataset_heatmaps(
                    path1,
                    path2,
                    (512, 512),
                    os.path.join(args.output_dir, f"{name1}_{name2}_pool32"),
                    32,
                )
                compare_dataset_heatmaps(
                    path1,
                    path2,
                    (512, 512),
                    os.path.join(args.output_dir, f"{name1}_{name2}_pool64"),
                    64,
                )

                # Load metadata
                with open(
                    os.path.join(args.output_dir, f"{name1}_{name2}_pool1.json"), "r"
                ) as f:
                    metadata1 = json.load(f)
                with open(
                    os.path.join(args.output_dir, f"{name1}_{name2}_pool2.json"), "r"
                ) as f:
                    metadata2 = json.load(f)
                with open(
                    os.path.join(args.output_dir, f"{name1}_{name2}_pool4.json"), "r"
                ) as f:
                    metadata4 = json.load(f)
                with open(
                    os.path.join(args.output_dir, f"{name1}_{name2}_pool8.json"), "r"
                ) as f:
                    metadata8 = json.load(f)
                with open(
                    os.path.join(args.output_dir, f"{name1}_{name2}_pool32.json"), "r"
                ) as f:
                    metadata32 = json.load(f)
                with open(
                    os.path.join(args.output_dir, f"{name1}_{name2}_pool64.json"), "r"
                ) as f:
                    metadata64 = json.load(f)

                # Add metadata to output
                row = {
                    "Dataset 1": name1,
                    "Dataset 2": name2,
                    "MSE": metadata1["comparison"]["MSE"]["value"],
                    "RMSE": metadata1["comparison"]["RMSE"]["value"],
                    "SSIM": metadata1["comparison"]["SSIM"]["value"],
                    "PSNR": metadata1["comparison"]["PSNR"]["value"],
                    "MSE (pool2)": metadata2["comparison"]["MSE"]["value"],
                    "RMSE (pool2)": metadata2["comparison"]["RMSE"]["value"],
                    "SSIM (pool2)": metadata2["comparison"]["SSIM"]["value"],
                    "PSNR (pool2)": metadata2["comparison"]["PSNR"]["value"],
                    "MSE (pool4)": metadata4["comparison"]["MSE"]["value"],
                    "RMSE (pool4)": metadata4["comparison"]["RMSE"]["value"],
                    "SSIM (pool4)": metadata4["comparison"]["SSIM"]["value"],
                    "PSNR (pool4)": metadata4["comparison"]["PSNR"]["value"],
                    "MSE (pool8)": metadata8["comparison"]["MSE"]["value"],
                    "RMSE (pool8)": metadata8["comparison"]["RMSE"]["value"],
                    "SSIM (pool8)": metadata8["comparison"]["SSIM"]["value"],
                    "PSNR (pool8)": metadata8["comparison"]["PSNR"]["value"],
                    "MSE (pool32)": metadata32["comparison"]["MSE"]["value"],
                    "RMSE (pool32)": metadata32["comparison"]["RMSE"]["value"],
                    "SSIM (pool32)": metadata32["comparison"]["SSIM"]["value"],
                    "PSNR (pool32)": metadata32["comparison"]["PSNR"]["value"],
                    "MSE (pool64)": metadata64["comparison"]["MSE"]["value"],
                    "RMSE (pool64)": metadata64["comparison"]["RMSE"]["value"],
                    "SSIM (pool64)": metadata64["comparison"]["SSIM"]["value"],
                    "PSNR (pool64)": metadata64["comparison"]["PSNR"]["value"],
                }
                output_rows.append(row)
    
    output = pd.DataFrame(output_rows, columns=output_columns)
    output.to_csv(os.path.join(args.output_dir, "comparison_results.csv"), index=False)


if __name__ == "__main__":
    main()
