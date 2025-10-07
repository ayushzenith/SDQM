import argparse
from dataset_selection.select_datasets import run as select_datasets
from dataset_selection.utils.aggregate_custom_split import (
    dimo_scene,
    rareplanes_real_scene,
    rareplanes_synthetic_scene,
    wasabi_scene,
    wasabi_synthetic_scene,
)
import os
import pandas as pd
from regression import run_regression
from sdqm import calculate_sdqm

def replicate_experiment(
    wasabi_real_yolo_dir: str,
    wasabi_synthetic_yolo_dir: str,
    dimo_real_yolo_dir: str,
    dimo_synthetic_yolo_dir: str,
    rareplanes_real_yolo_dir: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "wasabi"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "dimo"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rareplanes"), exist_ok=True)
    
    select_datasets(
        10,
        wasabi_real_yolo_dir,
        wasabi_synthetic_yolo_dir,
        os.path.join(output_dir, "wasabi"),
        ["train", "val", "eval", "test"],
        {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
        },
        None,
        wasabi_scene,
        wasabi_scene,
        "wasabi",
    )
    select_datasets(
        10,
        dimo_real_yolo_dir,
        dimo_synthetic_yolo_dir,
        os.path.join(output_dir, "dimo"),
        ["train", "val", "eval", "test"],
        {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
        },
        None,
        dimo_scene,
        dimo_scene,
        "dimo",
    )
    select_datasets(
        10,
        rareplanes_real_yolo_dir,
        rareplanes_real_yolo_dir,
        os.path.join(output_dir, "rareplanes"),
        ["train", "val", "eval", "test"],
        {
            "train": 0.8,
            "val": 0.1,
            "test": 0.1,
        },
        None,
        rareplanes_real_scene,
        rareplanes_real_scene,
        "rareplanes",
    )

    csv_files = []
    for dataset in ["wasabi", "dimo", "rareplanes"]:
        for split in ["train", "val"]:
            df_files = pd.read_csv(os.path.join(output_dir, dataset, f"{split}_map50.csv"))
            real_files = df_files.iloc[:, 1].tolist()
            synthetic_files = df_files.iloc[:, 0].tolist()

            calculate_sdqm(real_files, synthetic_files, None, df_files, os.path.join(output_dir, f"{dataset}_{split}_dsqm.csv", ["all"], dataset, None))
            csv_files.append(os.path.join(output_dir, f"{dataset}_{split}_dsqm.csv"))


    
    args = argparse.Namespace()
    args.all_methods = True
    args.input_path = csv_files
    args.start_column = 1
    args.y_column = 0
    args.shuffle_split = True
    args.test_size = 0.2
    args.last = False
    args.val_input_path = []
    args.output_path = os.path.join(output_dir, "regression_results")
    args.load_results = False
    args.standardize = False
    args.pca = False
    args.method = "all"
    args.k_folds = None
    args.sequential_test = False
    args.scaler = None
    args.separately_scale = False
    args.correlation_threshold = 0.2
    args.sequential_test = False
    args.scaler = False
    args.separately_scale = False

    run_regression(args)
    



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wasabi_real_yolo_dir", type=str, help="Path to the YOLO directory for the WASABI Real dataset"
    )
    parser.add_argument(
        "--wasabi_synthetic_yolo_dir", type=str, help="Path to the YOLO directory for the WASABI Synthetic dataset"
    )
    parser.add_argument("--dimo_real_yolo_dir", type=str, help="Path to the YOLO directory for the DIMO Real dataset")
    parser.add_argument(
        "--dimo_synthetic_yolo_dir", type=str, help="Path to the YOLO directory for the DIMO Synthetic dataset"
    )
    parser.add_argument(
        "--rareplanes_real_yolo_dir", type=str, help="Path to the YOLO directory for the RarePlanes Real dataset"
    )
    parser.add_argument(
        "--rareplanes_synthetic_yolo_dir", type=str, help="Path to the YOLO directory for the RarePlanes Real dataset"
    )
    parser.add_argument("--output_dir", type=str, default="experiment", help="Path to the output directory")
    args = parser.parse_args()
    
    replicate_experiment(
        args.wasabi_real_yolo_dir,
        args.wasabi_synthetic_yolo_dir,
        args.dimo_real_yolo_dir,
        args.dimo_synthetic_yolo_dir,
        args.rareplanes_real_yolo_dir,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
