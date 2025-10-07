"""
Version: 2024-10-01 13:20
"""

import json
import os
import shutil
import subprocess
import time
from typing import Any

import pandas as pd
import torch


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def get_time(self):
        return time.time() - self.start_time

    def get_time_formatted(self):
        return time.strftime("%H:%M:%S", time.gmtime(self.get_time()))


class SDQM_Outputter:
    def __init__(
        self,
        output_directory: str,
        git_hash: bool = False,
        gpu_info: int | list[int] = -1,
    ) -> None:
        assert os.path.isdir(output_directory), f"{output_directory} is not a directory"
        self._output_directory: str = output_directory
        self._results: dict = {}
        self._running_timers: list[list[str]] = []

        if git_hash:
            self._add_git_hash()
        if gpu_info != -1:
            self._add_gpu_info(gpu_info)

    def _add_git_hash(self) -> None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        try:
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=script_dir)
                .strip()
                .decode()
            )
        except subprocess.CalledProcessError:
            print("Could not get git commit hash")
            commit_hash = "N/A"

        self.add_result("git_commit_hash", commit_hash)

    def _add_gpu_info(self, devices: int | list[int] = 0) -> None:
        if isinstance(devices, int):
            devices = [devices]

        gpu_info = {}
        for device in devices:
            if torch.cuda.is_available():
                gpu_info[device] = {
                    "id": device,
                    "name": torch.cuda.get_device_name(device),
                    "memory_total": torch.cuda.get_device_properties(
                        device
                    ).total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(device),
                    "memory_cached": torch.cuda.memory_reserved(device),
                }
            else:
                print("CUDA is not available")
                gpu_info[device] = {
                    "id": device,
                    "name": "N/A",
                    "memory_total": "N/A",
                    "memory_allocated": "N/A",
                    "memory_cached": "N/A",
                } #

        self.add_result("gpu_info", gpu_info)

    def create_timer(self, name: str | list[str]) -> None:
        if isinstance(name, str):
            name = [name]
        self.add_result(name, Timer())
        self._running_timers.append(name)

    def end_timer(self, name: str | list[str]) -> None:
        if isinstance(name, str):
            name = [name]

        def get_nested_value(d, keys):
            for key in keys:
                d = d[key]
            return d

        timer = get_nested_value(self._results, name)
        self.add_result(name, timer.get_time_formatted())

        # Remove the timer from self._running_timers
        self._running_timers.remove(name)

    def add_result(self, key: str | list[str], value: Any) -> None:
        def update_nested_value(d, keys, value):
            for key in keys[:-1]:  # Iterate over all keys except the last one
                if key not in d:
                    d[key] = {}  # Create a new dictionary if the key doesn't exist
                d = d[key]  # Move deeper into the dictionary
            d[keys[-1]] = value  # Set the value at the deepest level

        if isinstance(key, list):
            update_nested_value(self._results, key, value)
        else:
            self._results[key] = value

    def _prepare_to_save(self) -> None:
        os.makedirs(self._output_directory, exist_ok=True)

        for timer in self._running_timers:
            self.end_timer(timer)

    def save_json(self) -> None:
        output_path = os.path.join(self._output_directory, "results.json")
        with open(output_path, "w") as f:
            json.dump(self._results, f, indent=4)

    def update_csv(self) -> None:
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
        flat_results = flatten_dict(self._results)
        column_titles = flat_results.keys()
        csv_path = os.path.join(self._output_directory, "results.csv")

        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            df = None

        if df is None:
            df = pd.DataFrame(columns=column_titles)
        else:
            # Check if the column titles are the same
            if list(df.columns) != list(column_titles):
                df = None
                old_csv_path = os.path.join(self._output_directory, "results_old.csv")
                counter = 1
                while os.path.exists(old_csv_path):
                    old_csv_path = os.path.join(
                        self._output_directory, f"results_old_{counter}.csv"
                    )
                    counter += 1
                print(
                    f"Column titles have changed. Moving {csv_path} to {old_csv_path} and creating a new results.csv"
                )
                shutil.move(csv_path, old_csv_path)

        df = pd.concat([df, pd.DataFrame([flat_results])], ignore_index=True)
        df.to_csv(csv_path, index=False)
