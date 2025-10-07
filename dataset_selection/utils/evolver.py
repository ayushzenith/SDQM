import argparse
import json
import os
import pickle
import shutil
from abc import abstractmethod
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from deap import base, creator, tools
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils.deap_utils import (
    eaSimple_ckpt,
    initialize_no_overlap_individual,
    initialize_overlap_individual,
    mutSwpUniformInt,
)
from utils.metrics import (
    calculate_alpha_precision,
    calculate_authenticity,
    calculate_beta_recall,
    calculate_cluster_metric,
    calculate_frontier_integral,
    calculate_frontier_integral_star,
    calculate_mauve,
    calculate_mauve_star,
)
from utils.sqdm_tools import SDQM_Outputter


class Metric:
    def __init__(self, metric_function: Callable, metric_name: str, range: List[float]):
        assert len(range) == 2, "Range list should contain two floats"
        self.metric_function = metric_function
        self.name = metric_name
        self.range = range

    def __call__(self, embeddings1, embeddings2):
        return self.metric_function(embeddings1, embeddings2)

    def __str__(self):
        return self.name


ALL_METRICS = [
    Metric(calculate_alpha_precision, "Alpha Precision", [0, 1]),
    Metric(calculate_beta_recall, "Beta Recall", [0, 1]),
    Metric(calculate_authenticity, "Authenticity", [0, 1]),
    Metric(calculate_mauve, "Mauve", [0, 1]),
    Metric(calculate_mauve_star, "Mauve*", [0, 1]),
    Metric(calculate_frontier_integral, "Frontier Integral", [0, 1]),
    Metric(calculate_frontier_integral_star, "Frontier Integral*", [0, 1]),
    Metric(calculate_cluster_metric, "Cluster Metric", [0, 1]),
]


class DatasetSelectEvolver:
    def __init__(
        self,
        dataset1: Dict[List[float], List[str]],
        dataset2: Dict[List[float], List[str]],
        output_dir: str,
        n: int,
        k_l: int,
        k_u: int,
        metrics: List[Metric],
        num_generations: int = 10,
        mutpb: float = 0.5,
        cxpb: float = 0.5,
        value_indices: List[int] = [],
        pop_size: int = 300,
        multi_target: bool = False,
    ):
        """
        :param dataset1: A dictionary containing the embeddings and file list for dataset 1
        :param dataset2: A dictionary containing the embeddings and file list for dataset 2
        :param output_dir: The directory to save the output results
        :param n: The number of values to optimize for
        :param k_l: The lower bound of the number of images to select from the two datasets
        :param k_u: The upper bound of the number of images to select from the two datasets
        :param metrics: A list of Metric objects to optimize for
        :param num_generations: The number of generations to run the evolution for
        :param mutpb: The mutation probability
        :param cxpb: The crossover probability
        :param value_indices: The indices of the values to optimize for in np.linspace(metric.range[0], metric.range[1], n)
        :param pop_size: The population size
        """
        self.population = [dataset1["file_list"], dataset2["file_list"]]
        self.range = [k_l, k_u]
        self.output = output_dir
        self.results = {}
        self.fitness_record = []
        self.logbooks = {}
        self.metrics = metrics
        self.num_generations = num_generations
        self.mutpb = mutpb
        self.cxpb = cxpb
        self.pop_size = pop_size
        self.n = n
        self.value_indices = value_indices
        self.multi_target = multi_target

        # Get the embeddings for the images in the two datasets
        self.embedding_list = [
            dataset1["embeddings"],
            dataset2["embeddings"],
        ]

    def run(self):
        for i, metric in enumerate(self.metrics):
            print(f"Running for metric {metric}")
            self.results[str(metric)] = {}

            self.linspace = np.linspace(metric.range[0], metric.range[1], self.n)
            if self.value_indices:
                self.linspace = [self.linspace[i] for i in self.value_indices]

            values = self.linspace if not self.multi_target else [1]

            for target_value in values:
                self.current_target = target_value
                self.current_metric = metric
                toolbox = self._create_toolbox(
                    metric
                )  # Create the toolbox for the current metric function
                ckpt_dir = os.path.join(
                    self.output, "checkpoints", f"{str(metric)}_{target_value}"
                )

                hofs, logbook = self._evolve(toolbox, ckpt_dir)  # Evolve the population

                # Save the best individuals
                for hof in hofs:
                    for individual in hof:
                        embeddings1, embeddings2 = self._individual_to_selection(
                            individual, self.embedding_list
                        )
                        files1, files2 = self._individual_to_selection(
                            individual, self.population
                        )
                        self.results[str(metric)][metric(embeddings1, embeddings2)] = [
                            files1,
                            files2,
                        ]

                # plot the fitness
                os.makedirs(os.path.join(self.output, "plots"), exist_ok=True)
                plt.plot(logbook.select("min"))
                plt.plot(logbook.select("mean"))
                plt.plot(logbook.select("max"))
                plt.xlabel("Generation")
                plt.ylabel("Fitness")
                plt.title("Fitness over generations")
                plt.legend(["min", "mean", "max"])
                plt.savefig(
                    os.path.join(
                        self.output,
                        "plots",
                        f"fitness_{str(metric)}_{target_value}.png",
                    )
                )

                # save the logbook as csv
                logbook_df = pd.DataFrame(logbook)
                logbook_df.to_csv(
                    os.path.join(
                        self.output,
                        f"logbook_{str(metric)}_{target_value}.csv",
                    )
                )

    def select_subsets_for_metric(
        self, symlink: bool = False, yolo: bool = False
    ) -> None:
        result_dir = os.path.join(self.output, "result_dicts")
        os.makedirs(result_dir, exist_ok=True)

        # Sort the results by the metric value
        for metric in self.results:
            self.results[metric] = dict(sorted(self.results[metric].items()))

        # Save the results to a file
        with open(os.path.join(result_dir, "all_results.pkl"), "wb") as f:
            pickle.dump(self.results, f)
        # JSON
        with open(os.path.join(result_dir, "all_results.json"), "w") as f:
            json.dump(self.results, f)

        # For each metric, select the n subsets closest to the values
        # in np.linspace(metric.range[0], metric.range[1], self.n)
        print(f"Exporting subsets for metric {metric}")
        for metric in self.metrics:
            results = self.results[str(metric)]
            target_values = (
                [
                    np.linspace(metric.range[0], metric.range[1], self.n)[i]
                    for i in self.value_indices
                ]
                if self.value_indices
                else np.linspace(metric.range[0], metric.range[1], self.n)
            )
            selected_subsets = {}

            for target_value in target_values:
                closest_value = min(
                    results.keys(), key=lambda x: abs(float(x) - target_value)
                )
                selected_subsets[target_value] = results[closest_value]

            # Save the selected subsets to a file
            with open(
                os.path.join(result_dir, f"{str(metric)}_results.pkl"), "wb"
            ) as f:
                pickle.dump(selected_subsets, f)
            # JSON
            with open(
                os.path.join(result_dir, f"{str(metric)}_results.json"), "w"
            ) as f:
                json.dump(selected_subsets, f, indent=4)

        # Export the selected subsets to a directory
        selected_subsets_dir = os.path.join(self.output, "selected_subsets")
        os.makedirs(selected_subsets_dir, exist_ok=True)

        for metric in self.metrics:
            with open(
                os.path.join(result_dir, f"{str(metric)}_results.pkl"), "rb"
            ) as f:
                selected_subsets = pickle.load(f)

            for target_value in selected_subsets:
                subset_dir = os.path.join(
                    selected_subsets_dir, f"{str(metric)}_{target_value}"
                )
                os.makedirs(subset_dir, exist_ok=True)
                if yolo:
                    # Create the dataset structure
                    for split in ["train", "val"]:
                        os.makedirs(
                            os.path.join(subset_dir, "images", split), exist_ok=True
                        )
                        os.makedirs(
                            os.path.join(subset_dir, "labels", split), exist_ok=True
                        )
                    # create yaml file
                    original_dir1 = os.path.dirname(
                        os.path.dirname(os.path.dirname(self.population[0][0]))
                    )
                    yaml_files = [
                        os.path.join(original_dir1, file)
                        for file in os.listdir(original_dir1)
                        if file.endswith(".yaml")
                    ]
                    if not yaml_files:
                        print("No .yaml file found in yolo_dir. Not creating data.yaml")
                    else:
                        with open(yaml_files[0], "r") as f:
                            original_data = yaml.load(f, Loader=yaml.FullLoader)
                            original_classes = original_data["names"]
                        with open(os.path.join(subset_dir, "data.yaml"), "w") as f:
                            yaml.dump(
                                {
                                    "path": subset_dir,
                                    "train": "images/train",
                                    "val": "images/val",
                                    "names": original_classes,
                                },
                                f,
                            )

                image_dir1 = (
                    os.path.join(subset_dir, "images", "train")
                    if yolo
                    else os.path.join(subset_dir, "1")
                )
                image_dir2 = (
                    os.path.join(subset_dir, "images", "val")
                    if yolo
                    else os.path.join(subset_dir, "2")
                )

                for i, subset in enumerate(selected_subsets[target_value]):
                    image_dir = image_dir1 if i == 0 else image_dir2
                    os.makedirs(image_dir, exist_ok=True)
                    for file in tqdm(
                        subset, desc=f"Copying files for {str(metric)}_{target_value}"
                    ):
                        if symlink:
                            os.symlink(
                                file,
                                os.path.join(image_dir, os.path.basename(file)),
                            )
                            if yolo:
                                label = (
                                    file.replace("images", "labels")
                                    .replace(".jpg", ".txt")
                                    .replace(".png", ".txt")
                                )
                                os.symlink(
                                    label,
                                    os.path.join(
                                        image_dir.replace("images", "labels"),
                                        os.path.basename(label),
                                    ),
                                )
                        else:
                            shutil.copy(
                                file,
                                os.path.join(image_dir, os.path.basename(file)),
                            )
                            if yolo:
                                label = (
                                    file.replace("images", "labels")
                                    .replace(".jpg", ".txt")
                                    .replace(".png", ".txt")
                                )
                                shutil.copy(
                                    label,
                                    os.path.join(
                                        image_dir.replace("images", "labels"),
                                        os.path.basename(label),
                                    ),
                                )

    def _evolve(
        self, toolbox: base.Toolbox, ckpt_dir: str
    ) -> Tuple[tools.HallOfFame, tools.Logbook]:
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("mean", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        stats.register("median", np.median)
        stats.register("iqr", lambda x: np.percentile(x, 75) - np.percentile(x, 25))
        if os.path.exists(ckpt_dir):
            latest_ckpt = max(
                [int(f.split("_")[-1].split(".")[0]) for f in os.listdir(ckpt_dir)]
            )
            print(f"Resuming from checkpoint {latest_ckpt} in {ckpt_dir}")

            pop, logbook, hofs = eaSimple_ckpt(
                None,
                None,
                cxpb=None,
                mutpb=None,
                ngen=self.num_generations,
                values=None,
                stats=stats,
                verbose=True,
                save_dir=ckpt_dir,
                checkpoint=os.path.join(ckpt_dir, f"checkpoint_{latest_ckpt}.pkl"),
                min_fitness_threshold=0.05,
            )

        else:
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            population = toolbox.population(n=self.pop_size)  # Population size

            ckpt_save_dir = os.path.join(self.output, "checkpoints")
            os.makedirs(ckpt_save_dir, exist_ok=True)

            values = self.linspace if self.multi_target else [self.current_target]

            pop, logbook, hofs = eaSimple_ckpt(
                population,
                toolbox,
                cxpb=self.cxpb,
                mutpb=self.mutpb,
                ngen=self.num_generations,
                values=values,
                stats=stats,
                verbose=True,
                save_dir=ckpt_save_dir,
                min_fitness_threshold=0.05,
            )

        return hofs, logbook

    def _evaluate_individual(
        self,
        individual: List,
        metric: Metric,
        values: List[float],
        return_closest: bool = False,
    ) -> Tuple[float] | float:
        embeddings1, embeddings2 = self._individual_to_selection(
            individual, self.embedding_list
        )
        metric_value = metric(embeddings1, embeddings2)
        if self.multi_target:
            closest_value = values[
                np.argmin([abs(metric_value - target) for target in values])
            ]
            fitness = abs(closest_value - metric_value)
        else:
            fitness = abs(metric_value - self.current_target)

        # Penalize for being outside of range
        fitness += abs(
            self._range_penalty(individual)
            / (self.range[1] + self.range[0])
            / 2
            * (self.current_metric.range[1] - self.current_metric.range[0])
        )

        self.fitness_record.append(fitness)

        if not return_closest:
            return (fitness,)
        else:
            return closest_value

    @abstractmethod
    def _create_toolbox(
        self,
        metric: Metric,
    ) -> base.Toolbox:
        pass

    @abstractmethod
    def _individual_to_selection(
        self, individual: List, population: List[str]
    ) -> Tuple[List[str], List[str]]:
        pass

    @abstractmethod
    def _range_penalty(self, individual: List) -> float:
        pass


class DatasetSelectEvolverNoOverlap(DatasetSelectEvolver):
    def _create_toolbox(
        self,
        metric: Metric,
    ) -> base.Toolbox:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        toolbox.register(
            "individual",
            initialize_no_overlap_individual,
            creator.Individual,
            n_1=len(self.population[0]),
            n_2=len(self.population[1]),
            k_l=self.range[0],
            k_u=self.range[1],
        )

        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register(
            "evaluate",
            self._evaluate_individual,
            metric=metric,
        )

        return toolbox

    def _individual_to_selection(
        self, individual: List, population: List[str]
    ) -> Tuple[List[str], List[str]]:
        subset1 = individual[: len(population[0])]
        subset2 = individual[len(population[0]) :]
        assert len(subset1) == len(population[0])
        assert len(subset2) == len(population[1])

        # Select the images from the population marked as "True" in the individual
        selection1 = [population[0][i] for i in range(len(subset1)) if subset1[i]]
        selection2 = [population[1][i] for i in range(len(subset2)) if subset2[i]]

        return selection1, selection2

    def _range_penalty(self, individual: List) -> float:
        len1 = sum(individual[: len(self.population[0])])
        len2 = sum(individual[len(self.population[0]) :])

        return (
            max(0, self.range[0] - len1)
            + max(0, len1 - self.range[1])
            + max(0, self.range[0] - len2)
            + max(0, len2 - self.range[1])
        )


class DatasetSelectEvolverOverlap(DatasetSelectEvolver):
    def _create_toolbox(self, metric: Metric) -> base.Toolbox:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()

        toolbox.register(
            "individual",
            initialize_overlap_individual,
            creator.Individual,
            n_1=len(self.population[0]),
            n_2=len(self.population[1]),
            k_l=self.range[0],
            k_u=self.range[1],
        )

        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutSwpUniformInt, low=0, up=2, swappb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register(
            "evaluate",
            self._evaluate_individual,
            metric=metric,
        )

        return toolbox

    def _individual_to_selection(
        self, individual: List, population: List[str]
    ) -> Tuple[List[str], List[str]]:
        full_population = np.concatenate((population[0], population[1]), axis=0)

        assert len(individual) == len(full_population)

        selection1 = [
            full_population[i] for i in range(len(individual)) if individual[i] == 1
        ]
        selection2 = [
            full_population[i] for i in range(len(individual)) if individual[i] == 2
        ]

        return selection1, selection2

    def _range_penalty(self, individual: List) -> float:
        len1 = sum([1 for i in individual if i == 1])
        len2 = sum([1 for i in individual if i == 2])

        return (
            max(0, self.range[0] - len1)
            + max(0, len1 - self.range[1])
            + max(0, self.range[0] - len2)
            + max(0, len2 - self.range[1])
        )


def main():
    parser = argparse.ArgumentParser(
        description="Select subsets of images from two datasets"
    )
    parser.add_argument(
        "mode",
        type=str,
        choices=["no-overlap", "overlap"],
        help="The mode to use for selecting the subsets",
    )
    parser.add_argument(
        "--dataset1_embeddings_path",
        type=str,
        nargs="+",
        help="The embedding file containing the embeddings for dataset 1",
    )
    parser.add_argument(
        "--dataset2_embeddings_path",
        type=str,
        nargs="+",
        help="The embeddings file containing the embeddings for dataset 2",
    )
    parser.add_argument(
        "--k_l",
        type=int,
        default=1000,
        help="The lower bound of the number of images to select from the two datasets",
    )
    parser.add_argument(
        "--k_u",
        type=int,
        default=2000,
        help="The upper bound of the number of images to select from the two datasets",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=10,
        help="The number of values to optimize for",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.pkl",
        help="The file to save the results to",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="*",
        default=["all"],
        help="The metrics to optimize for",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=10,
        help="The number of generations to run the evolution for",
    )
    parser.add_argument(
        "--mutpb",
        type=float,
        default=0.5,
        help="The mutation probability",
    )
    parser.add_argument(
        "--cxpb",
        type=float,
        default=0.5,
        help="The crossover probability",
    )
    parser.add_argument(
        "--value_indices",
        type=int,
        nargs="*",
        default=[],
        help="The indices of the values to optimize for in np.linspace(metric.range[0], metric.range[1], n)",
    )
    parser.add_argument(
        "--pop_size",
        type=int,
        default=300,
        help="The population size",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Use symlinks instead of copying files",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="The files belong to a YOLO dataset",
    )
    parser.add_argument(
        "--multi_target",
        action="store_true",
        help="Optimize for multiple target values",
    )
    args = parser.parse_args()

    evolve_datasets(
        mode=args.mode,
        dataset1_embeddings_path=args.dataset1_embeddings_path,
        dataset2_embeddings_path=args.dataset2_embeddings_path,
        k_l=args.k_l,
        k_u=args.k_u,
        n=args.n,
        output=args.output,
        metrics=args.metrics,
        num_generations=args.num_generations,
        mutpb=args.mutpb,
        cxpb=args.cxpb,
        value_indices=args.value_indices,
        pop_size=args.pop_size,
        symlink=args.symlink,
        yolo=args.yolo,
        multi_target=args.multi_target,
    )


def evolve_datasets(
    mode: str,
    dataset1_embeddings_path: List[str],
    dataset2_embeddings_path: List[str],
    k_l: int,
    k_u: int,
    n: int,
    output: str,
    metrics: List[str],
    num_generations: int,
    mutpb: float,
    cxpb: float,
    value_indices: List[int],
    pop_size: int,
    symlink: bool,
    yolo: bool,
    multi_target: bool,
):
    # Set the metrics
    metrics_to_optimize = (
        ALL_METRICS
        if "all" in metrics
        else [metric for metric in ALL_METRICS if metric.name in metrics]
    )

    # Check if corresponding CSV files exist
    csv1_paths = [path.replace(".pkl", ".csv") for path in dataset1_embeddings_path]
    csv2_paths = [path.replace(".pkl", ".csv") for path in dataset2_embeddings_path]

    if not all([os.path.exists(csv1_path) for csv1_path in csv1_paths]):
        raise FileNotFoundError("CSV file not found for dataset 1")
    if not all([os.path.exists(csv2_path) for csv2_path in csv2_paths]):
        raise FileNotFoundError("CSV file not found for dataset 2")

    # Load the embeddings
    dataset1_embeddings = []
    for path in dataset1_embeddings_path:
        with open(path, "rb") as f:
            dataset1_embeddings.append(pickle.load(f))
    dataset1_embeddings = np.concatenate(dataset1_embeddings, axis=0)

    dataset2_embeddings = []
    for path in dataset2_embeddings_path:
        with open(path, "rb") as f:
            dataset2_embeddings.append(pickle.load(f))
    dataset2_embeddings = np.concatenate(dataset2_embeddings, axis=0)

    # Load the file lists
    csv1s = [pd.read_csv(csv_path) for csv_path in csv1_paths]
    csv2s = [pd.read_csv(csv_path) for csv_path in csv2_paths]
    file_list1 = []
    for csv1 in csv1s:
        file_list1 += csv1["file_path"].tolist()
    file_list2 = []
    for csv2 in csv2s:
        file_list2 += csv2["file_path"].tolist()

    # Ensure the files belong to a YOLO dataset
    if yolo:
        first_file = file_list1[0]
        if not os.path.dirname(os.path.dirname(first_file)).endswith(
            "images"
        ) and not os.path.exists(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(first_file))), "labels"
            )
        ):
            raise ValueError("The files do not belong to a YOLO dataset")

    # Combine the embeddings and file lists
    dataset1 = {
        "embeddings": dataset1_embeddings,
        "file_list": file_list1,
    }
    dataset2 = {
        "embeddings": dataset2_embeddings,
        "file_list": file_list2,
    }

    if mode == "no-overlap":
        evolver = DatasetSelectEvolverNoOverlap(
            dataset1,
            dataset2,
            output,
            n,
            k_l,
            k_u,
            metrics_to_optimize,
            num_generations,
            mutpb,
            cxpb,
            value_indices,
            pop_size,
            multi_target,
        )
    elif mode == "overlap":
        evolver = DatasetSelectEvolverOverlap(
            dataset1,
            dataset2,
            output,
            n,
            k_l,
            k_u,
            metrics_to_optimize,
            num_generations,
            mutpb,
            cxpb,
            value_indices,
            pop_size,
            multi_target,
        )

    os.makedirs(output, exist_ok=True)

    outputter = SDQM_Outputter(output, git_hash=True)
    outputter.add_result("dataset1_embeddings_path", dataset1_embeddings_path)
    outputter.add_result("dataset2_embeddings_path", dataset2_embeddings_path)
    outputter.add_result("dataset1_size", len(dataset1["embeddings"]))
    outputter.add_result("dataset2_size", len(dataset2["embeddings"]))
    outputter.add_result("output_dir", output)
    outputter.add_result("symlink", symlink)
    outputter.add_result("mode", mode)
    outputter.add_result("k_l", k_l)
    outputter.add_result("k_u", k_u)
    outputter.add_result("num_generations", num_generations)
    outputter.add_result("mutpb", mutpb)
    outputter.add_result("cxpb", cxpb)
    outputter.add_result("pop_size", pop_size)
    outputter.add_result("metrics", [str(metric) for metric in metrics_to_optimize])
    outputter.add_result("value_indices", value_indices)
    outputter.add_result("multi_target", multi_target)
    outputter.add_result("n", n)

    outputter.create_timer("evolution_time")
    evolution_results_dict = evolver.run()
    outputter.end_timer("evolution_time")

    outputter.add_result("evolution_results", evolution_results_dict)

    outputter.create_timer("export_subsets_time")
    evolver.select_subsets_for_metric(symlink=symlink, yolo=yolo)
    outputter.end_timer("export_subsets_time")

    outputter.save_json()
    outputter.update_csv()


if __name__ == "__main__":
    main()
