import argparse
import json
import os
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd

from bounding_box_analysis import compare_stats as compare_distributions


class DistributionLoader(ABC):
    def __init__(self):
        self.distributions = self.get_distributions()

    @abstractmethod
    def get_distributions(self) -> list:
        pass


class AnnotationCountLoader(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_object_counts(self) -> dict:
        pass


class YOLOAnnotationCountLoader(AnnotationCountLoader):
    def __init__(self, label_dir: Union[str, list[str]]):
        super().__init__()
        # label_dir should be the path to the directory containing the YOLO labels
        # or a list of label files
        if isinstance(label_dir, str):
            self.label_files = [
                os.path.join(label_dir, label) for label in os.listdir(label_dir)
            ]
        elif isinstance(label_dir, list):
            self.label_files = label_dir
        super().__init__()

    def get_object_counts(self) -> dict:
        object_counts = {}
        number_of_classes = 0
        # get length of categories
        for label_file in self.label_files:
            if not label_file.endswith(".txt"):
                continue
            try:
                with open(label_file, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        category_id = int(line.strip().split(" ")[0])
                        if number_of_classes < category_id:
                            number_of_classes = category_id
            except FileNotFoundError:
                print(f"Error reading {label_file}")
        # get lowest category id
        for label_file in self.label_files:
            if not label_file.endswith(".txt"):
                continue
            try:
                with open(label_file, "r") as f:
                    image_object_counts = [0] * (number_of_classes + 1)
                    lines = f.readlines()
                    for line in lines:
                        category_id = int(line.strip().split(" ")[0])
                        image_object_counts[category_id] += 1
                    image_object_counts_str = "_".join(
                        [str(count) for count in image_object_counts]
                    )
                    if image_object_counts_str not in object_counts:
                        object_counts[image_object_counts_str] = 0
                    object_counts[image_object_counts_str] += 1
            except FileNotFoundError:
                print(f"Error reading {label_file}")

        return object_counts


class COCOAnnotationCountLoader:
    def __init__(self, json_path: str):
        # json_path should be the path to the COCO annotations file
        self.data = json.load(open(json_path))
        super().__init__()

    def get_object_counts(self) -> dict:
        # get length of categories
        categories_length = len(self.data["categories"])
        # get lowest category id
        lowest_category_id = min(
            [category["id"] for category in self.data["categories"]]
        )
        object_counts = {}
        for image in self.data["images"]:
            image_object_counts = [0] * categories_length
            annotations = [
                annotation
                for annotation in self.data["annotations"]
                if annotation["image_id"] == image["id"]
            ]
            for annotation in annotations:
                category_id = annotation["category_id"] - lowest_category_id
                image_object_counts[category_id] += 1
            image_object_counts_str = "_".join(
                [str(count) for count in image_object_counts]
            )
            if image_object_counts_str not in object_counts:
                object_counts[image_object_counts_str] = 0
            object_counts[image_object_counts_str] += 1

        return object_counts


class RarePlanesDistributionLoader(DistributionLoader):
    def __init__(self, json_path: str, csv_path: str, image_dir: str):
        # json_path should be the path to the COCO annotations file
        # csv_path should be the path to the RarePlanes_Public_Metadata.csv file
        # image_dir should be the path to the directory containing the images
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.attribute_counts = self.get_attribute_counts()
        self.object_counts = COCOAnnotationCountLoader(json_path).get_object_counts()
        super().__init__()

    def _extract_key(self, name: str) -> str:
        return "_".join(name.split("_")[:2])

    def get_attribute_counts(self) -> dict:
        attribute_counts = {}
        for column in self.data.columns:
            column = str(column).strip()
            attribute_counts[column] = {}
        for image_name in os.listdir(self.image_dir):
            if not image_name.endswith(".png"):
                continue
            key = self._extract_key(image_name)
            metadata = self.data[self.data["image_id"] == key]
            # loop through keys and values
            for key, value in metadata.items():
                key = str(key).strip()
                value = str(value.values[0]).strip()
                if value not in attribute_counts[key]:
                    attribute_counts[key][value] = 0
                attribute_counts[key][value] += 1
        return attribute_counts

    def get_distributions(self) -> list:
        distributions = []
        weather_distribution = {}
        weather_distribution["clear"] = self.attribute_counts["Weather"]["Clear Skies"]
        weather_distribution["cloudy"] = self.attribute_counts["Weather"][
            "Cloud Cover or Haze"
        ]
        weather_distribution["snowy"] = self.attribute_counts["Weather"]["Snow"]
        distributions.append([weather_distribution, 0.5, "Weather"])
        distributions.append([self.object_counts, 0.5, "Object Counts"])
        return distributions


class RarePlanesSyntheticDistributionLoader(DistributionLoader):
    def __init__(self, json_path: str, xml_dir: str):
        self.xml_dir = xml_dir
        self.attribute_counts = self.get_attribute_counts()
        self.object_counts = COCOAnnotationCountLoader(json_path).get_object_counts()
        super().__init__()

    def get_attribute_counts(self) -> dict:
        attribute_counts = {}
        for xml_file in os.listdir(self.xml_dir):
            root = ET.parse(os.path.join(self.xml_dir, xml_file)).getroot()
            parameters = root.find("JSON_Variation_Parameters")
            for parameter in parameters:
                name = parameter.attrib["name"]
                value = parameter.attrib["value"]
                if name not in attribute_counts:
                    attribute_counts[name] = {}
                if value not in attribute_counts[name]:
                    attribute_counts[name][value] = 0
                attribute_counts[name][value] += 1
        return attribute_counts

    def get_distributions(self) -> list:
        distributions = []
        weather_distribution = {}
        weather_distribution["clear"] = self.attribute_counts["CurrentWeather"][
            "Clear Sky"
        ]
        weather_distribution["cloudy"] = self.attribute_counts["CurrentWeather"][
            "Overcast"
        ]
        weather_distribution["cloudy"] += self.attribute_counts["CurrentWeather"][
            "Rain"
        ]
        weather_distribution["snowy"] = self.attribute_counts["CurrentWeather"]["Snow"]
        distributions.append([weather_distribution, 0.5, "Weather"])
        distributions.append([self.object_counts, 0.5, "Object Counts"])
        return distributions


def equalize_distribution_length(distribution1, distribution2):
    # sort dictionaries by key
    distribution1 = dict(sorted(distribution1.items()))
    distribution2 = dict(sorted(distribution2.items()))
    # get unique keys
    keys = set(distribution1.keys()).union(set(distribution2.keys()))
    # fill in missing keys
    for key in keys:
        if key not in distribution1:
            distribution1[key] = 0
        if key not in distribution2:
            distribution2[key] = 0
    return distribution1, distribution2


def main():
    parser = argparse.ArgumentParser(description="Label Overlap Analysis")
    parser.add_argument(
        "--real_json", type=str, help="Path to the COCO annotations file"
    )
    parser.add_argument(
        "--real_csv", type=str, help="Path to the RarePlanes_Public_Metadata.csv file"
    )
    parser.add_argument(
        "--real_images", type=str, help="Path to the directory containing the images"
    )
    parser.add_argument(
        "--synthetic_json", type=str, help="Path to the COCO annotations file"
    )
    parser.add_argument(
        "--synthetic_xml",
        type=str,
        help="Path to the directory containing the synthetic images",
    )
    parser.add_argument("--dataset", type=str, help="Name of the dataset")
    args = parser.parse_args()

    calculate_rareplanes_label_overlap(
        args.real_json,
        args.real_csv,
        args.real_images,
        args.synthetic_json,
        args.synthetic_xml,
        args.dataset,
    )


def calculate_rareplanes_label_overlap(
    real_json, real_csv, real_images, synthetic_json, synthetic_xml, dataset
):
    real_loader = RarePlanesDistributionLoader(real_json, real_csv, real_images)
    synthetic_loader = RarePlanesSyntheticDistributionLoader(
        synthetic_json, synthetic_xml
    )

    real_distributions = real_loader.distributions
    synthetic_distributions = synthetic_loader.distributions

    for real_distribution, synthetic_distribution in zip(
        real_distributions, synthetic_distributions
    ):
        print("Comparing")
        print(real_distribution[2])
        print(synthetic_distribution[2])

        real_distribution, synthetic_distribution = equalize_distribution_length(
            real_distribution[0], synthetic_distribution[0]
        )

        print(real_distribution)
        print(synthetic_distribution)

        real_distribution = list(real_distribution.values())
        synthetic_distribution = list(synthetic_distribution.values())

        results = []

        compare_distributions(
            real_distribution,
            synthetic_distribution,
            "Label Overlap",
            results,
            dataset,
        )

        return results
    

def calculate_yolo_label_overlap(
    real_labels, synthetic_labels
):
    real_loader = YOLOAnnotationCountLoader(real_labels)
    synthetic_loader = YOLOAnnotationCountLoader(synthetic_labels)

    real_object_counts = real_loader.get_object_counts()
    synthetic_object_counts = synthetic_loader.get_object_counts()

    real_object_counts, synthetic_object_counts = equalize_distribution_length(
        real_object_counts, synthetic_object_counts
    )

    real_object_counts = list(real_object_counts.values())
    synthetic_object_counts = list(synthetic_object_counts.values())

    results = []

    compare_distributions(
        real_object_counts,
        synthetic_object_counts,
        "Label Overlap",
        results,
        "dataset",
    )

    return results


if __name__ == "__main__":
    main()
