import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from skimage.measure import block_reduce
from PIL import Image


def convert_instance_seg_to_bbox(line):
    """
    Detects if the line is already in YOLO bounding-box format (5 tokens):
        class_id x_center y_center width height
    If so, returns the line unchanged.
    
    Otherwise, assumes the line is in instance-segmentation format:
        class_id x1 y1 x2 y2 ... xn yn
    Converts this polygon to a YOLO bounding box (normalized),
    and returns:
        class_id x_center y_center width height
    """
    parts = line.strip().split()
    
    # If exactly 5 tokens, it's bounding-box format, return unchanged.
    if len(parts) == 5:
        return line.strip()
    
    # Otherwise, assume instance-segmentation format
    class_id = parts[0]
    coords = parts[1:]  # x1, y1, x2, y2, ...
    coords = list(map(float, coords))
    
    # Separate into x_coords and y_coords
    x_coords = coords[0::2]  # even indices
    y_coords = coords[1::2]  # odd indices
    
    # Compute bounding box (in pixel space)
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    
    x_center = x_min + (bbox_width / 2.0)
    y_center = y_min + (bbox_height / 2.0)
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"


class YOLOBoundingBoxLoader:
    def __init__(self, annotations_path: str | list[str]):
        if isinstance(annotations_path, str):
            self.annotations = [
                os.path.join(annotations_path, annotation_file)
                for annotation_file in tqdm(
                    os.listdir(annotations_path),
                    desc=f"Loading YOLO annotations from {annotations_path}",
                )
                if annotation_file.endswith(".txt")
            ]
        elif isinstance(annotations_path, list):
            self.annotations = annotations_path

        self.index = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.annotations)

    def _load_annotation(self, file_path: str) -> list:
        try:
            with open(file_path, "r") as f:
                annotations = [
                    convert_instance_seg_to_bbox(line) for line in f.readlines()
                ]
        except FileNotFoundError:
            return None
        return [list(map(float, line.strip().split())) for line in annotations]

    def __next__(self):
        if self.index < len(self):
            annotation_file = self.annotations[self.index]
            annotation_attributes = self._load_annotation(annotation_file)
            if annotation_attributes is None:
                self.index += 1
                return []

            bboxes = [(bbox[1], bbox[2], bbox[3], bbox[4]) for bbox in annotation_attributes]

            self.index += 1
            return bboxes
        else:
            raise StopIteration


class SpatialDistribution:
    def __init__(
        self,
        annotations_path: str,
        image_shape: tuple,
        annotations_loader=YOLOBoundingBoxLoader,
    ):
        self.annotations_path = annotations_path
        self.annotations_loader = annotations_loader
        self.image_shape = image_shape
        self.heatmap = None

    def mean_pool(self, pool_size: int) -> None:
        pool_size = (pool_size, pool_size)
        self.heatmap = block_reduce(self.heatmap, pool_size, np.mean)

    def create_heatmap(self) -> np.ndarray:
        # Initialize a blank heatmap
        heatmap = np.zeros(self.image_shape, dtype=np.float32)

        img_height, img_width = self.image_shape

        # Update heatmap with bounding boxes
        for annotation_bboxes in tqdm(self.annotations_loader(self.annotations_path), desc="Creating Heatmap"):
            for bbox in annotation_bboxes:
                cx, cy, w, h = bbox
                x_min = int((cx - w / 2) * img_width)
                x_max = int((cx + w / 2) * img_width)
                y_min = int((cy - h / 2) * img_height)
                y_max = int((cy + h / 2) * img_height)

                heatmap[y_min:y_max, x_min:x_max] += 1

        # Normalize heatmap to range [0, 1]
        heatmap /= np.max(heatmap)

        self.heatmap = heatmap

        return heatmap

    def save_heatmap(self, output_path: str) -> None:
        plt.imshow(self.heatmap, cmap="hot", interpolation="nearest")
        plt.colorbar()
        plt.title("Bounding Box Heatmap")
        plt.savefig(output_path)
