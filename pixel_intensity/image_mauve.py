import argparse
from types import SimpleNamespace
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from PIL import Image
from transformers import (
    ViTImageProcessor,
    ViTModel,
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
)
from sklearn.decomposition import PCA
from sklearn.metrics import auc as compute_area_under_curve
from sklearn.preprocessing import normalize
from tqdm import tqdm

from mauve.compute_mauve import (
    get_fronter_integral,
    get_divergence_curve_for_multinomials,
    cluster_feats,
    get_features_from_input,
)


class Metric(ABC):
    @abstractmethod
    def calculate(self):
        pass


@dataclass
class KLDivergence(Metric):
    divergence_curve_discretization_size: int = 25
    mauve_scaling_factor: int = 5

    def calculate(self, p, q):
        mixture_weights = np.linspace(
            1e-6, 1 - 1e-6, self.divergence_curve_discretization_size
        )
        divergence_curve = get_divergence_curve_for_multinomials(
            p, q, mixture_weights, self.mauve_scaling_factor
        )
        x, y = divergence_curve.T
        idxs1 = np.argsort(x)
        idxs2 = np.argsort(y)
        mauve_score = 0.5 * (
            compute_area_under_curve(x[idxs1], y[idxs1])
            + compute_area_under_curve(y[idxs2], x[idxs2])
        )
        fi_score = get_fronter_integral(p, q)
        to_return = SimpleNamespace(
            p_hist=p,
            q_hist=q,
            divergence_curve=divergence_curve,
            mauve=mauve_score,
            fronter_integral=fi_score,
        )
        return to_return
