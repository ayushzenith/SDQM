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
from transformers import ViTImageProcessor, ViTModel, AutoImageProcessor, AutoModel, AutoProcessor
from sklearn.decomposition import PCA
from sklearn.metrics import auc as compute_area_under_curve
from sklearn.preprocessing import normalize
from tqdm import tqdm

from mauve.compute_mauve import get_fronter_integral, get_divergence_curve_for_multinomials, cluster_feats, get_features_from_input

class Quantization(ABC):
    @abstractmethod
    def quantize(self):
        pass

class Metric(ABC):
    @abstractmethod
    def calculate(self):
        pass


@dataclass
class ClusteringQuantization(Quantization):
  num_clusters: int
  norm: str = "l2"
  whiten: bool = False
  pca_max_data: int = -1
  explained_variance: float = 0.9
  num_redo: int = 5
  max_iter: int = 500
  seed: int = 25
  verbose: bool = False

  def quantize(self, p, q):
      if self.num_clusters == "auto":
        self.num_clusters= max(2, int(round(min(p.shape[0], q.shape[0]) / 10)))
      elif not isinstance(self.num_clusters, int):
        raise ValueError("num_buckets should be int or 'auto'")

      return_p, return_q, return_p_smooth, return_q_smooth = cluster_feats(
        p, q, 
        num_clusters=self.num_clusters, 
        norm=self.norm,
        whiten=self.whiten,
        pca_max_data=self.pca_max_data, 
        explained_variance=self.explained_variance, 
        num_redo=self.num_redo, 
        max_iter=self.max_iter, 
        seed=self.seed, 
        verbose=self.verbose)
      return return_p, return_q, return_p_smooth, return_q_smooth

@dataclass
class KLDivergence(Metric):
  divergence_curve_discretization_size: int = 25
  mauve_scaling_factor: int = 5

  def calculate(self, p, q):
      mixture_weights = np.linspace(1e-6, 1-1e-6, self.divergence_curve_discretization_size)
      divergence_curve = get_divergence_curve_for_multinomials(p, q, mixture_weights, self.mauve_scaling_factor)
      x, y = divergence_curve.T
      idxs1 = np.argsort(x)
      idxs2 = np.argsort(y)
      mauve_score = 0.5 * (
        compute_area_under_curve( x[idxs1], y[idxs1]) + 
        compute_area_under_curve( y[idxs2], x[idxs2])
      )
      fi_score = get_fronter_integral(p, q)
      to_return = SimpleNamespace(
        p_hist=p, q_hist=q, divergence_curve=divergence_curve,mauve=mauve_score,fronter_integral=fi_score)
      return to_return



def get_dataset_similarity(p_features, q_features):
  
    p_features = np.array(p_features)
    q_features = np.array(q_features)
    
    #print(f"q_features: {type(q_features)}, {q_features.shape}")
    #print(f"p_features: {type(p_features)}, {p_features.shape}")
    
    num_buckets = "auto"
    pca_max_data = -1
    kmeans_explained_var = 0.9
    kmeans_num_redo = 5
    kmeans_max_iter = 500
    divergence_curve_discretization_size = 25
    mauve_scaling_factor = 5
    verbose = False
    seed = 25
    batch_size = 1
    use_float64 = False
    
    #print("Creating quantization...")
    quantization = ClusteringQuantization(
      num_clusters=num_buckets,
      pca_max_data = pca_max_data,
      explained_variance = kmeans_explained_var,
      num_redo = kmeans_num_redo,
      max_iter = kmeans_max_iter,
      seed = seed,
      verbose = verbose)

    #print("Creating metric...")
    metric = KLDivergence(
      divergence_curve_discretization_size=divergence_curve_discretization_size,
      mauve_scaling_factor=mauve_scaling_factor)

    #print("Quantizing...")
    p, q, p_smooth, q_smooth  = quantization.quantize(p_features, q_features)

    #print("Computing metric...")
    results = metric.calculate(p, q)
    results.num_buckets = num_buckets

    smooth_results = metric.calculate(p_smooth, q_smooth)
    smooth_results.num_buckets = num_buckets

    return results, smooth_results


@dataclass
class ImageMauve:
  p_path: Path
  q_path: Path
  num_buckets: int | str = "auto"
  pca_max_data: int = -1
  kmeans_explained_var: float = 0.9
  kmeans_num_redo: int = 5
  kmeans_max_iter: int = 500
  divergence_curve_discretization_size: int = 25
  mauve_scaling_factor: int = 5
  verbose: bool = False
  seed: int = 25
  batch_size: int = 1
  use_float64: bool = False

  def run(self):
    
    print("Importing embedding...")
    
    file_extension = self.p_path.suffix.lower()
    if file_extension == ".npy":
      p_features = np.load(self.p_path)
    elif file_extension == ".pkl":
      with open(self.p_path, 'rb') as f:
        p_features = pickle.load(f)
    else:
      raise ValueError("File extension not supported")
    
    print("Imported Embedding set P")
    
    file_extension = self.q_path.suffix.lower()
    if file_extension == ".npy":
      q_features = np.load(self.q_path)
    elif file_extension == ".pkl":
      with open(self.q_path, 'rb') as f:
        q_features = pickle.load(f)
    else:
      raise ValueError("File extension not supported")
    
    print("Imported Embedding set Q")
    
    print(f"q_features: {type(q_features)}, {q_features.shape}")
    print(f"p_features: {type(p_features)}, {p_features.shape}")
    
    print("Creating quantization...")
    self.quantization = ClusteringQuantization(
      num_clusters=self.num_buckets,
      pca_max_data = self.pca_max_data,
      explained_variance = self.kmeans_explained_var,
      num_redo = self.kmeans_num_redo,
      max_iter = self.kmeans_max_iter,
      seed = self.seed,
      verbose = self.verbose)

    print("Creating metric...")
    self.metric = KLDivergence(
      divergence_curve_discretization_size=self.divergence_curve_discretization_size,
      mauve_scaling_factor=self.mauve_scaling_factor)

    print("Quantizing...")
    p, q, p_smooth, q_smooth  = self.quantization.quantize(p_features, q_features)

    print("Computing metric...")
    self.results = self.metric.calculate(p, q)
    self.results.num_buckets = self.num_buckets

    self.smooth_results = self.metric.calculate(p_smooth, q_smooth)
    self.smooth_results.num_buckets = self.num_buckets

    return self.results, self.smooth_results

def get_unix_timestamp() -> str:
  return str(round((datetime.now() - datetime(1970, 1, 1)).total_seconds()))

if __name__ == '__main__':
  command_line_entry2()
