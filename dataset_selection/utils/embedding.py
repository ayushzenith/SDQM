import argparse
from cgitb import text
from importlib import metadata
from pyexpat import features
from tracemalloc import start
from types import SimpleNamespace
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import clip
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from transformers import ViTImageProcessor, ViTModel, AutoImageProcessor, AutoModel, AutoProcessor
from tqdm import tqdm

class DataLoader(ABC):
    @abstractmethod
    def get_next(self):
        pass

    @abstractmethod
    def get_all(self):
        pass

class Embedding(ABC):
    @abstractmethod
    def embed(self, data):
        pass
    
    @abstractmethod
    def get_data(self):
        pass


@dataclass
class PathLoader(DataLoader):
  path: Path
  file_list: list = field(default_factory=list)
  recurse: bool = False
  index: int = 0
  is_finite: bool = True

  def __post_init__(self):
    file_extensions = {".jpg", ".jpeg", ".png", ".gif", ".tiff"}
    if self.recurse:
      self.file_list = [f for f in tqdm(self.path.rglob("*")) if f.suffix.lower() in file_extensions]
    else:
      self.file_list = [f for f in self.path.glob("*") if f.suffix.lower() in file_extensions]

  def get_count_of_images(self):
    return len(self.file_list)

  def get_next(self):
    if self.index == len(self.file_list):
      self.index = 0
      return None
    image_path = self.file_list[self.index]
    self.index = self.index + 1
    image = Image.open(image_path)
    # Ensure the image has 3 dimensions by converting to RGB
    filename = image.filename
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.filename = filename  # Reattach the filename
    return image

    def get_file_list(self):
        return self.file_list

  def get_all(self):
    data = list()
    for i in self.file_list:
      data.append(Image.Open(i))
    self.index = 0
    return data



@dataclass
class ViTEmbedding(Embedding):
  model_name: str = "SeyedAli/Remote-Sensing-UAV-image-classification"
  device: str = "cuda" if torch.cuda.is_available() else "cpu"
  verbose: bool = False
  data: list = field(default_factory=list)
  updated: bool = False
  array: np.array = None

  def __post_init__(self):
    self.processor = ViTImageProcessor.from_pretrained(self.model_name)
    self.model = ViTModel.from_pretrained(self.model_name).to(self.device)

  def embed(self, image_data: Image):
    inputs = self.processor(images=image_data, return_tensors="pt").to(self.device)
    with torch.no_grad():
      outputs = self.model(**inputs)
      vector = outputs.pooler_output.cpu()
    self.data.append(vector)
    return vector

  def get_data(self):
    if not self.updated and self.array is not None:
      return self.array
    self.array = torch.cat(self.data, dim=0).cpu().numpy()
    return self.array

  def get_last_data(self):
    return self.data[-1].cpu().numpy()

  def reset_data(self):
    self.data = list()
    self.update = True
    self.array = None
    


@dataclass
class nomicEmbedding(Embedding):
  model_name: str = "nomic-ai/nomic-embed-vision-v1.5"
  device: str = "cuda" if torch.cuda.is_available() else "cpu"
  verbose: bool = False
  data: list = field(default_factory=list)
  updated: bool = False
  array: np.array = None

  def __post_init__(self):
    self.processor = AutoImageProcessor.from_pretrained(self.model_name)
    self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

  def embed(self, image_data: Image):
    inputs = self.processor(images=image_data, return_tensors="pt").to(self.device)
    with torch.no_grad():
      img_emb = self.model(**inputs).last_hidden_state
      img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
      vector = img_embeddings.cpu()
    self.data.append(vector)
    return vector

  def get_data(self):
    if not self.updated and self.array is not None:
      return self.array
    self.array = torch.cat(self.data, dim=0).cpu().numpy()
    return self.array
  
  def get_last_data(self):
    return self.data[-1].cpu().numpy()

  def reset_data(self):
    self.data = list()
    self.update = True
    self.array = None


@dataclass
class hfEmbedding(Embedding):
  model_name: str = "facebook/dinov2-small"
  device: str = "cuda" if torch.cuda.is_available() else "cpu"
  verbose: bool = False
  data: list = field(default_factory=list)
  updated: bool = False
  array: np.array = None

  def __post_init__(self):
    self.processor = AutoImageProcessor.from_pretrained(self.model_name)
    self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

  def embed(self, image_data: Image):
    inputs = self.processor(images=image_data, return_tensors="pt").to(self.device)
    with torch.no_grad():
      outputs = self.model(**inputs)
      vector = outputs.last_hidden_state
      vector = vector.mean(dim=1).cpu()
    self.data.append(vector)
    return vector

  def get_data(self):
    if not self.updated and self.array is not None:
      return self.array
    self.array = torch.cat(self.data, dim=0).cpu().numpy()
    return self.array

  def get_last_data(self):
    return self.data[-1].cpu().numpy()

  def reset_data(self):
    self.data = list()
    self.update = True
    self.array = None



@dataclass
class groundingDINOEmbedding(Embedding):
  model_name: str = "IDEA-Research/grounding-dino-tiny"
  device: str = "cuda" if torch.cuda.is_available() else "cpu"
  text: str = "vehicle ."
  verbose: bool = False
  data: list = field(default_factory=list)
  updated: bool = False
  array: np.array = None

  def __post_init__(self):
    self.processor = AutoProcessor.from_pretrained(self.model_name)
    self.model = AutoModel.from_pretrained(self.model_name).to(self.device)

  def embed(self, image_data: Image):
    inputs = self.processor(images=image_data, text=self.text, return_tensors="pt").to(self.device)
    with torch.no_grad():
      outputs = self.model(**inputs, output_hidden_states=True)
      vector = outputs.encoder_last_hidden_state_vision
      vector = vector.mean(dim=1).cpu()
    self.data.append(vector)
    return vector

  def get_data(self):
    if not self.updated and self.array is not None:
      return self.array
    self.array = torch.cat(self.data, dim=0).cpu().numpy()
    return self.array

  def get_last_data(self):
    return self.data[-1].cpu().numpy()

  def reset_data(self):
    self.data = list()
    self.update = True
    self.array = None



@dataclass
class ClipEmbedding(Embedding):
  model_name: str = "ViT-B/32"
  device: str = "cuda" if torch.cuda.is_available() else "cpu"
  verbose: bool = False
  data: list = field(default_factory=list)
  updated: bool = False
  array: np.array= None

  def __post_init__(self):
    self.model, self.preprocess = clip.load(self.model_name, device=self.device)

  def embed(self, image_data: Image):
    image_data = self.preprocess(image_data).unsqueeze(0).to(self.device)
    with torch.no_grad():
      image_features = self.model.encode_image(image_data)

    self.data.append(image_features.cpu())
    self.updated = True
    return image_features

  def get_data(self):
    if not self.updated and self.array is not None:
      return self.array
    self.array = torch.cat(self.data, dim=0).cpu().numpy()
    return self.array

  def get_last_data(self):
    return self.data[-1].cpu().numpy()

  def reset_data(self):
    self.data = list()
    self.update = True
    self.array = None


@dataclass
class Embedder:
  p_path: Path
  device_id: int = 0
  save_path: Path = Path.cwd()
  embedding_model: str = "ViT-B/32"
  text: str = "vehicle ."
  dataset: str = "Rareplanes"
  metadata1: str = "Real"
  metadata2: str = "Train"
  verbose: bool = False

  def run(self):
    print(f"Running Embedder with {self.embedding_model} model")
 
 
    openai_clip=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'],
    hf_vit_clip=['SeyedAli/Remote-Sensing-UAV-image-classification']
    hf_other_clip=['facebook/dinov2-small']
    grounding_dino_clip=['IDEA-Research/grounding-dino-tiny']
    nomic_embedding_model = ["nomic-ai/nomic-embed-vision-v1.5"]


    if self.embedding_model in hf_vit_clip:
      self.embedding = ViTEmbedding(verbose=self.verbose, model_name=self.embedding_model, device=self.device_id)
    elif self.embedding_model in nomic_embedding_model:
        self.embedding = nomicEmbedding(verbose=self.verbose, model_name=self.embedding_model, device=self.device_id)
    elif self.embedding_model in grounding_dino_clip:
      if self.text is None:
        raise ValueError("Text must be provided for grounding dino model")
      self.embedding = groundingDINOEmbedding(verbose=self.verbose, model_name=self.embedding_model, text=self.text, device=self.device_id)
    elif self.embedding_model in hf_other_clip:
      self.embedding = hfEmbedding(verbose=self.verbose, model_name=self.embedding_model, device=self.device_id)
    else:
      self.embedding = ClipEmbedding(verbose=self.verbose, model_name=self.embedding_model, device=self.device_id)


    start_time = datetime.now()
    p_path = Path(self.p_path)
    
    if not p_path.exists():
        raise FileNotFoundError(f"Path {p_path} does not exist")
    
    if not p_path.is_dir():
        raise NotADirectoryError(f"Path {p_path} is not a directory")
    
    print(f"Loading data from {p_path}")
    loader = PathLoader(p_path, recurse=True)
    
    print(f"Found {loader.get_count_of_images()} images")
        
    print("Embedding images")
    
    image_details = []
    
    if loader.is_finite:
        for i in tqdm(range(loader.get_count_of_images())):
            image = loader.get_next()
            if image is None:
                break
            self.embedding.embed(image)
            image_details.append({
                'file_path': image.filename,
                'file_name': image.filename.split('/')[-1],
                'embedding': self.embedding.get_last_data(),
                'dataset': self.dataset,
                'metadata1': self.metadata1,
                'metadata2': self.metadata2
            })
    else:
        while True:
            image = loader.get_next()
            if image is None:
                break
            self.embedding.embed(image)
            image_details.append({
                'file_path': image.filename,
                'file_name': image.filename.split('/')[-1],
                'embedding': self.embedding.get_last_data(),
                'dataset': self.dataset,
                'metadata1': self.metadata1,
                'metadata2': self.metadata2
            })
            
    features = self.embedding.get_data()
    print(f"Embedding took {datetime.now() - start_time}")
    filesaveloc = self.save_path + '/' + self.dataset + self.metadata1 + self.metadata2 + '_' + str(datetime.now()).replace(' ', '_') + '_' + str(datetime.now() - start_time)
    
    
    df_embeddings = pd.DataFrame(image_details)
    df_embeddings.to_csv(filesaveloc + '.csv', index=False)
    
    print(f"Saving features to {self.save_path}")
    np.save(filesaveloc + '.npy', features)
    
    # Save features to .pkl
    with open(filesaveloc + '.pkl', 'wb') as f:
      pickle.dump(features, f)
    
    print(f"Saved features to {self.save_path}")
    
    return features
