import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.stats import gaussian_kde
from scipy.stats import anderson_ksamp, PermutationMethod
from scipy.stats import entropy
from scipy.stats import energy_distance
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import pandas as pd
import csv
import cv2
from PIL import Image
import re
import sys
from image_mauve import KLDivergence as mauveKL


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

# Store bounding boxes
def load_bboxes(label_dir):
  if isinstance(label_dir, str):
    files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.txt')]
  elif isinstance(label_dir, list):
    files = label_dir
  bboxes = []
  for file in files:
    if file.endswith('.txt'):
      try:
        with open(file, 'r') as f:
          for line in f:
            line = convert_instance_seg_to_bbox(line)
            _, x, y, w, h = map(float, line.split())
            bboxes.append((w, h))
      except FileNotFoundError:
        print(f"File {file} not found")
  bboxes = np.array(bboxes)
  if bboxes.ndim == 1:
    bboxes = bboxes.reshape(-1, 2)  # Reshape
  return bboxes

# Calculate the aspect ratio, size, and area
def calculate_stats(bboxes):
  widths, heights = bboxes[:, 0], bboxes[:, 1]
  aspect_ratios = widths / heights
  sizes = np.sqrt(widths**2 + heights**2)
  areas = widths * heights
  return aspect_ratios, sizes, areas

def calculate_pixels(dir):
  if isinstance(dir, str):
    all_images = [os.path.join(dir, f) for f in os.listdir(dir)]
  elif isinstance(dir, list):
    all_images = dir
  channels = ['Red', 'Green', 'Blue']
  valid_images = [img for img in all_images if os.path.splitext(img)[1].lower() in ['.jpg', '.png', '.tiff', '.tif', '.bmp']]
  frequencies = []

  def calculate_pixel_histogram(image_path, channel):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    channel_values = img[:, :, channel].flatten()
    hist, _ = np.histogram(channel_values, bins=256, range=[0, 256])
    return hist / np.sum(hist)

  channel_colors = {'Red': (1, 0, 0), 'Green': (0, 1, 0), 'Blue': (0, 0, 1)}

  for channel in range(3):
    channel_frequencies = []
    aggregated_hist = np.zeros(256)
    
    for image in valid_images:
      image_path = image
      image_hist = calculate_pixel_histogram(image_path, channel)
      aggregated_hist += image_hist

    aggregated_hist /= len(valid_images)

    for i in range(256):
      channel_frequencies.append(aggregated_hist[i])
    
    frequencies.append(channel_frequencies)
  return frequencies[0], frequencies[1], frequencies[2]

# Plot and compare
def compare_stats(real_data, synthetic_data, dataset, label, output):
  # Make relative frequency
  real_data = [x / sum(real_data) for x in real_data]
  synthetic_data = [x / sum(synthetic_data) for x in synthetic_data]

  real_pdf = gaussian_kde(real_data)
  synthetic_pdf = gaussian_kde(synthetic_data)
  x = np.linspace(min(min(real_data), min(synthetic_data)),
                  max(max(real_data), max(synthetic_data)), 1000)

  # Find Kolmogorov-Smirnov (K-S) Test, Kullback-Leibler (KL) divergence, Wasserstein Distance
  ks_statistic, ks_pvalue = ks_2samp(real_data, synthetic_data)

  # Anderson-Darling (A-D) test
  ad_statistic, ad_critical_values, ad_significance_level = anderson_ksamp([real_data, synthetic_data]) #, method=PermutationMethod(n_resamples=5))

  # Kullback-Leibler (KL) Divergence Test
  epsilon = 1e-12
  kl_divergence = entropy(real_pdf(x) + epsilon, synthetic_pdf(x) + epsilon)

  # Energy Distance (E-D) test
  ed_distance = energy_distance(real_data, synthetic_data)

  # Wasserstein Distance/Earth Mover's Distance (WD/EMD)
  w_distance = wasserstein_distance(real_data, synthetic_data)

  # Bhattacharyya Distance (BD)
  bc_distance = -np.log(np.sum(np.sqrt(real_pdf(x) * synthetic_pdf(x))) * (x[1] - x[0]))

  # Jensen-Shannon (JS) Divergence
  jsd = jensenshannon(real_pdf(x), synthetic_pdf(x))

  # Append results to list
  output.append({
      # 'Dataset': dataset,
      # 'Label': label,
      'K-S Statistic': ks_statistic,
      'K-S p-value': ks_pvalue,
      'A-D Statistic': ad_statistic,
      'KL Divergence': kl_divergence,
      'Energy Distance': ed_distance,
      'Wasserstein Distance': w_distance,
      'Bhattacharyya Distance': bc_distance,
      'Jensen-Shannon Divergence': jsd,
      #'Mauve KL': mauveKL().calculate(np.array(real_data, dtype=np.float32), np.array(synthetic_data, dtype=np.float32)).mauve,
  })

def analyze_bounding_boxes(dataset_name, real_labels, synthetic_labels):
  real_bboxes = load_bboxes(real_labels)
  synthetic_bboxes = load_bboxes(synthetic_labels)

  real_aspect_ratios, real_sizes, real_areas = calculate_stats(real_bboxes)
  synthetic_aspect_ratios, synthetic_sizes, synthetic_areas = calculate_stats(synthetic_bboxes)

  results = []
  compare_stats(real_aspect_ratios, synthetic_aspect_ratios, dataset_name, "Aspect Ratios", results)
  compare_stats(real_sizes, synthetic_sizes, dataset_name, "Sizes", results)
  compare_stats(real_areas, synthetic_areas, dataset_name, "Areas", results)
  return results

def analyze_pixels(dataset_name, real_images, synthetic_images):
  red_pixel_dist_real, green_pixel_dist_real, blue_pixel_dist_real = calculate_pixels(real_images)
  red_pixel_dist_synthetic, green_pixel_dist_synthetic, blue_pixel_dist_synthetic = calculate_pixels(synthetic_images)

  results = []
  compare_stats(red_pixel_dist_real, red_pixel_dist_synthetic, dataset_name, "Red Pixels", results)
  compare_stats(green_pixel_dist_real, green_pixel_dist_synthetic, dataset_name, "Green Pixels", results)
  compare_stats(blue_pixel_dist_real, blue_pixel_dist_synthetic, dataset_name, "Blue Pixels", results)
  return results

def calculate_pixel_intensity(real_label_paths, synthetic_label_paths):
  real_image_paths = [label.replace("labels", "images").replace(".txt", ".png") if os.path.exists(label.replace("labels", "images").replace(".txt", ".png")) else label.replace("labels", "images").replace(".txt", ".jpg") for label in real_label_paths]
  synthetic_image_paths = [label.replace("labels", "images").replace(".txt", ".png") if os.path.exists(label.replace("labels", "images").replace(".txt", ".png")) else label.replace("labels", "images").replace(".txt", ".jpg") for label in synthetic_label_paths]
  results = analyze_pixels("Data", real_image_paths, synthetic_image_paths)
  all_results = {}
  for i,result in enumerate(results):
    for key, value in result.items():
      to_append = "red" if i == 0 else "green" if i == 1 else "blue" if i == 2 else None
      all_results[f"{to_append}_{key}"] = value
  return all_results

def calculate_bounding_box_distribution(real_label_paths, synthetic_label_paths):
  # truncate
  results = analyze_bounding_boxes("Data", real_label_paths, synthetic_label_paths)
  all_results = {}
  for i,result in enumerate(results):
    for key, value in result.items():
      to_append = "aspect_ratios" if i == 0 else "sizes" if i == 1 else "areas" if i == 2 else None
      all_results[f"{to_append}_{key}"] = value
  return all_results




if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python bounding_box_analysis.py <base_directory>")
    sys.exit(1)

  base_dir = sys.argv[1]
  real_labels = os.path.join(base_dir, 'real/labels/train')
  synthetic_labels = os.path.join(base_dir, 'synthetic/labels/train')
  real_images = os.path.join(base_dir, 'real/images/train')
  synthetic_images = os.path.join(base_dir, 'synthetic/images/train')

  if (base_dir[-1] == '/'):
    dataset_name = base_dir.split('/')[-2]
  else:
    dataset_name = base_dir.split('/')[-1]
  bounding_box_results = analyze_bounding_boxes(dataset_name, real_labels, synthetic_labels)
  pixel_intensity_results = analyze_pixels(dataset_name, real_images, synthetic_images)

  df1 = pd.DataFrame(bounding_box_results)
  df2 = pd.DataFrame(pixel_intensity_results)
  df = pd.concat([df1, df2], axis=0)
  output_file = 'results.csv'

  if os.path.exists(output_file):
    df_existing = pd.read_csv(output_file)
    df_combined = pd.concat([df_existing, df], ignore_index=True)
    df_combined.to_csv(output_file, index=False)
    print(f"Appended results to {output_file}")
  else:
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")