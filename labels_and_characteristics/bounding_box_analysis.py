import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, ks_2samp, entropy, energy_distance, wasserstein_distance, anderson_ksamp
from scipy.spatial.distance import jensenshannon

def load_bboxes(label_dir):
    bboxes = []
    for file in os.listdir(label_dir):
        if file.endswith('.txt'):
            with open(os.path.join(label_dir, file), 'r') as f:
                for line in f:
                    _, x, y, w, h = map(float, line.split())
                    bboxes.append((w, h))
    bboxes = np.array(bboxes)
    if bboxes.ndim == 1:
        bboxes = bboxes.reshape(-1, 2)  # Reshape
    return bboxes

def calculate_stats(bboxes):
    widths, heights = bboxes[:, 0], bboxes[:, 1]
    aspect_ratios = widths / heights
    # Get diagonal lengths
    sizes = np.sqrt(widths**2 + heights**2)
    areas = widths * heights
    return aspect_ratios, sizes, areas

def compare_stats(real_data, synthetic_data, label, results, dataset_name):
    # Make relative frequency
    real_data = [x / sum(real_data) for x in real_data]
    synthetic_data = [x / sum(synthetic_data) for x in synthetic_data]
    print(f"real_data: {real_data}")
    print(f"synthetic_data: {synthetic_data}")
    real_pdf = gaussian_kde(real_data)
    synthetic_pdf = gaussian_kde(synthetic_data)
    x = np.linspace(min(min(real_data), min(synthetic_data)),
                  max(max(real_data), max(synthetic_data)), 1000)
    
    ks_statistic, ks_pvalue = ks_2samp(real_data, synthetic_data)
    ad_statistic, _, _ = anderson_ksamp([real_data, synthetic_data])
    epsilon = 1e-12
    kl_divergence = entropy(real_pdf(x) + epsilon, synthetic_pdf(x) + epsilon)
    ed_distance = energy_distance(real_data, synthetic_data)
    w_distance = wasserstein_distance(real_data, synthetic_data)
    bc_distance = -np.log(np.sum(np.sqrt(real_pdf(x) * synthetic_pdf(x))) * (x[1] - x[0]))
    jsd = jensenshannon(real_pdf(x), synthetic_pdf(x))

    results.append({
        'K-S Statistic': ks_statistic,
        'K-S p-value': ks_pvalue,
        'A-D Statistic': ad_statistic,
        'KL Divergence': kl_divergence,
        'Energy Distance': ed_distance,
        'Wasserstein Distance': w_distance,
        'Bhattacharyya Distance': bc_distance,
        'Jensen-Shannon Divergence': jsd,
    })

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python bounding_box_analysis.py <base_directory>")
        sys.exit(1)

    base_dir = sys.argv[1]
    real_dir = os.path.join(base_dir, 'real')
    synthetic_dir = os.path.join(base_dir, 'synthetic')

    real_bboxes = load_bboxes(os.path.join(real_dir, 'labels/train'))
    synthetic_bboxes = load_bboxes(os.path.join(synthetic_dir, 'labels/train'))

    real_aspect_ratios, real_sizes, real_areas = calculate_stats(real_bboxes)
    synthetic_aspect_ratios, synthetic_sizes, synthetic_areas = calculate_stats(synthetic_bboxes)

    results = []
    dataset_name = base_dir.split('/')[-2]

    compare_stats(real_aspect_ratios, synthetic_aspect_ratios, "Aspect Ratios", results, dataset_name)
    compare_stats(real_sizes, synthetic_sizes, "Sizes", results, dataset_name)
    compare_stats(real_areas, synthetic_areas, "Areas", results, dataset_name)

    df = pd.DataFrame(results)
    output_file = 'bounding_box_analysis_results.csv'

    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_csv(output_file, index=False)
        print(f"Appended results to {output_file}")
    else:
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")