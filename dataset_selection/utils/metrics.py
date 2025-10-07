import os
import sys

import numpy as np
from mauve import compute_mauve

from utils.alpha_precision_beta_recall_authenticity import compute_alpha_precision

# Get the directory two levels up
parent_of_parent_dir = os.path.abspath(os.path.join(__file__, "../.."))

# Add it to sys.path
sys.path.append(parent_of_parent_dir)

from separability.log_cluster_metric import compute_log_cluster_metric


def calculate_cluster_metric(p_embeddings, q_embeddings):
    return (
        10
        ** compute_log_cluster_metric(
            np.array(p_embeddings), np.array(q_embeddings), 10, "kmeans"
        )[0]
    )


def calculate_mauve(p_embeddings, q_embeddings):
    return compute_mauve(p_embeddings, q_embeddings).mauve


def calculate_mauve_star(p_embeddings, q_embeddings):
    return compute_mauve(p_embeddings, q_embeddings).mauve_star


def calculate_frontier_integral(p_embeddings, q_embeddings):
    return compute_mauve(p_embeddings, q_embeddings).frontier_integral


def calculate_frontier_integral_star(p_embeddings, q_embeddings):
    return compute_mauve(p_embeddings, q_embeddings).frontier_integral_star


def compute_alpha_precision_beta_recall_authenticity(p_embeddings, q_embeddings):
    # ensure p_embeddings and q_embeddings are numpy arrays
    p_embeddings = np.array(p_embeddings)
    q_embeddings = np.array(q_embeddings)

    # use real center
    embeddings_center = np.mean(p_embeddings, axis=0)

    # Calculate the alpha precision and beta recall
    (
        x,
        alpha_precision_curve,
        beta_recall_curve,
        Delta_alpha_precision,
        Delta_beta_recall,
        authenticity,
    ) = compute_alpha_precision(p_embeddings, q_embeddings, embeddings_center)

    # To all in alpha_precision_curve, subtract the corresponding x
    assert len(x) == len(alpha_precision_curve) == len(beta_recall_curve)

    return Delta_alpha_precision, Delta_beta_recall, authenticity


def calculate_alpha_precision(p_embeddings, q_embeddings):
    return compute_alpha_precision_beta_recall_authenticity(p_embeddings, q_embeddings)[
        0
    ]


def calculate_beta_recall(p_embeddings, q_embeddings):
    return compute_alpha_precision_beta_recall_authenticity(p_embeddings, q_embeddings)[
        1
    ]


def calculate_authenticity(p_embeddings, q_embeddings):
    return compute_alpha_precision_beta_recall_authenticity(p_embeddings, q_embeddings)[
        2
    ]
