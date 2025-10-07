"""
Log cluster metric:
$$
LC(R, S) = \log\left(\frac{1}{k}\sum_{i=1}^{k}{\left[ \frac{n_i^R}{n_i} - \frac{1}{2} \right]^2}\right)
$$
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans


def kmeans_cluster_assignments(p: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    d = np.vstack([p, q])
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(d)

    return labels[: len(p)], labels[len(p) :]


def hierarchical_cluster_assignments(
    p: np.ndarray, q: np.ndarray, k: int
) -> np.ndarray:
    d = np.vstack([p, q])
    ac = AgglomerativeClustering(n_clusters=k)
    labels = ac.fit_predict(d)

    return labels[: len(p)], labels[len(p) :]


def log_cluster_metric(
    p_assignments: np.ndarray,
    q_assignments: np.ndarray,
    k: int,
) -> float:
    sum = 0
    for i in range(k):
        n_i_p = np.sum(p_assignments == i)
        n_i = n_i_p + np.sum(q_assignments == i)

        lc = (n_i_p / n_i - 0.5) ** 2

        sum += lc

    return np.log(sum / k)


def normalized_log_cluster_metric(
    p_assignments: np.ndarray,
    q_assignments: np.ndarray,
    k: int,
) -> float:
    n_R = len(p_assignments)
    n_S = len(q_assignments)
    sum = 0
    for i in range(k):
        n_i_R = np.sum(p_assignments == i)
        n_i_S = np.sum(q_assignments == i)

        lc = ((n_i_R / n_R) / ((n_i_R / n_R) + (n_i_S / n_S)) - 0.5) ** 2

        sum += lc

    return np.log(sum / k)


def generalized_cluster_metric(
    p_assignments: np.ndarray,
    q_assignments: np.ndarray,
    k: int,
) -> float:
    n_R = len(p_assignments)
    n_S = len(q_assignments)
    sum = 0
    for i in range(k):
        n_i_R = np.sum(p_assignments == i)
        n_i_S = np.sum(q_assignments == i)
        n_i = n_i_R + n_i_S

        lc = ((n_i_R / n_i) - (n_R / (n_R + n_S))) ** 2

        sum += lc

    return sum / k


def compute_log_cluster_metric(
    p: np.ndarray, q: np.ndarray
) -> float:
    p_assignments, q_assignments = kmeans_cluster_assignments(p, q, 10)

    cm = generalized_cluster_metric(p_assignments, q_assignments, 10)

    return cm, np.log(cm)
