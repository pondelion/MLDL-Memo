from functools import reduce
from typing import List
import numpy as np
from scipy.spatial import distance


def mahalanobis_distance(
    dflist: List[np.ndarray]
) -> List[List[float]]:
    merged = reduce(lambda x, y: np.vstack([x, y]), dflist)
    mean = merged.mean(axis=0)
    cov_inv = np.linalg.inv(np.cov(merged.T))

    return [[distance.mahalanobis(data, mean, cov_inv) for data in df] for df in dflist]
