from typing import List
from functools import reduce
import numpy as np
from sklearn.mixture import GaussianMixture


def gmm_log_probs(
    dflist: List[np.ndarray],
    n_components: int=2
) -> List[np.ndarray]:
    merged = reduce(lambda x, y: np.vstack([x, y]), dflist)

    gmm = GaussianMixture(n_components=n_components, covariance_type='full')

    gmm.fit(merged)

    return [gmm.score_samples(df) for df in dflist]
