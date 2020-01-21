from typing import List
from functools import reduce
import numpy as np
from sklearn.decomposition import PCA, KernelPCA


def _pca(
    dflist: List[np.ndarray],
    pca_type: str='normal',
    n_components: int=2
) -> List[np.ndarray]:
    merged = reduce(lambda x, y: np.vstack([x, y]), dflist)

    if pca_type == 'normal':
        model = PCA(n_components=n_components)
    elif pca_type == 'rbf_kernel':
        model = KernelPCA(n_components=n_components, kernel='rbf')
    else:
        model = PCA(n_components=n_components)

    model.fit(merged)

    return [model.transform(df) for df in dflist], model


def pca(
    dflist: List[np.ndarray],
    n_components: int=2
) -> List[np.ndarray]:
    return _pca(dflist, pca_type='normal', n_components=n_components)


def kernel_pca(
    dflist: List[np.ndarray],
    n_components: int=2
) -> List[np.ndarray]:
    return _pca(dflist, pca_type='rbf_kernel', n_components=n_components)
