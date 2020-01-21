
from typing import List
import numpy as np
from ..models.ocsvm import OneClassSVM


def ocsvm_scores(
    dflist: List[np.ndarray],
    anom_dflist: List[np.ndarray]=None,
    nu: float=0.1,
    kernel: str='rbf',
    gamma: str='auto'
) -> List[np.ndarray]:
    clf = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    print('Fitting...')
    clf.fit(dflist)

    print('Calculating scores...')
    if anom_dflist is None:
        return clf.get_scores(dflist)
    else:
        return clf.get_scores(dflist), clf.get_scores(anom_dflist)
