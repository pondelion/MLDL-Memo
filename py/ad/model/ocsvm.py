from functools import reduce
from typing import List
from sklearn import svm
import numpy as np


class OneClassSVM:

    def __init__(
        self,
        nu: float=0.1,
        kernel: str='rbf',
        gamma: str='auto'
    ):
        self._nu = nu
        self._kernel = kernel
        self._gamma = gamma

    def fit(
        self,
        dflist: List[np.ndarray],
    ) -> None:
        merged = reduce(lambda x, y: np.vstack([x, y]), dflist)

        self._clf = svm.OneClassSVM(nu=self._nu, kernel=self._kernel, gamma=self._gamma)
        self._clf.fit(merged)

    def predict(
        self,
        dflist: List[np.ndarray],
    ) -> List[np.ndarray]:
        return [self._clf.predict(df) for df in dflist]

    def get_scores(
        self,
        dflist: List[np.ndarray],
    ) -> List[np.ndarray]:
        return [self._clf.score_samples(df) for df in dflist]

    def decision_function(
        self,
        dflist: List[np.ndarray],
    ) -> List[np.ndarray]:
        return [self._clf.decision_function(df) for df in dflist]
