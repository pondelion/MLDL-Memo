import itertools
from typing import List
from functools import reduce
import numpy as np
from sklearn.manifold import TSNE


def tsne(
    dflist: List[np.ndarray],
    n_components: int=2,
    random_state: int=42,
) -> List[np.ndarray]:
    merged = reduce(lambda x, y: np.vstack([x, y]), dflist)

    model = TSNE(n_components=n_components, random_state=random_state)

    reduced = model.fit_transform(merged)

    lens = [df.shape[0] for df in dflist]
    cumsum = [0] + list(itertools.accumulate(lens))

    return [reduced[cumsum[i]:cumsum[i]+lens[i], :] for i in range(len(dflist))]
