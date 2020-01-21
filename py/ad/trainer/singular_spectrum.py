from typing import List
from functools import reduce
import numpy as np


def singular_spectrum(
    dflist: List[np.ndarray],
    window_size: int,
    k: int,
    lag: int,
) -> np.ndarray:
    """特異スペクトル分解

    Args:
        dflist (List[np.ndarray]): 時系列データフレームのリスト
        window_size (int): 時間方向で抜き出すデータ数
        k (int): 履歴/テスト行列の列数
        lag (int): 履歴行列とテスト行列の時間差

    Returns:
        np.ndarray: (データ長-k-windowsize-lag, 特徴量数)の異常度行列
    """
    merged = reduce(lambda x, y: np.vstack([x, y]), dflist)
    feat_num = merged.shape[1]
    data_len = merged.shape[0]

    try:
        from tqdm import tqdm
        itr = tqdm(range(feat_num))
    except Exception:
        itr = range(feat_num)

    df_anom = []
    for feat_idx in itr:
        anoms = []
        for t in range(data_len-k-window_size-lag):
            hsitory_mats = []
            test_mats = []

            for i in range(k):
                hsitory_mats.append(merged[t+i:t+window_size+i, feat_idx])
                test_mats.append(merged[t+i+lag:t+window_size+i+lag, feat_idx])

            hsitory_mats = np.array(hsitory_mats).T
            test_mats = np.array(test_mats).T

            U_hist, S_hist, V_hist = np.linalg.svd(hsitory_mats)
            U_test, S_test, V_test = np.linalg.svd(test_mats)

            anom = 1 - np.linalg.svd(np.matmul(U_hist.T, U_test), compute_uv=False)[0]

            anoms.append(anom)
        df_anom.append(anoms)

    return np.array(df_anom).T
