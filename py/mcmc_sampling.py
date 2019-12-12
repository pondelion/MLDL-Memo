import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import os
from mpl_toolkits.mplot3d.axes3d import Axes3D
%matplotlib inline


# 尤度関数
def likelihood(x):
    # 混合ガウス分布
    return np.exp(-(x-1)**2 / 0.2) + np.exp(-(x+1)**2 / 0.2)


x = np.random.rand()

xs = []
xs.append(x)

DELTA_X = 0.02

for _ in range(2000000):
    # 1/2の確率で
    # 左を遷移方向の候補
    if np.random.rand() < 0.5:
        # 左の方が高い場合
        if likelihood(xs[-1]-DELTA_X) > likelihood(xs[-1]):
            # 左へ遷移
            xs.append(xs[-1]-DELTA_X)
        # 左の方が低い場合
        else:
            # 左の高さ/現在の高さ の確率で左へ遷移
            if likelihood(xs[-1]-DELTA_X)/likelihood(xs[-1]) > np.random.rand():
                xs.append(xs[-1]-DELTA_X)
            else:
                # 遷移キャンセル
                xs.append(xs[-1])
    # 右を遷移方向の候補
    else:
        # 右の方が高い場合
        if likelihood(xs[-1]+DELTA_X) > likelihood(xs[-1]):
            # 右へ遷移
            xs.append(xs[-1]+DELTA_X)
        else:
            # 左の高さ/現在の高さ の確率で右へ遷移
            if likelihood(xs[-1]+DELTA_X)/likelihood(xs[-1]) > np.random.rand():
                xs.append(xs[-1]+DELTA_X)
            else:
                # 遷移キャンセル
                xs.append(xs[-1])
                
plt.hist(xs, bins=40)
