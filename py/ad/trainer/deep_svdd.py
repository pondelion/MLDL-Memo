from typing import List
from functools import reduce
import torch
import torch.nn as nn
import numpy as np
from ..models.deep_svdd import DeepSVDD, DeepSVDD2D
from .. import DEVICE


def fit_deep_svdd(
    dflist: List[np.ndarray],
    input_dim: int=31,
    output_dim: int=2,
    in_channel: int=1,
    batch_n: int=50,
    train_n: int=40000,
    learning_rate: float=1e-3,
    weight_decay: float=1e-5,
    sampling_num: int=10000,
    l2_reg_coef: float=1.0,
    device: str=DEVICE,
    include_noise_reference_loss: bool=False,
    noise_reference_loss_weight: float=30.0,
    return_eval_mode: bool=True,
):
    merged = reduce(lambda x, y: np.vstack([x, y]), dflist)

    model = DeepSVDD(
        in_dim=input_dim,
        out_dim=output_dim,
        in_channel=in_channel
    ).to(device)
    # model.apply(model.weights_init)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    try:
        from tqdm import tqdm
        itr = tqdm(range(train_n))
    except Exception:
        itr = range(train_n)
    itr = range(train_n)

    loss_history = []

    for _ in itr:
        rd_indices = np.random.randint(0, merged.shape[0], batch_n)
        x = [torch.from_numpy(merged[idx, :]).float() for idx in rd_indices]
        x = torch.stack(x).view(batch_n, 1, merged.shape[1]).to(device)

        x_reduced = model(x)
        random_indices = np.random.permutation(merged.shape[0])[:sampling_num]
        sample_x_reduced = model(torch.from_numpy(merged[random_indices, :]).float().view(sampling_num, 1, input_dim).to(device))

        if include_noise_reference_loss:
            x_noised = merged[random_indices, :] + 0.1 * merged[random_indices, :].max() * np.random.randn(sampling_num, merged[random_indices, :].shape[1])
            x_noised = torch.from_numpy(x_noised).float().view(sampling_num, 1, input_dim).to(device)
            x_noised_reduced = model(x_noised)

        # 2次元に次元圧縮された全データの重心を求める
        c = torch.mean(sample_x_reduced, dim=0).to(device)  # (sampling_num, 2) => (1, 2)

        # 2次元に次元圧縮された各データの重心からの距離の2乗の全データ和を損失とする
        loss = torch.sum((x_reduced - c)**2).to(device)

        if include_noise_reference_loss:
            loss += noise_reference_loss_weight / torch.sum((x_noised_reduced - c)**2).to(device)

        l2_reg = None
        for w in model.parameters():
            if l2_reg is None:
                l2_reg = l2_reg_coef * w.norm(2).to(device)
            else:
                l2_reg += l2_reg_coef * w.norm(2).to(device)

        loss += l2_reg

        loss_history.append(loss.detach()[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if return_eval_mode:
        model.eval()

    return loss_history, model


def fit_deep_svdd2d(
    dflist: List[np.ndarray],
    input_dim_x: int=31,
    input_dim_y: int=31,
    output_dim: int=2,
    in_channel: int=1,
    batch_n: int=50,
    train_n: int=40000,
    learning_rate: float=1e-3,
    weight_decay: float=1e-5,
    sampling_num: int=10000,
    l2_reg_coef: float=1.0,
    device: str=DEVICE,
    include_noise_reference_loss: bool=False,
    noise_reference_loss_weight: float=30.0,
    return_eval_mode: bool=True,
    use_fixed_centroid: bool=True,
):
    merged = reduce(lambda x, y: np.vstack([x, y]), dflist)

    model = DeepSVDD2D(
        in_dim_x=input_dim_x,
        in_dim_y=input_dim_y,
        out_dim=output_dim,
        in_channel=in_channel
    ).to(device)
    # model.apply(model.weights_init)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    try:
        from tqdm import tqdm
        itr = tqdm(range(train_n))
    except Exception:
        itr = range(train_n)
    #itr = range(train_n)

    if use_fixed_centroid:
        centroids = None
        for i in range(int(merged.shape[0]/1000)):
            sliced_data = torch.stack([torch.from_numpy(merged[i*1000+idx:i*1000+idx+input_dim_y, :]).float() for idx in range(1000)]).float().view(1000, 1, input_dim_y, input_dim_x).to(device)
            slice_reduced = model(sliced_data)
            if centroids is None:
                centroids = torch.mean(slice_reduced, dim=0).cpu().detach().numpy()
            else:
                centroids = np.vstack([centroids, torch.mean(slice_reduced, dim=0).cpu().detach().numpy()])
        c = torch.mean(torch.from_numpy(centroids), dim=0).to(device)

    print(c)

    loss_history = []

    for _ in itr:
        rd_indices = np.random.randint(0, merged.shape[0]-input_dim_y, batch_n)
        x = [torch.from_numpy(merged[idx:idx+input_dim_y, :]).float() for idx in rd_indices]
        x = torch.stack(x).unsqueeze(1).to(device)

        x_reduced = model(x)
        random_indices = np.random.permutation(merged.shape[0]-input_dim_y)[:sampling_num]
        sample_data = torch.stack([torch.from_numpy(merged[idx:idx+input_dim_y, :]).float() for idx in random_indices]).float().view(sampling_num, 1, input_dim_y, input_dim_x).to(device)
        if not use_fixed_centroid:
            sample_x_reduced = model(sample_data)

        if include_noise_reference_loss:
            x_noised = sample_data + 0.1 * sample_data.max() * torch.randn(sampling_num, 1, input_dim_y, input_dim_x).to(device)
            x_noised = x_noised.to(device)
            x_noised_reduced = model(x_noised)

        if not use_fixed_centroid:
            # 2次元に次元圧縮された全データの重心を求める
            c = torch.mean(sample_x_reduced, dim=0).to(device)  # (sampling_num, 2) => (1, 2)

        # 2次元に次元圧縮された各データの重心からの距離の2乗の全データ和を損失とする
        loss = torch.sum((x_reduced - c)**2).to(device)

        if include_noise_reference_loss:
            loss += noise_reference_loss_weight / torch.sum((x_noised_reduced - c)**2).to(device)

        l2_reg = None
        for w in model.parameters():
            if l2_reg is None:
                l2_reg = l2_reg_coef * w.norm(2).to(device)
            else:
                l2_reg += l2_reg_coef * w.norm(2).to(device)

        loss += l2_reg

        loss_history.append(loss.detach()[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if return_eval_mode:
        model.eval()

    return loss_history, model


def svdd_transform(
    dflist: List[np.ndarray],
    model,
    device: str=DEVICE,
):
    return [model(torch.from_numpy(df).float().view(df.shape[0], 1, df.shape[1]).to(device)).detach() for df in dflist]


def svdd_transform2d(
    dflist: List[np.ndarray],
    model,
    input_dim_y: int,
    device: str=DEVICE,
):
    input_data_list = [[torch.from_numpy(df[i:i+input_dim_y, :]) for i in range(df.shape[0]-input_dim_y)] for df in dflist]
    return [model(torch.stack(data).float().unsqueeze(1).to(device)).detach().cpu() for data in input_data_list]


def get_centroid(
    reduced_list: List[np.ndarray],
    device: str=DEVICE,
):
    return torch.stack([torch.mean(reduced, dim=0) for reduced in reduced_list]).to(device).detach()


def svdd_loss(
    dflist: List[np.ndarray],
    model,
    centroid=None,
    device: str=DEVICE,
):
    reduced_list = svdd_transform(dflist, model, device)
    if centroid is None:
        merged = reduce(lambda x, y: np.vstack([x, y]), dflist)
        centroid = get_centroid([merged])[0]
    return [((reduced - centroid)**2).sum(dim=1).detach().cpu().numpy() for reduced in reduced_list]
