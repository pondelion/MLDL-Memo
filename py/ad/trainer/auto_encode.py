from functools import reduce
from typing import List
import numpy as np
import torch
import torch.nn as nn
from ..models.auto_ecoder import (
    AutoEncoder,
    CNNAutoEncoder,
    VariationalAutoEncoder,
    CNNAutoEncoder2D,
    LSTMAutoEncoder
)
from .. import DEVICE


def fit_auto_encoder(
    dflist: List[np.ndarray],
    input_dim: int,
    batch_n: int=50,
    train_n: int=40000,
    learning_rate: float=1e-3,
    output_dim: int=2,
    ae_type: str='normal_nn',
    sequence_len: int=50,
    lstm_hidden_size: int=50,
    vae_latent_dim: int=2,
    device: str=DEVICE
):
    merged = reduce(lambda x, y: np.vstack([x, y]), dflist)
    data_n = merged.shape[0]

    if ae_type == 'normal_nn':
        ae = AutoEncoder(
            input_dim=input_dim,
            output_dim=output_dim
        )
    elif ae_type == 'cnn':
        ae = CNNAutoEncoder(
            input_dim=input_dim,
            output_dim=output_dim
        )
    elif ae_type == 'vae':
        ae = VariationalAutoEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=vae_latent_dim,
        )
    elif ae_type == 'lstm':
        ae = LSTMAutoEncoder(
            in_dim=input_dim,
            out_dim=output_dim,
            hidden_size=lstm_hidden_size,
            device=device,
        )

    ae = ae.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        ae.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    loss_history = []

    try:
        from tqdm import tqdm
        itr = tqdm(range(train_n))
    except Exception:
        itr = range(train_n)

    print('Train start.')
    for _ in itr:
        if ae_type == 'normal_nn':
            random_indices = np.random.permutation(data_n)[:batch_n]
            minibatch = merged[random_indices, :]
            minibatch = torch.stack([torch.from_numpy(minibatch[i, :]).float() for i in range(minibatch.shape[0])]).to(device)
        elif ae_type == 'cnn' or ae_type == 'vae':
            random_indices = np.random.permutation(data_n)[:batch_n]
            minibatch = merged[random_indices, :]
            minibatch = torch.stack([torch.from_numpy(minibatch[i, :]).float().unsqueeze(0) for i in range(minibatch.shape[0])]).to(device)
        elif ae_type == 'lstm':
            random_indices = np.random.permutation(data_n-sequence_len)[:batch_n]
            minibatch = torch.stack([torch.from_numpy(merged[idx:idx+sequence_len, :]) for idx in random_indices]).float().to(device)

        decoded = ae(minibatch)

        loss = criterion(minibatch, decoded)
        loss_history.append(loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_history, ae


def fit_cnn_auto_encoder(
    dflist: List[np.ndarray],
    input_dim: int,
    batch_n: int=50,
    train_n: int=40000,
    learning_rate: float=1e-3,
    output_dim: int=2,
):
    return fit_auto_encoder(dflist, input_dim, batch_n, train_n,
                            learning_rate, output_dim, ae_type='cnn')


def fit_vae(
    dflist: List[np.ndarray],
    input_dim: int,
    batch_n: int=50,
    train_n: int=40000,
    learning_rate: float=1e-3,
    output_dim: int=2,
):
    return fit_auto_encoder(dflist, input_dim, batch_n, train_n,
                            learning_rate, output_dim, ae_type='vae',
                            vae_latent_dim=output_dim)


def fit_lstm_auto_encoder(
    dflist: List[np.ndarray],
    input_dim: int,
    batch_n: int=50,
    train_n: int=40000,
    learning_rate: float=1e-3,
    sequence_len: int=50,
    hidden_size: int=2,
    device: str=DEVICE,
):
    return fit_auto_encoder(dflist, input_dim, batch_n, train_n,
                            learning_rate, input_dim, ae_type='lstm',
                            sequence_len=sequence_len, lstm_hidden_size=hidden_size,
                            device=device)


def _fit_auto_encoder2d(
    dflist: List[np.ndarray],
    input_dim_x: int=31,
    input_dim_y: int=31,
    batch_n: int=50,
    train_n: int=40000,
    learning_rate: float=1e-3,
    output_dim: int=2,
    ae_type: str='normal_nn'
):
    merged = reduce(lambda x, y: np.vstack([x, y]), dflist)
    data_n = merged.shape[0]

    if ae_type == 'normal_nn':
        pass
    elif ae_type == 'cnn':
        ae = CNNAutoEncoder2D(
            input_dim_x=input_dim_x,
            input_dim_y=input_dim_y,
            output_dim=output_dim
        )
    elif ae_type == 'vae':
        pass

    ae = ae.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        ae.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )

    loss_history = []

    try:
        from tqdm import tqdm
        itr = tqdm(range(train_n))
    except Exception:
        itr = range(train_n)

    print('Train start.')
    for _ in itr:
        random_indices = np.random.permutation(data_n-input_dim_y)[:batch_n]
        minibatch = np.array([merged[idx:idx+input_dim_y, :] for idx in random_indices])
        if ae_type == 'normal_nn':
            pass
        elif ae_type == 'cnn' or ae_type == 'vae':
            minibatch = torch.stack([torch.from_numpy(minibatch[i, :]).float().unsqueeze(0) for i in range(minibatch.shape[0])]).to(DEVICE)

        decoded = ae(minibatch)

        loss = criterion(minibatch, decoded)
        loss_history.append(loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_history, ae


def fit_cnn_auto_encoder2d(
    dflist: List[np.ndarray],
    input_dim_x: int=31,
    input_dim_y: int=31,
    batch_n: int=50,
    train_n: int=40000,
    learning_rate: float=1e-3,
    output_dim: int=2,
):
    return _fit_auto_encoder2d(dflist, input_dim_x, input_dim_y, batch_n, train_n,
                               learning_rate, output_dim, ae_type='cnn')
