from typing import List
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
from ..models.lstm import LSTM
from ..import DEVICE


def fit_lstm(
    dflist: List[np.ndarray],
    sequence_len: int,
    input_dim: int=31,
    output_dim: int=31,
    hidden_dim: int=16,
    num_layers: int=2,
    batch_n: int=50,
    train_n: int=40000,
    learning_rate: float=1e-3,
):
    merged = reduce(lambda x, y: np.vstack([x, y]), dflist)
    data_n = merged.shape[0]

    model = LSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
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
    for n in itr:
        random_indices = np.random.permutation(data_n-(sequence_len+1))[:batch_n]
        batch = [merged[idx:idx+sequence_len+1, :] for idx in random_indices]
        x = torch.stack([torch.from_numpy(data[:-1, :]).float() for data in batch]).float().to(DEVICE)
        y = torch.stack([torch.from_numpy(data[1:, :]).float() for data in batch]).float().to(DEVICE)

        y_pred = model(x)

        loss = criterion(y, y_pred)
        loss_history.append(loss.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_history, model


def lstm_loss(
    dflist: List[np.ndarray],
    model,
    sequence_len: int,
    device: str=DEVICE,
)->List[np.array]:
    #merged = reduce(lambda x, y: np.vstack([x, y]), dflist)
    model = model.to(device)

    criterion = nn.MSELoss()

    losses = []

    for df in dflist:
        loss = []
        data_n = df.shape[0]

        x = torch.stack([torch.from_numpy(df[idx:idx+sequence_len, :]).float().to(device) for idx in range(data_n - sequence_len)])
        y = torch.stack([torch.from_numpy(df[idx+1:idx+sequence_len+1, :]).float().to(device) for idx in range(data_n - sequence_len)])

        y_pred = model(x)

        loss = ((y - y_pred)**2).sum(dim=(1, 2)).detach().cpu().numpy() / (y.shape[1]*y.shape[2])

        losses.append(loss)

    return losses


def lstm_loss_random_sampling(
    df: np.ndarray,
    model,
    sequence_len: int,
    sampling_num: int,
)->np.array:
    data_n = df.shape[0]

    model = model.to(DEVICE)

    random_indices = np.random.permutation(data_n-sequence_len)[:sampling_num]

    x = torch.stack([torch.from_numpy(df[idx:idx+sequence_len, :]).float().to(DEVICE) for idx in random_indices])
    y = torch.stack([torch.from_numpy(df[idx+1:idx+sequence_len+1, :]).float().to(DEVICE) for idx in random_indices])

    y_pred = model(x)

    losses = ((y - y_pred)**2).sum(dim=(1, 2)).detach().cpu().numpy() / (y.shape[1]*y.shape[2])

    return losses
