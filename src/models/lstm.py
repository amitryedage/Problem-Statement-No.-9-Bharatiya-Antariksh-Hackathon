"""
PURPOSE:
  Predicts future turbulence parameters (r0, tau0, wind) from history.
  Enables PREDICTIVE AO — pre-position DM before turbulence arrives.
  Eliminates 10-20% Strehl loss from temporal lag.
PHYSICS BASIS:
  Taylor's frozen turbulence hypothesis:
    Atmosphere moves as a rigid screen at wind speed v.
    Future wavefront = current wavefront shifted by v × Δt.
    This makes future states PREDICTABLE from current measurements.
INPUT : (r0(t-19:t), tau0(t-19:t), vx(t-19:t), vy(t-19:t)) — last 20 frames
OUTPUT: (r0(t+1:t+5), tau0, vx, vy) — next 5 frames predicted
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from torch.utils.data import TensorDataset, DataLoader
from config import CHECKPOINT_DIR


class TurbulenceLSTM(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128,
                 num_layers=2, pred_horizon=5, dropout=0.2):
        super(TurbulenceLSTM, self).__init__()
        self.hidden_dim   = hidden_dim
        self.num_layers   = num_layers
        self.pred_horizon = pred_horizon
        self.input_dim    = input_dim

        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, pred_horizon * input_dim)
        )

    def forward(self, x):
        out, _      = self.lstm(x)
        last_hidden = out[:, -1, :]
        pred        = self.output_head(last_hidden)
        return pred.view(-1, self.pred_horizon, self.input_dim)


def build_turbulence_lstm(input_dim=4, hidden_dim=128,
                           num_layers=2, pred_horizon=5):
    """Build TurbulenceLSTM with given config."""
    return TurbulenceLSTM(input_dim, hidden_dim, num_layers, pred_horizon)


def prepare_lstm_sequences(r0_series, wind_speed,
                            wavelength=500e-9, D=2.0,
                            seq_len=15, pred_horizon=5):
    T        = len(r0_series)
    features = np.zeros((T, 4), dtype=np.float32)

    for i, r0 in enumerate(r0_series):
        r0_v  = max(float(r0), 0.005)
        tau0  = 0.314 * r0_v / (wind_speed + 1e-8)
        features[i] = [r0_v, tau0, wind_speed, 0.0]

    feat_min   = features.min(axis=0)
    feat_max   = features.max(axis=0)
    feat_range = feat_max - feat_min + 1e-8
    feat_norm  = (features - feat_min) / feat_range

    X_list, Y_list = [], []
    for start in range(T - seq_len - pred_horizon + 1):
        X_list.append(feat_norm[start            : start + seq_len])
        Y_list.append(feat_norm[start + seq_len  : start + seq_len + pred_horizon])

    if len(X_list) == 0:
        raise ValueError(
            f"Not enough data: need {seq_len+pred_horizon} frames, got {T}. "
            "Generate a longer time-series in Day 1.")

    return (np.array(X_list, dtype=np.float32),
            np.array(Y_list, dtype=np.float32),
            feat_min, feat_max)


def train_lstm(model, X, Y, n_epochs=150, lr=5e-4,
               batch_size=32, device='cpu',
               checkpoint_dir=CHECKPOINT_DIR):
    device_obj = torch.device(device)
    model      = model.to(device_obj)
    optimizer  = optim.Adam(model.parameters(), lr=lr)
    loss_fn    = nn.MSELoss()

    N       = len(X)
    N_train = int(0.8 * N)

    # Build DataLoader for training — data stays on CPU until batch needed
    train_dataset = TensorDataset(
        torch.tensor(X[:N_train]),
        torch.tensor(Y[:N_train])
    )
    train_loader  = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # shuffle every epoch — better generalisation
        drop_last=False
    )

    # Validation stays as tensors — small enough, no OOM risk
    X_va = torch.tensor(X[N_train:]).to(device_obj)
    Y_va = torch.tensor(Y[N_train:]).to(device_obj)

    train_losses, val_losses = [], []
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val = float('inf')

    for epoch in range(n_epochs):
        # ── Mini-batch training ──
        model.train()
        epoch_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device_obj)
            Y_batch = Y_batch.to(device_obj)
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch), Y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)

        # ── Validation ──
        model.eval()
        with torch.no_grad():
            loss_va = loss_fn(model(X_va), Y_va).item()

        train_losses.append(epoch_loss)
        val_losses.append(loss_va)

        if loss_va < best_val:
            best_val = loss_va
            torch.save({'model_state': model.state_dict(),
                        'val_loss': best_val},
                       os.path.join(checkpoint_dir, 'lstm_best.pth'))

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Train: {epoch_loss:.6f} | Val: {loss_va:.6f} | "
                  f"Best: {best_val:.6f}")

    return train_losses, val_losses


def load_lstm(checkpoint_path, device='cpu', **kwargs):
    """Load trained LSTM from checkpoint."""
    model = TurbulenceLSTM(**kwargs)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


def predict_turbulence(model, recent_features_norm,
                        feat_min, feat_max, device='cpu'):
    model.eval()
    x     = torch.tensor(recent_features_norm[None]).to(device)
    with torch.no_grad():
        pred_norm = model(x).cpu().numpy()[0]   # (pred_horizon, 4)

    feat_range  = feat_max - feat_min + 1e-8
    predictions = pred_norm * feat_range + feat_min
    return predictions.astype(np.float32)


if __name__ == '__main__':
    print("Testing lstm.py...")
    model = build_turbulence_lstm(input_dim=4, hidden_dim=64,
                                   num_layers=2, pred_horizon=5)
    x   = torch.randn(8, 15, 4)
    out = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    assert out.shape == (8, 5, 4)
    print("lstm.py OK")