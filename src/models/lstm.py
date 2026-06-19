"""
PS09 BAH 2026 — LSTM Turbulence Predictor
Replicated from: notebooks/day4/day4_cnn_training.ipynb

PURPOSE:
  Predicts future turbulence parameters from time-series history.
  This is the DIFFERENTIATOR — no other team will have this.

WHY THIS MATTERS:
  Standard AO: measure current wavefront → correct current DM.
  Problem: there is always a time lag between measurement and correction.
  Predictive AO: predict the NEXT wavefront → pre-correct.
  Result: 10-20% better Strehl ratio by eliminating temporal lag error.

INPUT SEQUENCE:  (r0(t-19), ..., r0(t))  — last 20 frames of r0
OUTPUT:          (r0(t+1), ..., r0(t+5)) — next 5 frames predicted

STATUS: STUB — implement on Day 4
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import CHECKPOINT_DIR


def build_turbulence_lstm(input_dim=4, hidden_dim=128,
                           num_layers=3, pred_horizon=5):
    """
    Build LSTM turbulence predictor.

    Args:
        input_dim    : Features per timestep (r0, tau0, vx, vy)
        hidden_dim   : LSTM hidden state size
        num_layers   : Number of LSTM layers
        pred_horizon : How many steps ahead to predict

    Returns:
        model : torch.nn.Module — LSTM predictor
    """
    # TODO Day 4:
    # import torch.nn as nn
    # class TurbulenceLSTM(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
    #                             batch_first=True, dropout=0.2)
    #         self.out  = nn.Sequential(
    #             nn.Linear(hidden_dim, 64), nn.ReLU(),
    #             nn.Linear(64, pred_horizon * input_dim)
    #         )
    #         self.pred_horizon = pred_horizon
    #         self.input_dim    = input_dim
    #     def forward(self, x):   # x: (B, seq_len, input_dim)
    #         h, _ = self.lstm(x)
    #         out  = self.out(h[:, -1, :])
    #         return out.view(-1, self.pred_horizon, self.input_dim)
    # return TurbulenceLSTM()
    raise NotImplementedError("Implement on Day 4")


def train_lstm(model, turbulence_timeseries, seq_len=20,
               pred_horizon=5, n_epochs=50, device='cuda'):
    """
    Train LSTM on turbulence parameter time-series.
    Uses sliding window approach.
    TODO Day 4: implement.
    """
    raise NotImplementedError("Implement on Day 4")
