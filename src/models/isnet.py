"""
PAPER: DuBose et al. (2020) Intensity-Slopes Network
WHY DUAL INPUT:
  slopes alone → miss branch points (curl component of wavefront)
  spot images  → reveal branch points through spot shape/fragmentation
  both together→ reconstructs gradient + curl = full wavefront
ARCHITECTURE:
  CNN branch  : (1,H,W) spot image → (2048,) spatial features
  FC branch   : (200,) slopes      → (256,)  gradient features
  Fusion      : concat(2304,)      → (36,)   Zernike coefficients
CONNECTS TO FINAL:
  Trained in Day 4 → evaluated in Day 5 → deployed in demo
  Exported to ONNX for C++ deployment on ISRO hardware
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import N_SLOPES, N_ZERNIKE_MODES, GRID_SIZE, CHECKPOINT_DIR


class ISNet(nn.Module):
    """
    Intensity-Slopes Network for wavefront reconstruction.

    Key insight: spot IMAGES contain branch point signatures
    (elongation, fragmentation, asymmetry) invisible to slope
    centroiding alone. Feeding both inputs to the network
    achieves 30-60% better Strehl at strong turbulence.
    """

    def __init__(self, n_slopes=N_SLOPES, n_modes=N_ZERNIKE_MODES,
                 grid_size=GRID_SIZE):
        super(ISNet, self).__init__()
        self.n_modes  = n_modes
        self.n_slopes = n_slopes

        # ── Branch 1: CNN for spot images ──
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(1,  32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )   # → (B, 2048)

        # ── Branch 2: FC for slope vectors ──
        self.slope_branch = nn.Sequential(
            nn.Linear(n_slopes, 256), nn.ReLU(True), nn.Dropout(0.2),
            nn.Linear(256, 256),      nn.ReLU(True),
        )   # → (B, 256)

        # ── Fusion ──
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 256, 512), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(512, 128),        nn.ReLU(True), nn.Dropout(0.2),
            nn.Linear(128, n_modes)
        )   # → (B, 36)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                         nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, spot_image, slopes):
        """
        Args:
            spot_image : (B, 1, H, W) float32 — normalised [0,1]
            slopes     : (B, 2*N_sub) float32
        Returns:
            zernike_coeffs : (B, N_modes) float32 in nm
        """
        f_cnn   = self.cnn_branch(spot_image)
        f_slope = self.slope_branch(slopes)
        return self.fusion(torch.cat([f_cnn, f_slope], dim=1))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_numpy(self, spot_np, slopes_np, device='cpu'):
        """
        Convenience method: numpy arrays in, numpy array out.
        Used in evaluation and demo pipeline.

        Args:
            spot_np   : (H, W) uint8 spot image
            slopes_np : (2*N_sub,) float32 slopes

        Returns:
            coeffs : (N_modes,) float32 Zernike coefficients
        """
        self.eval()
        spot_t  = torch.tensor(
            spot_np[None, None].astype(np.float32) / 255.0).to(device)
        slope_t = torch.tensor(slopes_np[None]).to(device)
        with torch.no_grad():
            out = self.forward(spot_t, slope_t)
        return out.cpu().numpy()[0]


def build_isnet(n_slopes=N_SLOPES, n_modes=N_ZERNIKE_MODES,
                grid_size=GRID_SIZE):
    """Build ISNet with default config. Returns model on CPU."""
    return ISNet(n_slopes=n_slopes, n_modes=n_modes, grid_size=grid_size)


def train_isnet(model, train_loader, val_loader,
                n_epochs=50, lr=1e-3, device='cuda',
                checkpoint_dir=CHECKPOINT_DIR):
    """
    Full training loop for ISNet.

    Features:
      AdamW + weight decay     → prevents overfitting
      CosineAnnealingLR        → smooth LR decay
      Mixed precision (amp)    → 40% memory reduction on GPU
      Gradient clipping        → prevents exploding gradients
      Best checkpoint saving   → automatic model selection

    Returns:
        train_losses, val_losses : lists of per-epoch mean loss
    """
    device_obj  = torch.device(device)
    model       = model.to(device_obj)
    optimizer   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(
                      optimizer, T_max=n_epochs, eta_min=1e-5)
    loss_fn     = nn.MSELoss()
    use_amp     = (device_obj.type == 'cuda')
    scaler      = torch.cuda.amp.GradScaler(enabled=use_amp)

    os.makedirs(checkpoint_dir, exist_ok=True)
    train_losses, val_losses = [], []
    best_val   = float('inf')
    best_epoch = 0

    for epoch in range(n_epochs):
        # Training
        model.train()
        tr_loss = 0.0
        for spot, slopes, coeffs in train_loader:
            spot   = spot.to(device_obj)
            slopes = slopes.to(device_obj)
            coeffs = coeffs.to(device_obj)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = loss_fn(model(spot, slopes), coeffs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # Validation
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for spot, slopes, coeffs in val_loader:
                spot   = spot.to(device_obj)
                slopes = slopes.to(device_obj)
                coeffs = coeffs.to(device_obj)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    va_loss += loss_fn(model(spot, slopes), coeffs).item()
        va_loss /= len(val_loader)

        scheduler.step()
        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        if va_loss < best_val:
            best_val   = va_loss
            best_epoch = epoch + 1
            torch.save({'epoch': best_epoch,
                        'model_state': model.state_dict(),
                        'val_loss': best_val},
                       os.path.join(checkpoint_dir, 'isnet_best.pth'))

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Train: {tr_loss:.4f} | Val: {va_loss:.4f} | "
                  f"Best: ep{best_epoch}")

    return train_losses, val_losses


