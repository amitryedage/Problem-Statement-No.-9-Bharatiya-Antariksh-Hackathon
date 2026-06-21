"""
PURPOSE:
  Output : A(xi,yi) — actuator stroke map with coupling.
 PHYSICS:
  DM corrects wavefront by applying its CONJUGATE:
    target = -wavefront_phase_nm     (negative = conjugate)
    act_commands = IF_pinv @ target  (solve using influence function)
  Inter-actuator coupling:
    Moving actuator A also moves mirror at neighbouring actuators.
    The influence function IF encodes this coupling.
    IF_pinv automatically accounts for it in the solution.
    No additional code needed — coupling is in the matrix.
"""

import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import GRID_SIZE


def create_synthetic_influence_function(n_actuators_side=8,
                                         n_pixels=GRID_SIZE,
                                         coupling=0.15):
    """
    Create synthetic Gaussian influence function for testing.
    Used BEFORE ISRO provides real DM data.

    Physics:
      Each actuator deforms the mirror in a Gaussian-shaped region.
      coupling = fraction of peak deformation at adjacent actuator.
      Typical DM coupling: 10-20%.

    Args:
        n_actuators_side : actuators per side (NxN grid)
        n_pixels         : mirror surface resolution
        coupling         : inter-actuator coupling (0=none, 0.15=typical)

    Returns:
        IF      : (n_pix², n_act²) influence function
        IF_pinv : (n_act², n_pix²) pseudoinverse
    """
    n_act    = n_actuators_side ** 2
    n_pix    = n_pixels
    IF       = np.zeros((n_pix * n_pix, n_act), dtype=np.float32)
    step     = n_pix / n_actuators_side
    y_pix, x_pix = np.mgrid[0:n_pix, 0:n_pix]

    for act_idx in range(n_act):
        row   = act_idx // n_actuators_side
        col   = act_idx  % n_actuators_side
        cx    = col * step + step / 2
        cy    = row * step + step / 2
        sigma = step * (0.7 + coupling)
        gauss = np.exp(-((x_pix - cx)**2 + (y_pix - cy)**2) / (2 * sigma**2))
        IF[:, act_idx] = gauss.flatten()

    IF_pinv = np.linalg.pinv(IF.astype(np.float64)).astype(np.float32)
    print(f"  IF shape    : {IF.shape}")
    print(f"  IF_pinv shape: {IF_pinv.shape}")
    print(f"  Coupling    : {coupling*100:.0f}% (Gaussian sigma={sigma:.1f}px)")
    return IF, IF_pinv

#Used when real time data is provided 
def load_influence_function(if_path):

    if if_path.endswith('.npz'):
        data    = np.load(if_path)
        IF      = data['IF'].astype(np.float64)
    else:
        IF      = np.load(if_path).astype(np.float64)

    IF_pinv = np.linalg.pinv(IF).astype(np.float32)
    print(f"Loaded IF from {if_path}: shape {IF.shape}")
    return IF.astype(np.float32), IF_pinv


def compute_actuator_map(wavefront_phase_nm, IF_pinv,
                          n_actuators_side=8):
    """
    Convert reconstructed wavefront to DM actuator commands.
     Output 5: A(xi,yi) in actuator stroke length.

    Algorithm:
      1. target     = -wavefront_phase_nm  (conjugate = cancel distortion)
      2. act_cmds   = IF_pinv @ target.flatten()
      3. act_map    = act_cmds.reshape(n_act_side, n_act_side)

    Inter-actuator coupling:
      IF_pinv encodes all coupling effects.
      The solution automatically compensates for them.
      No additional correction code needed.

    Args:
        wavefront_phase_nm : (H, W) float32 — reconstructed phase in nm
        IF_pinv            : (N_act², N_pix²) float32
        n_actuators_side   : actuators per side

    Returns:
        actuator_map  : (N_act, N_act) float32 — stroke in nm
        actuator_cmds : (N_act²,) float32 — 1D command vector
    """
    phase_flat = wavefront_phase_nm.copy().astype(np.float32)

    # Replace NaN (outside aperture) with 0 — no correction outside aperture
    phase_flat = np.nan_to_num(phase_flat, nan=0.0)

    # Conjugate: DM must apply OPPOSITE shape to cancel wavefront
    target     = -phase_flat.flatten()

    # Matrix multiply: solve IF @ act_cmds ≈ target
    # IF_pinv handles inter-actuator coupling automatically
    act_cmds   = (IF_pinv @ target.astype(np.float64)).astype(np.float32)
    act_map    = act_cmds.reshape(n_actuators_side, n_actuators_side)

    return act_map, act_cmds


def compute_correction_quality(wavefront_nm, act_map, IF,
                                n_actuators_side=8):
    """
    Verify actuator map quality: reconstruct what DM surface would look like
    and compare with target.

    Args:
        wavefront_nm     : (H, W) original wavefront
        act_map          : (N_act, N_act) actuator commands
        IF               : (N_pix², N_act²) influence function

    Returns:
        corrected_nm : (H, W) residual wavefront after DM correction
        residual_rms : float — residual RMS in nm (lower = better DM)
    """
    n_pix       = int(np.sqrt(IF.shape[0]))
    act_cmds    = act_map.flatten().astype(np.float64)
    dm_surface  = (IF.astype(np.float64) @ act_cmds).reshape(n_pix, n_pix)

    phase_flat  = np.nan_to_num(wavefront_nm.copy(), nan=0.0).astype(np.float64)
    corrected   = (phase_flat + dm_surface).astype(np.float32)
    valid       = ~np.isnan(wavefront_nm)
    residual_rms= float(np.sqrt(np.mean(corrected[valid]**2)))

    return corrected, residual_rms


if __name__ == '__main__':
    print("Testing dm_control.py...")
    print("  Creating synthetic influence function...")
    IF, IF_pinv = create_synthetic_influence_function(
        n_actuators_side=8, n_pixels=64, coupling=0.15)

    # Test with random wavefront
    phase = np.random.randn(64, 64).astype(np.float32) * 100
    act_map, act_cmds = compute_actuator_map(phase, IF_pinv, 8)
    print(f"  Actuator map : {act_map.shape}")
    print(f"  Max stroke   : {np.abs(act_cmds).max():.1f}nm")

    corrected, rms = compute_correction_quality(phase, act_map, IF, 8)
    print(f"  Input RMS    : {float(np.sqrt(np.mean(phase**2))):.1f}nm")
    print(f"  Residual RMS : {rms:.1f}nm")
    print(f"  Correction   : {(1 - rms/float(np.sqrt(np.mean(phase**2))))*100:.0f}%")
    print("dm_control.py OK")
