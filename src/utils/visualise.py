import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import FIGURES_DIR


def _save(fig, filename, dpi=200):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {path}")
    return path


def plot_wavefront_comparison(phase_maps, r0_values,
                               filename='proposal_fig1_wavefronts.png'):
    n   = len(phase_maps)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]

    for ax, phase, r0 in zip(axes, phase_maps, r0_values):
        vmax = float(np.nanpercentile(np.abs(phase), 98))
        im   = ax.imshow(phase, cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, origin='lower')
        rms  = float(np.sqrt(np.nanmean(phase**2)))
        s    = float(np.exp(-(2*np.pi*rms/500)**2))
        ax.set_title(f'r₀={r0*100:.0f}cm\nRMS={rms:.0f}nm | Strehl={s:.3f}', fontsize=10)
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        plt.colorbar(im, ax=ax, label='Phase (nm)', fraction=0.046)

    fig.suptitle('Wavefront Phase Maps — CNN Reconstruction Targets\n'
                 'Red=ahead, Blue=behind | r₀=3cm shows branch points',
                 fontsize=11)
    plt.tight_layout()
    return _save(fig, filename)


"""
PS09 BAH 2026 — Visualisation Utilities
Generates all proposal figures and demo visualisations.

FIGURES GENERATED:
  proposal_fig1_wavefronts.png    → Section 1 (from Day 1)
  proposal_fig2_cnn_io.png        → Section 2 (from Day 1)
  proposal_fig3_comparison.png    → Section 4 (from Day 5)
  proposal_fig4_timeseries.png    → Section 4 (from Day 3)
  proposal_fig5_actuator.png      → Section 2 (from Day 4)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import FIGURES_DIR


def _save(fig, filename, dpi=200):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"  Saved: {path}")
    return path


def plot_wavefront_comparison(phase_maps, r0_values,
                               filename='proposal_fig1_wavefronts.png'):
    """
    3-panel wavefront comparison. → Proposal Figure 1.
    Built in Day 1. Goes directly into proposal Section 1.
    """
    n   = len(phase_maps)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    if n == 1:
        axes = [axes]

    for ax, phase, r0 in zip(axes, phase_maps, r0_values):
        vmax = float(np.nanpercentile(np.abs(phase), 98))
        im   = ax.imshow(phase, cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax, origin='lower')
        rms  = float(np.sqrt(np.nanmean(phase**2)))
        s    = float(np.exp(-(2*np.pi*rms/500)**2))
        ax.set_title(f'r₀={r0*100:.0f}cm\nRMS={rms:.0f}nm | Strehl={s:.3f}', fontsize=10)
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        plt.colorbar(im, ax=ax, label='Phase (nm)', fraction=0.046)

    fig.suptitle('Wavefront Phase Maps — CNN Reconstruction Targets\n'
                 'Red=ahead, Blue=behind | r₀=3cm shows branch points',
                 fontsize=11)
    plt.tight_layout()
    return _save(fig, filename)


def plot_cnn_input_output(spot_images, phase_maps, r0_values,
                           filename='proposal_fig2_cnn_io.png'):
    """
    2-row figure: spot images (top) vs phase maps (bottom). → Proposal Figure 2.
    The most powerful visualisation in the entire proposal.
    """
    n   = len(spot_images)
    fig = plt.figure(figsize=(6*n, 10))
    gs  = gridspec.GridSpec(2, n, hspace=0.4, wspace=0.3)

    for col, (spot, phase, r0) in enumerate(zip(spot_images, phase_maps, r0_values)):
        ax_t = fig.add_subplot(gs[0, col])
        ax_t.imshow(spot, cmap='hot', origin='lower')
        ax_t.set_title(f'CNN INPUT — r₀={r0*100:.0f}cm\nSH-WFS spot image (BMP)', fontsize=10)
        ax_t.axis('off')

        ax_b  = fig.add_subplot(gs[1, col])
        vmax  = float(np.nanpercentile(np.abs(phase), 98))
        im    = ax_b.imshow(phase, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
        rms   = float(np.sqrt(np.nanmean(phase**2)))
        ax_b.set_title(f'CNN TARGET — W(xi,yi)\nPhase map | RMS={rms:.0f}nm', fontsize=10)
        ax_b.axis('off')
        plt.colorbar(im, ax=ax_b, fraction=0.046, label='Phase (nm)')

    fig.suptitle('PS09 Problem: SH-WFS Spot Image → Wavefront Phase Map\n'
                 'CNN learns this mapping for all turbulence conditions',
                 fontsize=12, fontweight='bold')
    return _save(fig, filename)


def plot_rms_comparison(r0_values, results_dict,
                         filename='proposal_fig3_comparison.png'):
    """
    RMS comparison: CNN vs 3 classical baselines. → Proposal Figure 3.
    Generated on Day 5. The key evidence in proposal Section 4.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors  = {'modal':'#E24B4A','zonal':'#EF9F27',
                'direct':'#185FA5','ISNet':'#1D9E75'}
    r0_cm   = [r*100 for r in r0_values]

    for name, rms_list in results_dict.items():
        style  = '-o' if name == 'ISNet' else '--s'
        lw     = 2.5  if name == 'ISNet' else 1.5
        ax1.plot(r0_cm, rms_list, style,
                 label=name, color=colors.get(name,'gray'),
                 linewidth=lw, markersize=7)

    ax1.axhline(50, color='green', linestyle=':', linewidth=1.5,
                label='Target: 50nm RMS')
    ax1.axvspan(0, 7, alpha=0.08, color='red', label='CNN advantage zone')
    ax1.set_xlabel('Fried Parameter r₀ (cm)', fontsize=12)
    ax1.set_ylabel('RMS Wavefront Error (nm)', fontsize=12)
    ax1.set_title('RMS Error vs Turbulence Strength', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    for name, rms_list in results_dict.items():
        strehl = [float(np.exp(-(2*np.pi*r/500)**2)) for r in rms_list]
        style  = '-o' if name == 'ISNet' else '--s'
        lw     = 2.5  if name == 'ISNet' else 1.5
        ax2.plot(r0_cm, strehl, style,
                 label=name, color=colors.get(name,'gray'),
                 linewidth=lw, markersize=7)

    ax2.axhline(0.7, color='green', linestyle=':', linewidth=1.5,
                label='Target: Strehl > 0.70')
    ax2.set_xlabel('Fried Parameter r₀ (cm)', fontsize=12)
    ax2.set_ylabel('Strehl Ratio', fontsize=12)
    ax2.set_title('Strehl Ratio vs Turbulence Strength', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)

    fig.suptitle('ISNet CNN vs Classical Baselines — ISRO PS09 Evaluation',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return _save(fig, filename)


def plot_turbulence_timeseries(times_ms, r0_series, tau0_series=None,
                                lstm_pred=None,
                                filename='proposal_fig4_timeseries.png'):
    """
    r0 + tau0 time-series with LSTM prediction. → Proposal Figure 4.
    Generated on Day 3 (estimation) + Day 4 (LSTM prediction overlay).
    """
    n_rows = 2 if tau0_series is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4*n_rows))
    if n_rows == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(times_ms, np.array(r0_series)*100, 'b-',
            linewidth=2, label='Measured r₀(t)')
    if lstm_pred is not None:
        pred_times = times_ms[-len(lstm_pred):]
        ax.plot(pred_times, np.array(lstm_pred)*100, 'r--',
                linewidth=2, label='LSTM predicted r₀(t+Δt)', alpha=0.8)
    ax.set_ylabel('r₀ (cm)', fontsize=12)
    ax.set_title('Fried Parameter r₀ Over Time — Turbulence Characterisation', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (ms)')

    if tau0_series is not None and len(axes) > 1:
        axes[1].plot(times_ms, np.array(tau0_series)*1000, 'g-',
                     linewidth=2, label='Measured τ₀(t) (ms)')
        axes[1].set_ylabel('τ₀ (ms)', fontsize=12)
        axes[1].set_xlabel('Time (ms)', fontsize=12)
        axes[1].set_title('Coherence Time τ₀ Over Time', fontsize=11)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return _save(fig, filename)

