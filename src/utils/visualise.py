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



