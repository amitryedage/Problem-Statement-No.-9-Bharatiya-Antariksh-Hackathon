# 4 Task are excuted in this section 
# get_zernike_basis() — 36 modes on NxN grid using aotools
# decompose_into_zernike() — phase map → 36 coefficients (CNN training labels)
# reconstruct_from_coefficients() — 36 coefficients → phase map (CNN output decoder)
# build_interaction_matrix() — builds D matrix and D_pinv for Day 3 classical reconstruction
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import N_ZERNIKE_MODES, GRID_SIZE, N_LENSLETS, D


def get_zernike_basis(n_modes=N_ZERNIKE_MODES, grid_size=GRID_SIZE):
    import aotools
    return aotools.zernikeArray(n_modes, grid_size).astype(np.float32)


def decompose_into_zernike(phase_map, n_modes=N_ZERNIKE_MODES,
                            grid_size=GRID_SIZE):
   
    Z = get_zernike_basis(n_modes, grid_size)
    Z_flat = Z.reshape(n_modes, -1)
    phase_flat= phase_map.flatten()

    # Mask NaN (outside aperture)
    valid = ~np.isnan(phase_flat)
    coeffs, _, _, _ = np.linalg.lstsq(
        Z_flat[:, valid].T, phase_flat[valid], rcond=None)

    return coeffs.astype(np.float32)


def reconstruct_from_coefficients(coefficients, grid_size=GRID_SIZE):
    n_modes = len(coefficients)
    Z       = get_zernike_basis(n_modes, grid_size)
    phase   = np.tensordot(
        coefficients.astype(np.float64),
        Z.astype(np.float64),
        axes=[[0], [0]]
    )
    return phase.astype(np.float32)


def build_interaction_matrix(n_modes=N_ZERNIKE_MODES,
                              n_lenslets=N_LENSLETS,
                              D_aperture=D,
                              grid_size=GRID_SIZE,
                              alpha_tikhonov=0.01):
    
    import aotools
    n_sub = n_lenslets ** 2
    D_mat = np.zeros((2 * n_sub, n_modes), dtype=np.float64)
    step  = D_aperture / n_lenslets
    Z_all = aotools.zernikeArray(n_modes, grid_size)

    for sub_i in range(n_sub):
        row = sub_i // n_lenslets
        col = sub_i  % n_lenslets
        x_p = -D_aperture/2 + step*(col + 0.5)
        y_p = -D_aperture/2 + step*(row + 0.5)
        x_n = x_p / (D_aperture / 2)
        y_n = y_p / (D_aperture / 2)
        r   = np.sqrt(x_n**2 + y_n**2)
        if r > 1.0:
            continue

        # Pixel coordinates in Zernike array
        px  = int(np.clip((x_n+1)/2 * (grid_size-1), 1, grid_size-2))
        py  = int(np.clip((y_n+1)/2 * (grid_size-1), 1, grid_size-2))

        for mode_j in range(n_modes):
            Z_j  = Z_all[mode_j]
            dZdx = (Z_j[py, px+1] - Z_j[py, px-1]) / 2.0
            dZdy = (Z_j[py+1, px] - Z_j[py-1, px]) / 2.0
            D_mat[sub_i,          mode_j] = dZdx
            D_mat[n_sub + sub_i,  mode_j] = dZdy

    # Tikhonov regularised pseudoinverse
    DtD    = D_mat.T @ D_mat
    alpha2 = (alpha_tikhonov * np.max(np.diag(DtD) + 1e-12))**2
    D_pinv = np.linalg.solve(DtD + alpha2*np.eye(n_modes), D_mat.T)

    return D_mat.astype(np.float32), D_pinv.astype(np.float32)


if __name__ == '__main__':
    print("Testing zernike.py...")
    import aotools
    Z = get_zernike_basis(36, 64)
    print(f"  Basis shape: {Z.shape}")
    dummy_phase = Z[3] * 50 + Z[1] * 30
    coeffs = decompose_into_zernike(dummy_phase, 36, 64)
    print(f"  Coeffs shape: {coeffs.shape}")
    print(f"  Max coeff at index {np.argmax(np.abs(coeffs))}"
          f" = {coeffs[np.argmax(np.abs(coeffs))]:.2f}nm")
    recon = reconstruct_from_coefficients(coeffs, 64)
    rms   = float(np.sqrt(np.mean((dummy_phase - recon)**2)))
    print(f"  Reconstruction RMS: {rms:.2f}nm")
    print("zernike.py OK")
