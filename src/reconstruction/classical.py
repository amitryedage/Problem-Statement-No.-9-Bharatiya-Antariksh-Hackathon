"""modal_reconstruct() — one matrix multiply: D_pinv @ slopes
zonal_reconstruct() — Hudgin finite difference geometry
direct_integrate() — cumulative sum, fastest method
benchmark_all_methods() — runs all 3, returns comparison table
compute_slope_curl() — identifies branch point locations """
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import N_LENSLETS, N_ZERNIKE_MODES, GRID_SIZE, D, WAVELENGTH


def modal_reconstruct(slopes, D_pinv, grid_size=GRID_SIZE,
                       n_modes=N_ZERNIKE_MODES):
    N    = n_lenslets
    step = D_aperture / N
    sx   = slopes[:N*N].reshape(N, N).astype(np.float64)
    sy   = slopes[N*N:].reshape(N, N).astype(np.float64)

    n_unknowns = N * N
    n_x_eqs    = (N - 1) * N
    n_y_eqs    = N * (N - 1)
    n_eqs      = n_x_eqs + n_y_eqs + 1

    A   = np.zeros((n_eqs, n_unknowns), dtype=np.float64)
    b   = np.zeros(n_eqs,              dtype=np.float64)
    idx = 0

    # X-slope equations
    for i in range(N):
        for j in range(N - 1):
            A[idx, i*N + j+1] =  1.0
            A[idx, i*N + j  ] = -1.0
            b[idx]             = sx[i, j] * step
            idx += 1

    # Y-slope equations
    for i in range(N - 1):
        for j in range(N):
            A[idx, (i+1)*N + j] =  1.0
            A[idx,  i   *N + j] = -1.0
            b[idx]               = sy[i, j] * step
            idx += 1

    # Piston constraint (remove ambiguity)
    A[idx, :] = 1.0 / n_unknowns

    phi_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    phase_grid = phi_vec.reshape(N, N).astype(np.float32)

    # Convert rad*m to nm
    phase_grid = phase_grid * (500.0 / (2 * np.pi))

    return phase_grid


def direct_integrate(slopes, n_lenslets=N_LENSLETS,
                      D_aperture=D, grid_size=GRID_SIZE):
    from PIL import Image

    N    = n_lenslets
    step = D_aperture / N
    sx   = slopes[:N*N].reshape(N, N).astype(np.float64)
    sy   = slopes[N*N:].reshape(N, N).astype(np.float64)

    phi_x = np.cumsum(sx, axis=1) * step
    phi_y = np.cumsum(sy, axis=0) * step
    phi   = (phi_x + phi_y) / 2.0
    phi  -= phi.mean()

    phi_full = np.array(
        Image.fromarray(phi.astype(np.float32)).resize(
            (grid_size, grid_size), Image.BILINEAR))

    return (phi_full * (500.0 / (2 * np.pi))).astype(np.float32)


def benchmark_all_methods(slopes_list, true_phases_list, D_pinv,
                           n_lenslets=N_LENSLETS, grid_size=GRID_SIZE,
                           n_modes=N_ZERNIKE_MODES):
    from PIL import Image
    import time

    methods = {
        'modal' : lambda s: modal_reconstruct(s, D_pinv, grid_size, n_modes)[0],
        'zonal' : lambda s: np.array(
                      Image.fromarray(zonal_reconstruct(s, n_lenslets)).resize(
                          (grid_size, grid_size), Image.BILINEAR)),
        'direct': lambda s: direct_integrate(s, n_lenslets, D, grid_size),
    }

    results = {m: {'rms_nm': [], 'strehl': [], 'time_ms': []}
               for m in methods}

    for slopes, true_phase in zip(slopes_list, true_phases_list):
        valid = ~np.isnan(true_phase)
        for name, func in methods.items():
            t0   = time.perf_counter()
            pred = func(slopes)
            ms   = (time.perf_counter() - t0) * 1000
            rms  = float(np.sqrt(np.mean((pred[valid]-true_phase[valid])**2)))
            strehl = float(np.exp(-(2*np.pi*rms/500)**2))
            results[name]['rms_nm'].append(rms)
            results[name]['strehl'].append(strehl)
            results[name]['time_ms'].append(ms)

    # Average over all samples
    for name in results:
        for key in results[name]:
            results[name][key] = float(np.mean(results[name][key]))

    return results


def compute_slope_curl(slopes, n_lenslets=N_LENSLETS):
    N  = n_lenslets
    sx = slopes[:N*N].reshape(N, N).astype(np.float64)
    sy = slopes[N*N:].reshape(N, N).astype(np.float64)

    dsy_dx = sy[:, 1:] - sy[:, :-1]
    dsx_dy = sx[1:, :] - sx[:-1, :]
    curl   = dsy_dx[:-1, :] - dsx_dy[:, :-1]

    return curl.astype(np.float32)


if __name__ == '__main__':
    print("Testing classical.py...")
    slopes = np.random.randn(200).astype(np.float32) * 0.001
    D_p    = np.random.randn(36, 200).astype(np.float32) * 0.01
    pm, c  = modal_reconstruct(slopes, D_p, grid_size=64, n_modes=36)
    print(f"Modal output: {pm.shape}, coeffs: {c.shape}")
    pz     = zonal_reconstruct(slopes, n_lenslets=10)
    print(f"Zonal output: {pz.shape}")
    pd     = direct_integrate(slopes, n_lenslets=10, grid_size=64)
    print(f"Direct output: {pd.shape}")
    curl   = compute_slope_curl(slopes, n_lenslets=10)
    print(f"Curl map: {curl.shape}")
    print("classical.py OK")

