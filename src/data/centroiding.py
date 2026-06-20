import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import N_LENSLETS, MLA_FOCAL_LENGTH, PIXEL_SIZE


def extract_subapertures(frame, n_lenslets=N_LENSLETS):
    H, W    = frame.shape
    patch_h = H // n_lenslets
    patch_w = W // n_lenslets
    patches = []
    grid_pos= []

    for i in range(n_lenslets):
        for j in range(n_lenslets):
            y0 = i * patch_h
            x0 = j * patch_w
            patches.append(frame[y0:y0+patch_h, x0:x0+patch_w])
            grid_pos.append([y0, x0])

    return (np.array(patches, dtype=np.float32),
            patch_h, patch_w,
            np.array(grid_pos))


def centroid_com(patch):
    h, w    = patch.shape
    y_grid  = np.arange(h, dtype=np.float32)
    x_grid  = np.arange(w, dtype=np.float32)

    patch_bg = np.maximum(patch - patch.min(), 0)
    total    = patch_bg.sum() + 1e-12

    cx = float((x_grid[np.newaxis, :] * patch_bg).sum() / total)
    cy = float((y_grid[:, np.newaxis] * patch_bg).sum() / total)
    return cx, cy


def centroid_all_patches(patches):
    patches, _, _, _ = extract_subapertures(
        flat_frame.astype(np.float32), n_lenslets)
    return centroid_all_patches(patches)


def extract_slopes_from_frame(frame, reference_centroids,
                               n_lenslets=N_LENSLETS,
                               focal_length=MLA_FOCAL_LENGTH,
                               pixel_size=PIXEL_SIZE):
    patches, _, _, _ = extract_subapertures(
        frame.astype(np.float32), n_lenslets)
    centroids = centroid_all_patches(patches)

    delta_x = centroids[:, 0] - reference_centroids[:, 0]
    delta_y = centroids[:, 1] - reference_centroids[:, 1]

    slope_x = (delta_x * pixel_size / focal_length).astype(np.float32)
    slope_y = (delta_y * pixel_size / focal_length).astype(np.float32)

    return np.concatenate([slope_x, slope_y])


def extract_slopes_batch(frames, reference_centroids,
                          n_lenslets=N_LENSLETS,
                          focal_length=MLA_FOCAL_LENGTH,
                          pixel_size=PIXEL_SIZE):
    N       = len(frames)
    N_sub   = n_lenslets ** 2
    results = np.zeros((N, N_sub * 2), dtype=np.float32)
    for i, frame in enumerate(frames):
        results[i] = extract_slopes_from_frame(
            frame, reference_centroids,
            n_lenslets, focal_length, pixel_size)
    return results


if __name__ == '__main__':
    print("Testing centroiding.py...")
    dummy = np.random.rand(128, 128).astype(np.float32)
    patches, ph, pw, _ = extract_subapertures(dummy, 10)
    print(f"  Patches: {patches.shape}")
    cx, cy = centroid_com(patches[0])
    print(f"  Centroid: ({cx:.2f}, {cy:.2f})")
    ref = compute_reference_centroids(dummy, 10)
    slopes = extract_slopes_from_frame(dummy, ref, 10)
    print(f"  Slopes shape: {slopes.shape}")
    print(" centroiding.py OK")
