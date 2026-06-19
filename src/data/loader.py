import numpy as np
import cv2
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from config import FRAME_DT_MS, PIXEL_SIZE, N_LENSLETS


def load_shwfs_sequence(bmp_dir,
                         frame_dt_ms=FRAME_DT_MS,
                         pixel_size_um=PIXEL_SIZE * 1e6,
                         n_lenslets=N_LENSLETS):
    if not os.path.isdir(bmp_dir):
        raise NotADirectoryError(f"Directory not found: {bmp_dir}")

    bmp_files = sorted([
        f for f in os.listdir(bmp_dir)
        if f.lower().endswith('.bmp')
    ])

    if len(bmp_files) == 0:
        raise FileNotFoundError(f"No .bmp files found in: {bmp_dir}")

    frames_list = []
    skipped     = 0
    for fname in bmp_files:
        path  = os.path.join(bmp_dir, fname)
        frame = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if frame is not None:
            frames_list.append(frame.astype(np.float32))
        else:
            skipped += 1
            print(f"  Warning: could not read {fname}")

    if len(frames_list) == 0:
        raise ValueError("No valid BMP frames could be loaded")

    frames       = np.array(frames_list)          # (N, H, W)
    N            = len(frames)
    timestamps   = np.arange(N, dtype=np.float32) * frame_dt_ms

    metadata = {
        'n_frames'          : N,
        'n_skipped'         : skipped,
        'frame_shape'       : tuple(frames[0].shape),
        'frame_dt_ms'       : float(frame_dt_ms),
        'pixel_size_um'     : float(pixel_size_um),
        'n_lenslets'        : int(n_lenslets),
        'n_subapertures'    : int(n_lenslets ** 2),
        'total_duration_ms' : float(timestamps[-1]),
        'framerate_hz'      : float(1000.0 / frame_dt_ms),
        'bmp_dir'           : bmp_dir,
    }

    return frames, timestamps, metadata


def preprocess_frame(frame, normalize=True):
    
    # Background subtraction: subtract minimum
    proc = frame - frame.min()

    # Clip hot pixels (above 99.9th percentile)
    clip_val = np.percentile(proc, 99.9)
    proc     = np.clip(proc, 0, clip_val)

    if normalize and proc.max() > 0:
        proc = proc / proc.max()

    return proc.astype(np.float32)


def save_frames_as_bmp(frames, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        path = os.path.join(output_dir, f'frame_{i:04d}.bmp')
        cv2.imwrite(path, frame.astype(np.uint8))
    print(f"  Saved {len(frames)} BMP files to {output_dir}")


if __name__ == '__main__':
    print("loader.py ready")
    print("Usage:")
    print("  frames, ts, meta = load_shwfs_sequence('data/raw/isro_bmps/')")
    print("  When ISRO provides BMP files, just change the path above.")
