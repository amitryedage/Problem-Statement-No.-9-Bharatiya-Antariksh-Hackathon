import os

#  Telescope / Beam 
D                = 2.0       # Pupil diameter (metres)
WAVELENGTH       = 500e-9    # Observing wavelength (metres)

#  MLA Parameters (ISRO provides these)
N_LENSLETS       = 10        # Lenslets per side (10x10 = 100 subapertures)
MLA_FOCAL_LENGTH = 5e-3      # Focal length per lenslet (metres)
PIXEL_SIZE       = 10e-6     # Camera pixel size (metres)
GRID_SIZE        = 128       # Simulation grid resolution

# Derived Parameters
N_SUBAPERTURES   = N_LENSLETS ** 2
N_SLOPES         = N_SUBAPERTURES * 2
N_ZERNIKE_MODES  = 36

#  Turbulence Conditions for Training 
R0_TRAIN         = [0.03, 0.05, 0.10, 0.15, 0.20]
L0               = 100.0
WIND_SPEED       = 10.0

# Camera / Timing 
FRAME_DT_MS      = 2.0
CAMERA_FPS       = 500

#  Training Hyperparameter
BATCH_SIZE       = 32
LEARNING_RATE    = 1e-3
N_EPOCHS         = 100
TRAIN_SAMPLES    = 10000     # Per turbulence level
DEVICE           = 'cuda'

# Speed Target (Excution time must be lower than 10 ms )
MAX_PIPELINE_MS  = 10.0
# Paths
ROOT_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(ROOT_DIR, 'data')
RAW_DIR          = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR    = os.path.join(DATA_DIR, 'processed')
CHECKPOINT_DIR   = os.path.join(DATA_DIR, 'checkpoints')
OUTPUT_DIR       = os.path.join(ROOT_DIR, 'outputs')
FIGURES_DIR      = os.path.join(OUTPUT_DIR, 'figures')
RESULTS_DIR      = os.path.join(OUTPUT_DIR, 'results')
BENCHMARKS_DIR   = os.path.join(OUTPUT_DIR, 'benchmarks')

