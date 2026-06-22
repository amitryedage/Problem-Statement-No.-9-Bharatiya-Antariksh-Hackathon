"""


USAGE (on ISRO's real BMP data):
    from src.pipeline import PS09Pipeline
    pipeline = PS09Pipeline.load('data/checkpoints/')
    results  = pipeline.run('data/raw/isro_bmps/')

ALL 5 ISRO OUTPUTS PRODUCED:
    results['phase_maps']      # W(xi,yi) per frame        [Output 1]
    results['zernike_coeffs']  # Zernike coeffs per frame  [Output 2]
    results['r0_cm']           # Fried parameter (cm)      [Output 3]
    results['tau0_ms']         # Coherence time (ms)       [Output 4]
    results['actuator_maps']   # A(xi,yi) per frame        [Output 5]
    results['timing']          # Per-module timing (ms)    [benchmark]
    results['wind_speed_ms']   # Wind speed (m/s)          [bonus]
    results['r0_series']       # r0 per sliding window     [bonus]

SPEED: < 10ms per frame total (ISRO Evaluation Criterion )
"""

