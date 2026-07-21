"""Blind SNR estimation (WADA-SNR) and the progressive-relaxation filter.

WADA-SNR (C. Kim and R. M. Stern, "Robust signal-to-noise ratio estimation
based on waveform amplitude distribution analysis," Proceedings of
Interspeech 2008, pp. 2598-2601, doi: 10.21437/Interspeech.2008-644)
estimates the SNR of a speech signal with no clean reference by assuming
speech amplitudes follow a Gamma distribution (shape ~0.4) and noise is
Gaussian. The estimate comes from the gap between log-of-mean and
mean-of-log magnitudes, looked up in a precomputed table.

The lookup table below is the published dVal curve as transcribed in the
LabROSA snreval MATLAB package (https://labrosa.ee.columbia.edu/projects/snreval/)
and John Meade's NumPy port.
"""

from __future__ import annotations

import numpy as np

from .config import ImprontaConfig

_EPS = 1e-10

_DB_VALS = np.arange(-20, 101)

# fmt: off
_G_VALS = np.array([
    0.40974774, 0.40986926, 0.40998566, 0.40969089, 0.40986186, 0.40999006,
    0.41027138, 0.41052627, 0.41101024, 0.41143264, 0.41231718, 0.41337272,
    0.41526426, 0.4178192, 0.42077252, 0.42452799, 0.42918886, 0.43510373,
    0.44234195, 0.45161485, 0.46221153, 0.47491647, 0.48883809, 0.50509236,
    0.52353709, 0.54372088, 0.56532427, 0.58847532, 0.61346212, 0.63954496,
    0.66750818, 0.69583724, 0.72454762, 0.75414799, 0.78323148, 0.81240985,
    0.84219775, 0.87166406, 0.90030504, 0.92880418, 0.95655449, 0.9835349,
    1.01047155, 1.0362095, 1.06136425, 1.08579312, 1.1094819, 1.13277995,
    1.15472826, 1.17627308, 1.19703503, 1.21671694, 1.23535898, 1.25364313,
    1.27103891, 1.28718029, 1.30302865, 1.31839527, 1.33294817, 1.34700935,
    1.3605727, 1.37345513, 1.38577122, 1.39733504, 1.40856397, 1.41959619,
    1.42983624, 1.43958467, 1.44902176, 1.45804831, 1.46669568, 1.47486938,
    1.48269965, 1.49034339, 1.49748214, 1.50435106, 1.51076426, 1.51698915,
    1.5229097, 1.528578, 1.53389835, 1.5391211, 1.5439065, 1.54858517,
    1.55310776, 1.55744391, 1.56164927, 1.56566348, 1.56938671, 1.57307767,
    1.57654764, 1.57980083, 1.58304129, 1.58602496, 1.58880681, 1.59162477,
    1.5941969, 1.59693155, 1.599446, 1.60185011, 1.60408668, 1.60627134,
    1.60826199, 1.61004547, 1.61192472, 1.61369656, 1.61534074, 1.61688905,
    1.61838916, 1.61985374, 1.62135878, 1.62268119, 1.62390423, 1.62513143,
    1.62632463, 1.6274027, 1.62842767, 1.62945532, 1.6303307, 1.63128026,
    1.63204102,
])
# fmt: on


def wada_snr(x: np.ndarray) -> float:
    """Estimate the SNR (dB) of a speech signal without a clean reference.

    Returns a value clamped to [-20, 100]. Silent or empty input returns the
    floor (-20.0). This is an *estimator* — treat the output as a quality
    band, not a measurement.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float(_DB_VALS[0])
    peak = np.abs(x).max()
    if peak < _EPS:
        return float(_DB_VALS[0])
    abs_x = np.abs(x / peak)
    abs_x[abs_x < _EPS] = _EPS

    v1 = max(_EPS, float(abs_x.mean()))  # E[|z|]
    v2 = float(np.log(abs_x).mean())  # E[log|z|]
    v3 = np.log(v1) - v2  # log(E[|z|]) - E[log|z|]

    below = np.where(v3 > _G_VALS)[0]
    if below.size == 0:
        return float(_DB_VALS[0])
    idx = int(below.max())
    if idx == len(_DB_VALS) - 1:
        return float(_DB_VALS[-1])
    frac = (v3 - _G_VALS[idx]) / (_G_VALS[idx + 1] - _G_VALS[idx])
    return float(_DB_VALS[idx] + frac * (_DB_VALS[idx + 1] - _DB_VALS[idx]))


def filter_by_snr(
    snrs: list[float], cfg: ImprontaConfig
) -> tuple[list[int], str | None]:
    """Progressive relaxation over ``cfg.snr_tiers``.

    Returns (indices of segments passing the first non-empty tier, tier
    label), or ([], None) when nothing passes even the lowest rung — that
    speaker is unusable.
    """
    for threshold, tier in cfg.snr_tiers:
        kept = [i for i, s in enumerate(snrs) if s >= threshold]
        if kept:
            return kept, tier
    return [], None
