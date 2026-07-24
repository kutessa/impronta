"""WADA-SNR estimator behavior and the progressive-relaxation filter.

WADA is an estimator — absolute assertions are loose bands; the strict
assertion is monotonicity in the true mixed SNR.
"""

import numpy as np
import pytest
from conftest import make_voice_audio

from impronta import ImprontaConfig
from impronta.snr import filter_by_snr, wada_snr

CFG = ImprontaConfig()


def test_clean_speechlike_reads_high():
    assert wada_snr(make_voice_audio(3.0, 440.0)) > 40


def test_heavy_noise_reads_low():
    assert wada_snr(make_voice_audio(3.0, 440.0, noise_db=0.0)) < 12


def test_monotone_in_true_snr():
    estimates = [
        wada_snr(make_voice_audio(3.0, 440.0, noise_db=db)) for db in (0, 10, 20, 30, 40)
    ]
    assert estimates == sorted(estimates)


@pytest.mark.parametrize("true_db", [10, 20, 30])
def test_loose_absolute_accuracy(true_db):
    # WADA is an estimator and our synthetic voice is "more gamma than speech";
    # accuracy is asserted only as a wide band — monotonicity is the strict test.
    est = wada_snr(make_voice_audio(4.0, 440.0, noise_db=true_db))
    assert abs(est - true_db) < 8.0


def test_silence_and_empty_return_floor():
    assert wada_snr(np.zeros(16_000, dtype=np.float32)) == -20.0
    assert wada_snr(np.array([], dtype=np.float32)) == -20.0


def test_pure_gaussian_noise_reads_floorish():
    rng = np.random.default_rng(0)
    assert wada_snr(rng.normal(size=48_000)) < 0


def test_output_clamped_to_table_range():
    x = make_voice_audio(3.0, 440.0)
    assert -20.0 <= wada_snr(x) <= 100.0


# -- progressive relaxation -------------------------------------------------


def test_first_tier_wins():
    kept, tier = filter_by_snr([25.0, 8.0, 22.0], CFG)
    assert (kept, tier) == ([0, 2], "high")


def test_relaxes_to_medium_then_low():
    assert filter_by_snr([17.0, 16.0], CFG) == ([0, 1], "medium")
    assert filter_by_snr([13.0], CFG) == ([0], "low")


def test_all_fail_returns_empty_and_none():
    assert filter_by_snr([5.0, 11.9], CFG) == ([], None)


def test_empty_input():
    assert filter_by_snr([], CFG) == ([], None)
