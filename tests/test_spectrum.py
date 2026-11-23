"""Frequency analysis of a profile (affine_ransac.spectrum, D173), on synthetic profiles: it checks the code, not
what the real profiles hold."""

import numpy as np
import pytest

from affine_ransac import spectrum


def uneven_rows(spacing_nm: float, length_nm: float, seed: int = 0) -> np.ndarray:
    """Rows every spacing_nm along length_nm, a tenth of them missing (lines or spaces not found)."""
    rows = np.arange(0.0, length_nm, spacing_nm)
    keep = np.random.default_rng(seed).random(len(rows)) > 0.1
    return rows[keep] + 123_456.0  # away from zero, as mask positions are


def test_a_sine_on_uneven_rows_gives_its_period_and_amplitude():
    position = uneven_rows(50.0, 40_000.0)
    rng = np.random.default_rng(1)
    values = 0.8 * np.sin(2 * np.pi * position / 412.0 + 0.3) + rng.normal(0, 0.05, len(position)) + 2.0
    frequency = spectrum.frequency_grid(position)
    amplitude = spectrum.amplitude_spectrum(position, values, frequency)

    assert frequency[0] == spectrum.resolution(position) and frequency[-1] <= spectrum.limit(position)
    np.testing.assert_allclose(np.diff(frequency), spectrum.resolution(position) / spectrum.OVERSAMPLE)
    assert spectrum.row_spacing(position) == 50.0 and spectrum.limit(position) == 0.01
    peak = spectrum.peak_near(position, values, frequency, amplitude, 1 / 400.0, 1e-4)  # a click beside the peak
    assert abs(peak - 1 / 412.0) < spectrum.resolution(position) / 10
    assert abs(amplitude.max() - 0.8) < 0.05
    fitted, curve = spectrum.fit_sines(position, values, [peak])
    assert abs(fitted[0] - 0.8) < 0.02
    assert np.std(values - curve) < 0.06  # what is left: the mean and the noise


def test_a_period_shorter_than_two_row_spacings_shows_folded_and_says_where_from():
    """Rows every 100 nm cannot show a 120 nm period: it shows at 1/100 − 1/120 cycles per nm (a 600 nm period), and
    folded_from lists 120 nm among the periods it may come from."""
    position = np.arange(0.0, 30_000.0, 100.0)
    values = 0.5 * np.cos(2 * np.pi * position / 120.0)
    frequency = spectrum.frequency_grid(position)
    amplitude = spectrum.amplitude_spectrum(position, values, frequency)

    shown = spectrum.peak_near(position, values, frequency, amplitude, frequency[np.argmax(amplitude)], 1e-5)
    assert abs(1 / shown - 600.0) < 2.0
    candidates = spectrum.folded_from(shown, 100.0)
    assert len(candidates) == 4 and np.all(np.diff(candidates) > 0)
    assert np.min(np.abs(1 / candidates - 120.0)) < 0.5


def test_two_close_sines_are_fitted_together_and_leave_the_noise():
    """A beat: two sines whose periods differ by 10 %. Fitted together, each keeps its own amplitude."""
    position = uneven_rows(40.0, 60_000.0, seed=2)
    rng = np.random.default_rng(3)
    values = (0.5 * np.sin(2 * np.pi * position / 400.0) + 0.3 * np.sin(2 * np.pi * position / 440.0 + 1.0)
              + rng.normal(0, 0.05, len(position)))
    amplitude, curve = spectrum.fit_sines(position, values, [1 / 400.0, 1 / 440.0])
    np.testing.assert_allclose(amplitude, [0.5, 0.3], atol=0.01)
    assert np.std(values - curve) < 0.055
    none, nothing = spectrum.fit_sines(position, values, [])
    assert len(none) == 0 and not nothing.any()


def test_the_spectrum_is_the_same_computed_a_few_frequencies_at_a_time(monkeypatch):
    position = uneven_rows(50.0, 10_000.0)
    values = np.sin(2 * np.pi * position / 300.0) + np.random.default_rng(4).normal(0, 0.1, len(position))
    frequency = spectrum.frequency_grid(position)
    whole = spectrum.amplitude_spectrum(position, values, frequency)
    monkeypatch.setattr(spectrum, "CHUNK_CELLS", 3 * len(position) + 1)  # three frequencies a call
    np.testing.assert_allclose(spectrum.amplitude_spectrum(position, values, frequency), whole, rtol=1e-12)
    assert spectrum.amplitude_spectrum(position, values, frequency[5]).shape == ()


def test_the_running_median_follows_a_ramp_and_keeps_a_step_sharp():
    position = np.arange(0.0, 5000.0, 50.0)
    ramp = 0.002 * position
    inside = (position >= 500) & (position <= 4450)  # a whole window around them
    np.testing.assert_allclose(spectrum.running_median(position, ramp, 1000.0)[inside], ramp[inside])
    step = (position > 2525).astype(float)
    np.testing.assert_array_equal(spectrum.running_median(position, step, 1000.0), step)


def test_removing_the_slow_part_leaves_the_sine_as_the_highest_peak():
    """A sawtooth with a small sine on it: the profile's highest peak is the sawtooth's; once the running median is
    removed, the sine's."""
    position = np.arange(0.0, 39_000.0, 50.0)
    values = 1.5 * ((position % 13_000) / 13_000 - 0.5) + 0.3 * np.sin(2 * np.pi * position / 410.0)
    frequency = spectrum.frequency_grid(position)
    profile = spectrum.amplitude_spectrum(position, values, frequency)
    remainder = values - spectrum.running_median(position, values, 2050.0)  # five periods of the sine
    left = spectrum.amplitude_spectrum(position, remainder, frequency)
    assert abs(1 / frequency[np.argmax(profile)] - 13_000) < 1000
    assert abs(1 / frequency[np.argmax(left)] - 410) < 2 and left.max() > 0.25


def test_the_fft_shows_a_sine_on_its_grid_with_its_amplitude_and_bridges_gaps():
    position = np.arange(0.0, 51_200.0, 50.0)  # 1024 rows
    values = 0.4 * np.sin(2 * np.pi * position * 100 / 51_200.0) + 3.0  # 100 periods: on the FFT's grid
    frequency, amplitude = spectrum.fft_spectrum(position, values)
    assert frequency[0] == pytest.approx(1 / 51_200.0) and frequency[-1] == pytest.approx(spectrum.limit(position))
    assert frequency[np.argmax(amplitude)] == pytest.approx(100 / 51_200.0)
    assert amplitude.max() == pytest.approx(0.4, rel=1e-3)
    rows = uneven_rows(50.0, 51_200.0)
    frequency, amplitude = spectrum.fft_spectrum(rows, 0.4 * np.sin(2 * np.pi * rows / 512.0))
    assert abs(1 / frequency[np.argmax(amplitude)] - 512.0) < 6


def test_a_straight_line_per_ramp_takes_a_sawtooth_out_and_leaves_the_sine():
    position = np.arange(0.0, 39_000.0, 50.0)
    sawtooth = 1.5 * ((position % 13_000) / 13_000 - 0.5)
    np.testing.assert_allclose(spectrum.line_per_ramp(position, sawtooth, 0.5), sawtooth, atol=1e-12)
    sine = 0.1 * np.sin(2 * np.pi * position / 410.0)
    left = sawtooth + sine - spectrum.line_per_ramp(position, sawtooth + sine, 0.5)
    amplitude, curve = spectrum.fit_sines(position, left, [1 / 410.0])
    assert amplitude[0] == pytest.approx(0.1, abs=0.005)
    assert np.abs(left - curve).max() < 0.01  # no row left on a drop's line
