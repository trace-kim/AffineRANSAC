"""Frequency analysis of a registration profile (docs/SPEC.md D173): the amplitude of the error at each spatial
frequency, for positions that need not be evenly spaced (a line or space not found, mixed pitches, a stripe left out).

Positions and values in nm, frequencies in cycles per nm (the window shows cycles per µm).
"""

import numpy as np
from scipy.signal import lombscargle

OVERSAMPLE = 5  # frequency grid steps per resolution step (one over the profile's length)
CHUNK_CELLS = 2_000_000  # points x frequencies per call of scipy's periodogram (16 MB per table it builds)


def row_spacing(position_nm: np.ndarray) -> float:
    """Median distance between neighbouring positions (nm)."""
    steps = np.diff(np.sort(position_nm))
    steps = steps[steps > 0]
    return float(np.median(steps)) if len(steps) else 0.0


def resolution(position_nm: np.ndarray) -> float:
    """One over the profile's length (cycles per nm): two frequencies closer than this are not told apart."""
    return 1.0 / float(np.ptp(position_nm))


def limit(position_nm: np.ndarray) -> float:
    """One over twice the median row spacing (cycles per nm): a higher frequency shows folded below it."""
    return 0.5 / row_spacing(position_nm)


def frequency_grid(position_nm: np.ndarray) -> np.ndarray:
    """Frequencies (cycles per nm) from the resolution up to the limit, OVERSAMPLE steps per resolution step."""
    step = resolution(position_nm) / OVERSAMPLE
    return np.arange(resolution(position_nm), limit(position_nm) + step / 2, step)


def amplitude_spectrum(position_nm: np.ndarray, values: np.ndarray, frequency: np.ndarray) -> np.ndarray:
    """Amplitude (nm) at each frequency (cycles per nm) of the least-squares sine with a free mean: the Lomb–Scargle
    periodogram (scipy), which needs no even spacing."""
    x = np.asarray(position_nm, float)
    x, y, angular = x - x.min(), np.asarray(values, float), 2 * np.pi * np.ravel(frequency).astype(float)
    # scipy holds several (points x frequencies) tables at once: a few frequencies at a time keeps them small.
    chunk = max(1, CHUNK_CELLS // len(x))
    amplitude = np.concatenate([np.abs(np.atleast_1d(lombscargle(x, y, angular[k:k + chunk], normalize="amplitude",
                                                                 floating_mean=True)))
                                for k in range(0, len(angular), chunk)] or [np.empty(0)])
    return amplitude.reshape(np.shape(frequency))


def fit_sines(position_nm: np.ndarray, values: np.ndarray, frequencies) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit of a mean plus one sine at each frequency (cycles per nm), all together.

    Returns the sines' amplitudes (nm) and their sum at the positions, without the mean.
    """
    x = np.asarray(position_nm, float) - np.min(position_nm)
    angles = 2 * np.pi * np.outer(x, np.asarray(frequencies, float))
    design = np.column_stack([np.ones_like(x), np.cos(angles), np.sin(angles)])
    coefficients = np.linalg.lstsq(design, np.asarray(values, float), rcond=None)[0]
    count = len(frequencies)
    amplitude = np.hypot(coefficients[1:1 + count], coefficients[1 + count:])
    return amplitude, design[:, 1:] @ coefficients[1:]


def peak_near(position_nm, values, frequency: np.ndarray, amplitude: np.ndarray, near: float, within: float) -> float:
    """The top of the peak nearest to a click: the highest point of a spectrum (amplitude at frequency, cycles per nm)
    within `within` of near, then uphill to the peak's top, refined on a grid 10 times finer."""
    inside = np.flatnonzero(np.abs(frequency - near) <= within)
    if len(inside) == 0:
        inside = np.array([np.argmin(np.abs(frequency - near))])
    k = inside[np.argmax(amplitude[inside])]
    while k > 0 and amplitude[k - 1] > amplitude[k]:
        k -= 1
    while k < len(amplitude) - 1 and amplitude[k + 1] > amplitude[k]:
        k += 1
    step = frequency[1] - frequency[0] if len(frequency) > 1 else resolution(position_nm) / OVERSAMPLE
    fine = np.linspace(frequency[k] - step, frequency[k] + step, 21)
    fine = fine[fine > 0]
    return float(fine[np.argmax(amplitude_spectrum(position_nm, values, fine))])


def running_median(position_nm: np.ndarray, values: np.ndarray, window_nm: float) -> np.ndarray:
    """A profile's slow part: at each position (from the lowest up) the median of the values within half a window of
    it. It follows a ramp and keeps a step sharp; near a step on a ramp it lags by up to half a window of the ramp. A
    sine whose period fits in the window a few times hardly moves it; a shorter window follows part of the sine."""
    low = np.searchsorted(position_nm, position_nm - window_nm / 2, side="left")
    high = np.searchsorted(position_nm, position_nm + window_nm / 2, side="right")
    return np.array([np.median(values[a:b]) for a, b in zip(low, high)])


def folded_from(frequency: float, spacing_nm: float, multiples: int = 2) -> np.ndarray:
    """The higher frequencies (cycles per nm) that rows every spacing_nm show at this frequency, from the lowest:
    k / spacing ± frequency for k = 1 ... multiples."""
    rate = np.arange(1, multiples + 1) / spacing_nm
    return np.sort(np.concatenate([rate - frequency, rate + frequency]))
