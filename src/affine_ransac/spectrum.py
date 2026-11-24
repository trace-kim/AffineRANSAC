"""Frequency analysis of a registration profile (docs/SPEC.md D173): the amplitude of the error at each spatial
frequency, for positions that need not be evenly spaced (a line or space not found, mixed pitches, a stripe left out).

Positions and values in nm, frequencies in cycles per nm (the window shows cycles per µm).
"""

import numpy as np
from scipy.signal import lombscargle

from affine_ransac.registration import turning_points

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
    """Frequencies (cycles per nm) from the resolution up to the limit, OVERSAMPLE steps per resolution step, without
    those closer than the resolution to a multiple of the limit (D178): on evenly spaced rows a sine there is zero at
    every row (or, at twice the limit, the same at every row), so its least-squares amplitude is the noise divided by
    almost nothing. The grid starts one resolution above zero for the same reason."""
    step, top = resolution(position_nm) / OVERSAMPLE, limit(position_nm)
    grid = np.arange(resolution(position_nm), top + step / 2, step)
    return grid[np.abs(grid - top * np.round(grid / top)) >= resolution(position_nm)]


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


def fft_spectrum(position_nm: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude spectrum by FFT, for comparison with the least-squares one (D175): the profile (positions from the
    lowest up) resampled every median row spacing by straight lines between its points (across a gap too), its mean
    removed, a Hann window. Returns the frequencies (cycles per nm, from the first above zero to the limit) and the
    amplitudes (nm: 2 |X| / sum of the window, so a sine on a frequency of the grid shows its amplitude)."""
    spacing = row_spacing(position_nm)
    grid = np.arange(position_nm[0], position_nm[-1] + spacing / 2, spacing)
    resampled = np.interp(grid, position_nm, values)
    window = np.hanning(len(grid))
    transform = np.fft.rfft((resampled - resampled.mean()) * window)
    return np.fft.rfftfreq(len(grid), spacing)[1:], 2 * np.abs(transform[1:]) / window.sum()


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


def line_per_ramp(position_nm: np.ndarray, values: np.ndarray, height: float) -> np.ndarray:
    """A profile's slow part as straight lines (D175): the profile (positions from the lowest up) split into ramps
    where it turns by more than height (registration.turning_points, as the tab's ramp range), a least-squares line
    through each ramp's points, its ends included. Each point takes its ramp's line (a turn, the longer ramp's of its
    two). A turn is found at the highest (lowest) point, which a wiggle can put a few rows from a sawtooth's drop, so
    the short ramp of a drop holds rows of the slopes beside it: a point of a ramp shorter than a neighbour takes that
    neighbour's line where it passes within height of the point (the nearer, if both do)."""
    values = np.asarray(values, dtype=float)
    x = np.asarray(position_nm, dtype=float) - position_nm[0]
    turns = turning_points(values, height)
    ramps = list(zip(turns[:-1], turns[1:]))
    length = [b - a for a, b in ramps]
    lines = [np.polyfit(x[a:b + 1], values[a:b + 1], 1) for a, b in ramps]
    owner = np.empty(len(values), int)  # the ramp each point belongs to
    for k, (a, b) in enumerate(ramps):
        owner[a:b + 1] = k
    for k in range(1, len(ramps)):  # a turn, shared by two ramps
        owner[ramps[k][0]] = k if length[k] > length[k - 1] else k - 1
    slow = np.empty_like(values)
    for k in range(len(ramps)):
        slow[owner == k] = np.polyval(lines[k], x[owner == k])
    for k in range(len(ramps)):
        mine = np.flatnonzero(owner == k)
        nearest = np.full(len(mine), np.inf)
        for j in (k - 1, k + 1):
            if 0 <= j < len(ramps) and length[j] > length[k]:
                fitted = np.polyval(lines[j], x[mine])
                miss = np.abs(values[mine] - fitted)
                take = (miss <= height) & (miss < nearest)
                slow[mine[take]], nearest[take] = fitted[take], miss[take]
    return slow


def folded_from(frequency: float, spacing_nm: float, multiples: int = 2) -> np.ndarray:
    """The higher frequencies (cycles per nm) that rows every spacing_nm show at this frequency, from the lowest:
    k / spacing ± frequency for k = 1 ... multiples."""
    rate = np.arange(1, multiples + 1) / spacing_nm
    return np.sort(np.concatenate([rate - frequency, rate + frequency]))
