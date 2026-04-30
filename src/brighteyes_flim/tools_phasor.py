"""Phasor utilities, lifetime estimators, and plotting helpers."""

import math
import os

import h5py
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb
from matplotlib.pyplot import gca
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.ndimage import median_filter, shift
from scipy.signal import savgol_filter
from skimage.registration import phase_cross_correlation
from tqdm.auto import tqdm

try:
    import brighteyes_ism.analysis.Graph_lib as gra
except ImportError:  # pragma: no cover - optional dependency
    gra = None
try:
    import torch
    from torch.fft import fftn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    fftn = None

__all__ = [
    "correct_phasor",
    "sum_adjacent_pixel",
    "periodic_single_exponential_model",
    "pad_tensor",
    "torch_median_filter",
    "partial_convolution_fft",
    "richardson_lucy_deconvolution",
    "estimate_irf",
    "estimate_lifetime_from_birfi",
    "estimate_lifetime_from_log",
    "estimate_lifetime_from_circmean",
    "plot_universal_circle",
    "flatten_and_remove_nan",
    "g0s0_to_m_phi",
    "m_phi_to_g0s0",
    "phasor",
    "calculate_phasor",
    "calculate_tau_phi",
    "calculate_tau_m",
    "calculate_m_phi_tau_phi_tau_m",
    "calculate_phasor_on_img_ch",
    "phasor_h5",
    "calculate_phasor_on_img",
    "calculate_phasor_on_img_pixels",
    "plot_tau",
    "plot_funny_single_phasor",
    "plot_phasor",
    "fourier_shift",
    "cmap2d",
    "linear_shift",
    "correction_phasor",
    "threshold_phasor",
    "median_phasors",
    "calculate_irf_correction",
    "m_phi_to_complex",
    "find_shifts_irf_data",
    "find_temporal_shifts",
    "align_decays",
    "align_image",
    "calibrate_phasor",
    "show_lifetime_histogram",
    "plot_flim_image",
    "build_lifetime_equalizer",
    "apply_lifetime_equalizer",
    "equalized_lifetime_tick_values",
    "show_flim_equalized",
]

def correct_phasor(phasor_data, laser_phasor, coeff=1., laser_correction=True):
    if phasor_data is None:
        raise (ValueError("call before calculate_phasor_on_img_ch(...)"))

    if laser_correction:
        return phasor_data * coeff * np.exp(1j * (-np.angle(laser_phasor)))
    else:
        return phasor_data * coeff


def sum_adjacent_pixel(data, n=4):
    """Sum square pixel blocks while preserving the histogram axes."""
    if n <= 1:
        return data
    if len(data.shape) == 2:
        data = data[:, 0:((data.shape[0] // n) * n)]
        data = data[0:((data.shape[1] // n) * n), :]
        assert (data.shape[0] % n == 0)
        assert (data.shape[1] % n == 0)
        d = data[0::n, ::n]
        for i in range(1, n):
            d = d + data[i::n, i::n]
        return d
    elif len(data.shape) == 6:
        data = data[:, :, :, 0:((data.shape[2] // n) * n), :, :]
        data = data[:, :, 0:((data.shape[3] // n) * n), :, :, :]
        assert (data.shape[2] % n == 0)
        assert (data.shape[3] % n == 0)
        d = data[:, :, 0::n, 0::n, :, :]
        for i in range(1, n):
            d = d + data[:, :, i::n, i::n, :, :]
        return d


def _require_torch():
    from .alignment import Alignment
    Alignment._require_torch()


def periodic_single_exponential_model(t, C, tau, period):
    """
    Periodic single-exponential decay centered on the middle bin.

    Parameters
    ----------
    t : array-like
        Time axis.
    C : float
        Decay amplitude.
    tau : float
        Lifetime in the same time units as ``t`` and ``period``.
    period : float
        Excitation period in the same time units as ``t``.
    """

    t = np.asarray(t, dtype=float)
    offset = (t.max() - t.min()) / 2
    return C * (np.heaviside(t - offset, 1) + 1 / (np.exp(period / tau) - 1)) * np.exp((-t + offset) / tau)


def pad_tensor(x, pad_left: int, pad_right: int, dim: int, mode: str = "reflect"):
    """
    Pad a torch tensor along one dimension.
    """
    from .alignment import Alignment
    return Alignment.pad_tensor(x, pad_left, pad_right, dim, mode=mode)


def torch_median_filter(x, window_size=3, dims=None, mode="reflect"):
    """
    Apply a median filter to a torch tensor along selected dimensions.
    """
    from .alignment import Alignment
    return Alignment.median_filter(x, window_size=window_size, dims=dims, mode=mode)


def partial_convolution_fft(volume, kernel, dim1: str = "ijk", dim2: str = "jkl",
                            axis: str = "jk", fourier: tuple = (False, False)):
    """
    Convolution through FFT with einsum-based dimension bookkeeping.
    """
    from .alignment import Alignment
    return Alignment.partial_convolution_fft(
        volume,
        kernel,
        dim1=dim1,
        dim2=dim2,
        axis=axis,
        fourier=fourier,
    )


def richardson_lucy_deconvolution(ref_data, t, C_R, tau_R, T, iterations=30, eps=1e-8, regularization=3):
    """
    Estimate an IRF from a measured reference decay using Richardson-Lucy deconvolution.

    Parameters
    ----------
    ref_data : torch.Tensor or array-like
        Measured reference decay histogram.
    t : array-like
        Time axis.
    C_R : float
        Amplitude of the reference mono-exponential model.
    tau_R : float
        Lifetime of the reference sample in the same time units as ``t`` and ``T``.
    T : float
        Excitation period in the same time units as ``t``.
    iterations : int, optional
        Number of Richardson-Lucy iterations.
    eps : float, optional
        Numerical floor used to avoid divisions by zero.
    regularization : int, optional
        Odd median-filter window applied at each iteration. Set to ``1`` to disable.
    """

    _require_torch()

    if not torch.is_tensor(ref_data):
        ref_data = torch.as_tensor(ref_data)

    ref_data = ref_data.to(dtype=torch.float64)
    kernel = torch.as_tensor(
        periodic_single_exponential_model(t, C_R, tau_R, T),
        dtype=ref_data.dtype,
        device=ref_data.device,
    )
    kernel_t = kernel.flip(0)

    irf_est = torch.ones_like(ref_data)
    kernel_fft = fftn(kernel, dim=0)
    kernel_t_fft = fftn(kernel_t, dim=0)
    y = torch.clamp(ref_data, min=0)

    for _ in range(iterations):
        conv = partial_convolution_fft(irf_est, kernel_fft, dim1="t", dim2="t", axis="t", fourier=(False, True))
        conv = torch.clamp(conv, min=eps)
        relative_blur = y / conv
        correction = partial_convolution_fft(
            relative_blur, kernel_t_fft, dim1="t", dim2="t", axis="t", fourier=(False, True)
        )
        irf_est = torch.clamp(irf_est * correction, min=0)
        if regularization > 1:
            irf_est = torch_median_filter(irf_est, window_size=regularization, dims=[0], mode="replicate")

    return irf_est


def estimate_irf(ref_data, t, C_R, tau, period, iterations=300, eps=1e-8, regularization=3):
    """
    Convenience wrapper around :func:`richardson_lucy_deconvolution`.
    """

    return richardson_lucy_deconvolution(
        ref_data,
        t,
        C_R,
        tau,
        period,
        iterations=iterations,
        eps=eps,
        regularization=regularization,
    )


# ---------------------------------------------------------------------------
# Lifetime estimators
# ---------------------------------------------------------------------------
def estimate_lifetime_from_birfi(
    x,
    y,
    window_length=11,
    polyorder=3,
    persistence=5,
    threshold=0.05,
    axis=0,
    return_bounds=False,
):
    """
    Estimate fluorescence lifetime from the centroid of a detected decay window.
    (from birfi)

    The function merges decay-window detection and centroid lifetime estimation
    into a NumPy-based utility. The start index ``t0`` is defined as the global
    minimum of the Savitzky-Golay first derivative. The end index ``t1`` is the
    first point after ``t0`` where the derivative stays positive on average for
    ``persistence`` samples and the signal amplitude remains above a fraction of
    the channel dynamic range.

    Parameters
    ----------
    x : array-like, shape (n_time,)
        Time axis.
    y : array-like
        Decay histogram. Accepted shapes are ``(n_time,)`` or multi-channel
        arrays with the time dimension specified by ``axis``.
    window_length : int, default 11
        Savitzky-Golay window length. Must be odd and greater than
        ``polyorder``.
    polyorder : int, default 3
        Savitzky-Golay polynomial order.
    persistence : int, default 5
        Number of consecutive derivative samples used to confirm ``t1``.
    threshold : float, default 0.05
        Minimum signal amplitude, expressed as a fraction of the per-channel
        range, required for ``t1`` detection.
    axis : int, default 0
        Time axis in ``y``.
    return_bounds : bool, default False
        If ``True``, also return the detected ``t0`` and ``t1`` indices.

    Returns
    -------
    float or ndarray
        Lifetime estimate(s). Returns a scalar for 1D input and an array for
        multi-channel input.
    tuple
        When ``return_bounds=True``, returns ``(tau, t0, t1)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1:
        raise ValueError("x must be a 1D time axis")
    if y.ndim == 0:
        raise ValueError("y must contain at least one sample")
    if y.shape[axis] != x.shape[0]:
        raise ValueError("x and y must have matching lengths along the time axis")

    y_moved = np.moveaxis(y, axis, 0)
    original_shape = y_moved.shape[1:]
    y_2d = y_moved.reshape(x.shape[0], -1)
    n_time = y_2d.shape[0]

    if n_time < 3:
        raise ValueError("at least three time samples are required")
    if persistence < 1:
        raise ValueError("persistence must be at least 1")
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd")
    if window_length > n_time:
        raise ValueError("window_length cannot exceed the number of time samples")
    if polyorder >= window_length:
        raise ValueError("polyorder must be smaller than window_length")

    delta = float(np.mean(np.diff(x))) if n_time > 1 else 1.0
    dy = savgol_filter(
        y_2d,
        window_length=window_length,
        polyorder=polyorder,
        deriv=1,
        delta=delta,
        axis=0,
        mode="interp",
    )

    t0 = np.argmin(dy, axis=0)
    y_min = np.min(y_2d, axis=0)
    y_range = np.max(y_2d, axis=0) - y_min
    t1 = np.full(y_2d.shape[1], n_time - 1, dtype=int)
    tau = np.full(y_2d.shape[1], np.nan, dtype=float)

    for ch in range(y_2d.shape[1]):
        stop = n_time - persistence
        for idx in range(t0[ch] + 1, stop):
            avg_diff = np.mean(dy[idx:idx + persistence, ch])
            amplitude = max(y_2d[idx + persistence, ch] - y_min[ch], 0.0)
            if avg_diff > 0.0 and amplitude > threshold * y_range[ch]:
                t1[ch] = idx
                break

        x_window = x[t0[ch]:t1[ch] + 1]
        y_window = y_2d[t0[ch]:t1[ch] + 1, ch]
        x_local = x_window - np.min(x_window)
        y_clamped = np.clip(y_window - np.min(y_window), a_min=0.0, a_max=None)
        weight_sum = np.sum(y_clamped)
        if weight_sum > 0.0:
            tau[ch] = np.sum(x_local * y_clamped) / weight_sum

    tau = tau.reshape(original_shape) if original_shape else tau[0]
    t0 = t0.reshape(original_shape) if original_shape else int(t0[0])
    t1 = t1.reshape(original_shape) if original_shape else int(t1[0])

    if return_bounds:
        return tau, t0, t1
    return tau

def estimate_lifetime_from_log(
    data_hist,
    t_ns,
    dt_ns,
    nbin,
    period_ns,
    start_level=0.95,
    end_level=0.25,
):
    data_hist = np.asarray(data_hist, dtype=float)
    t_ns = np.asarray(t_ns, dtype=float)
    if (
        data_hist.size < 4
        or t_ns.size != data_hist.size
        or not np.isfinite(dt_ns)
        or dt_ns <= 0
        or nbin <= 0
        or not np.isfinite(period_ns)
        or period_ns <= 0
        or not np.isfinite(start_level)
        or not np.isfinite(end_level)
        or not (0.0 < end_level < start_level < 1.0)
    ):
        return None, None, None, None, None

    peak_idx = int(np.argmax(data_hist))
    trace_sum_peak0 = np.roll(data_hist, -peak_idx)
    trace_x_peak0_ns = np.roll(
        np.mod(t_ns - t_ns[peak_idx], period_ns),
        -peak_idx,
    )

    peak_value = float(trace_sum_peak0[0])
    if not np.isfinite(peak_value) or peak_value <= 0:
        return None, None, peak_idx, trace_sum_peak0, trace_x_peak0_ns

    y_start = start_level * peak_value
    idx_start_candidates = np.flatnonzero(trace_sum_peak0 <= y_start)
    fallback_levels = [end_level, 0.30, 0.40]
    fallback_levels = [level for level in fallback_levels if 0.0 < level < start_level]
    fallback_levels = list(dict.fromkeys(fallback_levels))
    selected_end_level = None
    idx_end_candidates = np.asarray([], dtype=int)
    y_end = None
    for candidate_end_level in fallback_levels:
        y_end_candidate = candidate_end_level * peak_value
        idx_end_candidate = np.flatnonzero(trace_sum_peak0 <= y_end_candidate)
        if idx_start_candidates.size != 0 and idx_end_candidate.size != 0:
            selected_end_level = candidate_end_level
            idx_end_candidates = idx_end_candidate
            y_end = y_end_candidate
            break

    if idx_start_candidates.size == 0 or idx_end_candidates.size == 0:
        return None, None, peak_idx, trace_sum_peak0, trace_x_peak0_ns

    start_idx = int(idx_start_candidates[0])
    end_idx = int(idx_end_candidates[0])
    if end_idx <= start_idx:
        return None, None, peak_idx, trace_sum_peak0, trace_x_peak0_ns

    y_section = trace_sum_peak0[start_idx : end_idx + 1]
    x_section_ns = trace_x_peak0_ns[start_idx : end_idx + 1]
    positive = y_section > 0
    if np.count_nonzero(positive) < 4:
        return None, None, peak_idx, trace_sum_peak0, trace_x_peak0_ns

    x_fit_bins = np.arange(start_idx, end_idx + 1, dtype=float)[positive]
    x_fit_ns = x_section_ns[positive]
    y_fit_input = y_section[positive]
    slope, intercept = np.polyfit(x_fit_bins, np.log(y_fit_input), 1)
    if not np.isfinite(slope) or slope >= 0:
        return None, None, peak_idx, trace_sum_peak0, trace_x_peak0_ns

    tau_ns = -float(dt_ns) / slope
    if not np.isfinite(tau_ns) or tau_ns <= 0:
        return None, None, peak_idx, trace_sum_peak0, trace_x_peak0_ns

    y_fit = np.exp(intercept + slope * x_fit_bins)
    x_fit_bins_plot = np.mod(x_fit_bins + int(peak_idx), int(nbin))
    x_fit_ns_plot = np.mod(x_fit_ns + int(peak_idx) * dt_ns, period_ns)

    fit_curve = {
        "fit_len": int(end_idx - start_idx + 1),
        "x_fit_bins": x_fit_bins_plot,
        "x_fit_ns": x_fit_ns_plot,
        "y_fit": y_fit,
    }
    return float(tau_ns), fit_curve, peak_idx, trace_sum_peak0, trace_x_peak0_ns


def estimate_lifetime_from_circmean(counts, t0_ns, repetition_rate_MHz=40.0,
                                 background_per_bin=0.0, truncate_ns=None):
    y = np.asarray(counts, dtype=float)
    n = y.size

    Tcycle_ns = 1e3 / repetition_rate_MHz
    dt_ns = Tcycle_ns / n

    t_ns = (np.arange(n) + 0.5) * dt_ns

    delay_ns = (t_ns - t0_ns) % Tcycle_ns

    w = np.clip(y - background_per_bin, 0.0, None)

    if truncate_ns is not None:
        keep = delay_ns < truncate_ns
        w = w[keep]
        delay_ns = delay_ns[keep]

    s0 = np.sum(w)
    if s0 <= 0:
        return np.nan

    tau_ns = np.sum(w * delay_ns) / s0
    return tau_ns


# ---------------------------------------------------------------------------
# Phasor analysis and visualization
# ---------------------------------------------------------------------------
def plot_universal_circle(ax=None):
    """Draw the universal circle and the unit circle on the current axes."""
    if ax is None:
        ax = gca()
    n = np.linspace(0, 2 * np.pi)
    g = 0.5 * (np.cos(n) + 1)
    s = 0.5 * np.sin(n)
    ax.plot(g, s, "--")
    n = np.linspace(0, 2 * np.pi)
    g = np.cos(n)
    s = np.sin(n)
    ax.plot(g, s, "--")
    ax.plot([-1.1, 1.1], [0, 0], "--k")
    ax.plot([0, 0], [-1.1, 1.1], "--k")


def flatten_and_remove_nan(a, b=None):
    """Flatten arrays and drop entries that contain NaNs."""

    a = a.flatten()
    if b is None:
        cond = np.isfinite(a)
        return a[cond]
    if b is not None:
        b = b.flatten()
        cond = np.logical_not(np.logical_or(np.isnan(a), np.isnan(b)))
        return a[cond], b[cond]


def g0s0_to_m_phi(g0, s0):
    """Convert Cartesian phasor coordinates into modulus and phase."""
    phi = np.arctan2(s0, g0)
    m = np.sqrt(s0 ** 2 + g0 ** 2)
    return m, phi


def m_phi_to_g0s0(m, phi):
    """Convert modulus and phase into Cartesian phasor coordinates."""
    g0 = np.cos(phi) * m
    s0 = np.sin(phi) * m
    return g0, s0


def phasor(data: np.ndarray, threshold: float = 0, harmonic: int = 1,
           time_axis: int = -1):
    dset = data.copy()
    flux = dset.sum(time_axis)
    if time_axis != -1:
        dset = np.swapaxes(dset, time_axis, -1)
    transform = np.fft.fft(dset, axis=time_axis)[..., harmonic].conj()
    if transform.ndim == dset.ndim:
        transform = np.swapaxes(transform, -1, time_axis)

    return np.where(flux < threshold, np.nan + 1j * np.nan, transform / flux)


def calculate_phasor(data_hist, threshold: float = 1, harmonic: int = 1):
    """Calculate the selected phasor harmonic from 1D, 2D, or 3D histograms."""

    data = np.asarray(data_hist, dtype=float).copy()
    if len(data.shape) == 1:
        flux = float(np.sum(data))
        if flux < threshold:
            return np.nan + 1j * np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.fft.fft(data / flux)
        return f[harmonic].conj()
    elif len(data.shape) == 2:
        flux = np.sum(data, axis=0, dtype=float)
        normalized = np.divide(
            data,
            flux,
            out=np.zeros_like(data, dtype=float),
            where=flux > 0,
        )
        f = np.fft.fft(normalized, axis=0)
        out = f[harmonic]
        out[flux < threshold] = np.nan + 1j * np.nan
        return out.conj()
    elif len(data.shape) == 3:
        flux = data.sum(-1)
        transform = np.fft.fft(data, axis=-1)[..., harmonic].conj()
        out = np.divide(
            transform,
            flux,
            out=np.full(transform.shape, np.nan + 1j * np.nan, dtype=complex),
            where=flux > 0,
        )
        out[flux < threshold] = np.nan + 1j * np.nan
        return out
    else:
        raise ValueError("Wrong input dimension")


def calculate_tau_phi(g0_or_complex, s0=None, dfd_freq=41.48e6):
    if s0 is None:
        phi = np.angle(g0_or_complex)
        m = np.abs(g0_or_complex)
    else:
        g0 = g0_or_complex
        phi = np.arctan2(s0, g0)
        m = np.sqrt(s0 ** 2 + g0 ** 2)

    tau_phi = np.tan(phi) / (2 * np.pi * dfd_freq)

    return tau_phi


def calculate_tau_m(g0_or_complex, s0=None, dfd_freq=41.48e6):
    if s0 is None:
        phi = np.angle(g0_or_complex)
        m = np.abs(g0_or_complex)
    else:
        g0 = g0_or_complex
        phi = np.arctan2(s0, g0)
        m = np.sqrt(s0 ** 2 + g0 ** 2)

    tau_m = np.sqrt((1. / (m ** 2)) - 1) / (2 * np.pi * dfd_freq)

    return tau_m


def calculate_m_phi_tau_phi_tau_m(g0_or_complex, s0=None, dfd_freq=41.48e6):
    if s0 is None:
        phi = np.angle(g0_or_complex)
        m = np.abs(g0_or_complex)
    else:
        g0 = g0_or_complex
        phi = np.arctan2(s0, g0)
        m = np.sqrt(s0 ** 2 + g0 ** 2)

    tau_phi = np.tan(phi) / (2 * np.pi * dfd_freq)
    tau_m = np.sqrt((1. / (m ** 2)) - 1) / (2 * np.pi * dfd_freq)

    return phi, m, tau_phi, tau_m


def calculate_phasor_on_img_ch(data_input, threshold=1, harmonic=1, phasor_data_size=100):
    '''

    :param data_input: data_input [r, z, y, x, t, ch]
    :param threshold:
    :param harmonic:
    :param merge_pixels:

    :return: out G=[r,z,y,x,y,ch,0] S=[r,z,y,x,y,ch,1]
    '''

    x_dim, y_dim, bin_dim, channel_dim = data_input.shape[0], data_input.shape[1], data_input.shape[2], \
        data_input.shape[3]
    with h5py.File('dataset_of_phasor_per_channel', 'w') as fi:

        h5_dim = (x_dim, y_dim, channel_dim)
        h5_dataset_p = fi.create_dataset('h5_dataset_p', shape=h5_dim, dtype=np.complex128)

        for x_start in range(0, x_dim, phasor_data_size):
            for y_start in range(0, y_dim, phasor_data_size):
                x_stop = min(x_start + phasor_data_size, x_dim)
                y_stop = min(y_start + phasor_data_size, y_dim)

                slice_term_p = np.s_[x_start:x_stop, y_start:y_stop, :, :]
                sub_image_p = data_input[slice_term_p]

                aligned_phasor = np.zeros(sub_image_p.shape[:-2] + (sub_image_p.shape[-1],), dtype=np.complex128)

                for cc in np.arange(sub_image_p.shape[-1]):
                    for yy in np.arange(sub_image_p.shape[-4]):
                        for xx in np.arange(sub_image_p.shape[-3]):
                            aligned_phasor[yy, xx, cc] = phasor(sub_image_p[yy, xx, :, cc],
                                                                threshold,
                                                                harmonic)
                h5_dataset_p[x_start:x_stop, y_start:y_stop,
                :] = aligned_phasor
    fi.close()


def phasor_h5(data_path, data_input, harmonic: int = 1):
    data_input_shape = data_input.shape
    if len(data_input_shape) == 4:
        data_input_3d = np.sum(data_input, axis=-1)
    else:
        data_input_3d = data_input.copy()

    directory, filename = os.path.split(data_path)
    filename = os.path.splitext(filename)[0]
    new_filename = filename + "_phasors_matrix.h5"
    new_file_path = os.path.join(directory, new_filename)
    if len(data_input_shape) < 3:
        raise ValueError("data_input must have 3 or more dimensions")
    elif len(data_input_shape) >= 6:
        raise ValueError("use calculate_phasor_on_img_ch() instead")

    with h5py.File(new_file_path, 'w') as fil:

        x_dim, y_dim = data_input_3d.shape[0], data_input_3d.shape[1]
        h5_dim = (x_dim, y_dim)
        h5_dataset_phasor_pix = fil.create_dataset('h5_dataset_phasor_pix', shape=h5_dim, dtype=np.complex128)

        h5_dataset_phasor_pix[:, :] = calculate_phasor(data_input_3d[:, :, :],
                                                       harmonic)
        return h5_dataset_phasor_pix

    fil.close()


def calculate_phasor_on_img(data_input, threshold=1, harmonic=1):
    '''

    :param data_input: data_input [y, x, t] or [z, y, x, t] or [r, z, y, x, t]
    :param threshold:
    :param harmonic:

    :return: out G=[r,z,y,x,y,0] S=[r,z,y,x,y,1]
    '''

    data_input_shape = data_input.shape
    if len(data_input_shape) <= 3:
        raise ValueError("data_input must have 3 or more dimensions")
    elif len(data_input_shape) >= 6:
        raise ValueError("use calculate_phasor_on_img_ch() instead")

    if len(data_input_shape) == 3:
        data_input = data_input[None, None, :, :, :]
    elif len(data_input_shape) == 4:
        data_input = data_input[None, :, :, :, :]

    out = np.zeros(data_input.shape)

    for rr in tqdm(np.arange(data_input.shape[-5])):
        for zz in tqdm(np.arange(data_input.shape[-4])):
            for yy in tqdm(np.arange(data_input.shape[-3])):
                for xx in np.arange(data_input.shape[-2]):
                    out[rr, zz, yy, xx] = phasor(data_input[rr, zz, yy, xx, :], threshold, harmonic)

    if len(data_input_shape) == 3:
        out = out[0, 0, :, :, :]
    elif len(data_input_shape) == 4:
        out = out[0, :, :, :, :]

    return out


def calculate_phasor_on_img_pixels(data_path, data_input, threshold=1, harmonic=1, phasor_pix_data_size=100):
    '''

    :param data_path: path of the analyzed sample
    :param data_input: data_input [y, x, t] or [z, y, x, t] or [r, z, y, x, t]
    :param threshold:
    :param harmonic:
    :param phasor_pix_data_size: slice of pixels on which the phasors are computed
    :return: out G=[r,z,y,x,y,0] S=[r,z,y,x,y,1]
    '''
    data_input_3d = np.sum(data_input, axis=-1)
    data_input_shape = data_input_3d.shape

    directory, filename = os.path.split(data_path)
    filename = os.path.splitext(filename)[0]
    new_filename = filename + "_phasors_matrix.h5"
    new_file_path = os.path.join(directory, new_filename)
    if len(data_input_shape) < 3:
        raise ValueError("data_input must have 3 or more dimensions")
    elif len(data_input_shape) >= 6:
        raise ValueError("use calculate_phasor_on_img_ch() instead")

    with h5py.File(new_file_path, 'w') as fil:

        x_dim, y_dim = data_input_3d.shape[0], data_input_3d.shape[1]
        h5_dim = (x_dim, y_dim)
        h5_dataset_phasor_pix = fil.create_dataset('h5_dataset_phasor_pix', shape=h5_dim, dtype=np.complex128)

        for x_start in range(0, x_dim, phasor_pix_data_size):
            for y_start in range(0, y_dim, phasor_pix_data_size):
                x_stop = min(x_start + phasor_pix_data_size, x_dim)
                y_stop = min(y_start + phasor_pix_data_size, y_dim)

                slice_term_pix = np.s_[x_start:x_stop, y_start:y_stop, :]
                sub_image_pix = data_input_3d[slice_term_pix]

                aligned_phasor_pix = np.zeros(sub_image_pix.shape[:-1], dtype=np.complex128)

                for yyy in np.arange(sub_image_pix.shape[-3]):
                    for xxx in np.arange(sub_image_pix.shape[-2]):
                        aligned_phasor_pix[yyy, xxx] = phasor(sub_image_pix[yyy, xxx, :],
                                                              threshold,
                                                              harmonic)
                h5_dataset_phasor_pix[x_start:x_stop, y_start:y_stop] = aligned_phasor_pix
    fil.close()


def plot_tau(list_value=None, dfd_freq=41.48e6, ax=None):
    '''
    :param list_value: values to draw on the universal circle, default = np.arange(10) * 1e-9
    :param dfd_freq:
    :return:
    '''

    if ax is None:
        ax = gca()

    if list_value is None:
        list_value = np.array([1, 2, 3, 4, 6, 9, 16]) * 1e-9

    for i in list_value:
        x0 = np.cos(np.arctan(2 * np.pi * i * dfd_freq))
        y0 = np.sin(np.arctan(2 * np.pi * i * dfd_freq))
        x = [0, x0]
        y = [0, y0]
        m, phi = g0s0_to_m_phi(x0, y0)

        m = 1.
        ttt = y0 / x0
        x0 = 1 / (1 + ttt ** 2)
        y0 = np.sqrt(0.25 - (x0 - 0.5) ** 2)
        x = [0, x0]
        y = [0, y0]

        m, phi = g0s0_to_m_phi(x0, y0)
        ax.plot(x, y, ".k")

        gamma = np.arctan2(y0, x0 - 0.5)

        ax.text(.57 * np.cos(gamma) + 0.5, .57 * np.sin(gamma),
                " %d ns" % (i * 1e9),
                horizontalalignment="center",
                verticalalignment="center",
                color="red")


def plot_funny_single_phasor(data, correction_phi=0, correction_m=1., ax=None):
    '''
    :param correction_phi:
    :param correction_m:
    :return:
    '''
    N = sum(data)
    l = data.shape[0]
    w = np.linspace(0, 2 * np.pi, l)
    s = np.sin(w - correction_phi) / correction_m
    c = np.cos(w - correction_phi) / correction_m
    g0 = c * data
    s0 = s * data
    ax.plot(len(g0) * g0 / N, len(s0) * s0 / N, ".k")
    ax.plot(sum(g0 / N), sum(s0 / N), "or")
    ax.ylim(-10, 10)
    ax.xlim(-10, 10)
    ax.plot(np.cos(np.linspace(0, 2 * np.pi)), np.sin(np.linspace(0, 2 * np.pi)))


def plot_phasor(phasors, bins_2dplot=100, log_scale=True, draw_universal_circle=True, tau_labels=True, quadrant='all',
                fig=None, ax=None, dfd_freq=41.48e6, cmap='viridis'):

    if fig is None:
        fig, ax = plt.subplots()
    elif fig is not None and ax is None:
        ax = gca()
    phasors_flat = flatten_and_remove_nan(phasors)

    if draw_universal_circle == True:
        plot_universal_circle(ax)

    if log_scale:
        im = ax.hist2d(np.real(phasors_flat), np.imag(phasors_flat), range=[[-1, 1], [-1, 1]], bins=bins_2dplot,
                       norm=colors.LogNorm(), cmap=cmap)
    else:
        im = ax.hist2d(np.real(phasors_flat), np.imag(phasors_flat), range=[[-1, 1], [-1, 1]], bins=bins_2dplot,
                       cmap=cmap)

    if quadrant == 'all':
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
    elif quadrant == 'first':
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, 0.6)

    if tau_labels == True:
        plot_tau(ax=ax, dfd_freq=dfd_freq)

    ax.set_xlabel('g')
    ax.set_ylabel('s')

    ax.set_aspect('equal', 'box')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im[-1], cax=cax)
    cax.set_ylabel('Pixel counts')

    fig.tight_layout()

    return fig, ax


def _finite_bounds(array):
    finite = np.asarray(array, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return [0.0, 1.0]
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    if np.isclose(lower, upper):
        upper = lower + 1.0
    return [lower, upper]


def _weighted_quantile(values, quantiles, sample_weight=None):
    values = np.asarray(values, dtype=float).ravel()
    quantiles = np.asarray(quantiles, dtype=float)
    if sample_weight is None:
        sample_weight = np.ones(values.size, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float).ravel()
        if sample_weight.size != values.size:
            raise ValueError("sample_weight must match values size")
    mask = np.isfinite(values) & np.isfinite(sample_weight) & (sample_weight > 0)
    values = values[mask]
    sample_weight = sample_weight[mask]
    if values.size == 0:
        return np.full_like(quantiles, np.nan, dtype=float)
    order = np.argsort(values)
    values = values[order]
    sample_weight = sample_weight[order]
    cumulative = np.cumsum(sample_weight)
    cumulative = (cumulative - 0.5 * sample_weight) / cumulative[-1]
    return np.interp(
        np.clip(quantiles, 0.0, 1.0),
        cumulative,
        values,
        left=values[0],
        right=values[-1],
    )


def build_lifetime_equalizer(reference_lifetime, lifetime_bounds, weights=None, n_quantiles=2048,
                             strength=1.0, equalization_bins=None):
    """Build a lifetime-to-equalized-axis mapping from a reference distribution."""
    low, high = map(float, lifetime_bounds)
    if np.isclose(low, high):
        return np.array([low, high + 1.0]), np.array([0.0, 1.0])

    strength = float(strength)
    if strength <= 0.0:
        return np.array([low, high]), np.array([0.0, 1.0])

    reference_lifetime = np.asarray(reference_lifetime, dtype=float).ravel()
    if weights is None:
        weights = None
    else:
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.size != reference_lifetime.size:
            raise ValueError("weights must match reference_lifetime size")

    mask = np.isfinite(reference_lifetime)
    mask &= (reference_lifetime >= low) & (reference_lifetime <= high)
    if weights is not None:
        mask &= np.isfinite(weights) & (weights > 0)
    if not np.any(mask):
        return np.array([low, high]), np.array([0.0, 1.0])

    if equalization_bins is None:
        n_bins = max(int(n_quantiles) - 1, 32)
    else:
        n_bins = max(int(equalization_bins), 32)

    histogram, edges = np.histogram(
        reference_lifetime[mask],
        bins=n_bins,
        range=(low, high),
        weights=None if weights is None else weights[mask],
    )
    histogram = histogram.astype(float)
    histogram = np.power(histogram, strength)
    keep = histogram > 0.0
    if not np.any(keep):
        return np.array([low, high]), np.array([0.0, 1.0])

    support = 0.5 * (edges[:-1] + edges[1:])
    support = support[keep]
    histogram = histogram[keep]
    cumulative = np.cumsum(histogram)
    equalized_axis = (cumulative - 0.5 * histogram) / cumulative[-1]
    support = np.concatenate(([low], support, [high]))
    equalized_axis = np.concatenate(([0.0], equalized_axis, [1.0]))
    return support, equalized_axis


def apply_lifetime_equalizer(lifetime, support, equalized_axis):
    """Map lifetime values onto a precomputed equalized axis."""
    lifetime = np.asarray(lifetime, dtype=float)
    clipped = np.clip(lifetime, support[0], support[-1])
    equalized = np.interp(clipped, support, equalized_axis, left=0.0, right=1.0)
    equalized[~np.isfinite(lifetime)] = np.nan
    return equalized


def equalized_lifetime_tick_values(reference_lifetime, lifetime_bounds, weights=None, n_ticks=6,
                                   n_quantiles=2048, strength=1.0, equalization_bins=None):
    """Return equalized-axis tick positions together with their lifetime labels."""
    support, equalized_axis = build_lifetime_equalizer(
        reference_lifetime,
        lifetime_bounds,
        weights=weights,
        n_quantiles=n_quantiles,
        strength=strength,
        equalization_bins=equalization_bins,
    )
    tick_pos = np.linspace(0.0, 1.0, int(n_ticks))
    tick_values = np.interp(tick_pos, equalized_axis, support)
    return tick_pos, tick_values


def _equal_luminance_colormap_array(colormap):
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0.0, 1.0, cmap.N))[:, :3].copy()
    luminance = (299 * colors[:, 0] + 587 * colors[:, 1] + 114 * colors[:, 2]) / 1000.0
    max_luminance = np.max(luminance)
    for idx, lum in enumerate(luminance):
        if lum <= 0:
            continue
        scale = min(max_luminance / lum, 1.0 / max(np.max(colors[idx]), 1e-12))
        colors[idx] *= scale
    return colors


def _normalize_intensity(image, intensity_bounds):
    low, high = map(float, intensity_bounds)
    image = np.asarray(image, dtype=float)
    clipped = np.clip(image, low, high)
    normalized = (clipped - low) / max(high - low, np.finfo(float).eps)
    normalized[~np.isfinite(image)] = 0.0
    return normalized


def _flim_rgb_from_equalized_axis(image, equalized_lifetime, intensity_bounds, colormap):
    cmap_array = _equal_luminance_colormap_array(colormap)
    n_colors = cmap_array.shape[0]
    intensity = _normalize_intensity(image, intensity_bounds)
    equalized_lifetime = np.asarray(equalized_lifetime, dtype=float)
    invalid = ~np.isfinite(image) | ~np.isfinite(equalized_lifetime)
    equalized_lifetime = np.clip(np.nan_to_num(equalized_lifetime, nan=0.0), 0.0, 1.0)
    idx = np.clip(np.floor(equalized_lifetime * (n_colors - 1)).astype(int), 0, n_colors - 1)
    rgb = cmap_array[idx] * intensity[..., None]
    rgb[invalid] = 0.0
    return rgb


def show_flim_equalized(image, lifetime, pxsize, pxdwelltime, lifetime_bounds=None,
                        intensity_bounds=None, colormap="gist_rainbow", fig=None, ax=None,
                        equalization_reference=None, equalization_weights=None,
                        n_quantiles=2048, colorbar_ticks=6, equalization_strength=1.0,
                        equalization_bins=None):
    """Show a FLIM image using an equalized lifetime axis for the hue mapping."""
    from matplotlib_scalebar.scalebar import ScaleBar

    image = np.asarray(image, dtype=float)
    lifetime = np.asarray(lifetime, dtype=float)
    pxsize = float(np.asarray(pxsize).squeeze())
    pxdwelltime = float(np.asarray(pxdwelltime).squeeze())

    if intensity_bounds is None:
        intensity_bounds = _finite_bounds(image)
    if lifetime_bounds is None:
        lifetime_bounds = _finite_bounds(lifetime)
    if equalization_reference is None:
        equalization_reference = lifetime

    support, equalized_axis = build_lifetime_equalizer(
        equalization_reference,
        lifetime_bounds,
        weights=equalization_weights,
        n_quantiles=n_quantiles,
        strength=equalization_strength,
        equalization_bins=equalization_bins,
    )
    lifetime_equalized = apply_lifetime_equalizer(lifetime, support, equalized_axis)
    rgb = _flim_rgb_from_equalized_axis(image, lifetime_equalized, intensity_bounds, colormap)

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    img_extent = (0, image.shape[1] * pxsize, 0, image.shape[0] * pxsize)
    ax.imshow(rgb, extent=img_extent)
    ax.axis("off")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    n_bar = 256
    intensity_gradient = np.linspace(intensity_bounds[0], intensity_bounds[1], n_bar)
    equalized_gradient = np.linspace(0.0, 1.0, n_bar)
    intensity_grid = np.tile(intensity_gradient, (n_bar, 1)).T
    equalized_grid = np.tile(equalized_gradient, (n_bar, 1))
    rgb_colorbar = _flim_rgb_from_equalized_axis(
        intensity_grid, equalized_grid, intensity_bounds, colormap
    )
    rgb_colorbar = np.moveaxis(rgb_colorbar, 0, 1)
    cax.imshow(
        rgb_colorbar,
        origin="lower",
        aspect="auto",
        extent=(intensity_bounds[0], intensity_bounds[1], 0.0, 1.0),
    )
    cax.set_xticks([int(intensity_bounds[0]), int(intensity_bounds[1])])
    cax.set_xlabel(f"Counts/{pxdwelltime} " + "$\\mathregular{\\mu s}$")
    tick_pos = np.linspace(0.0, 1.0, int(colorbar_ticks))
    tick_values = np.interp(tick_pos, equalized_axis, support)
    cax.set_yticks(tick_pos)
    cax.set_yticklabels([f"{tick:.2f}" for tick in tick_values])
    cax.set_ylabel("Lifetime (ns, equalized axis)")
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position("right")

    scalebar = ScaleBar(1, "um", box_alpha=0, color="w", length_fraction=0.25)
    ax.add_artist(scalebar)
    plt.tight_layout()
    return fig, ax


def fourier_shift(data, shift_angle=0.):
    if len(data.shape) == 1:
        w = np.arange(data.shape[0])
    elif len(data.shape) > 1:
        w = np.asarray([np.arange(data.shape[0])] * data.shape[1]).T
        fft_hists = np.fft.fft(data)
    return np.abs(np.fft.ifft(fft_hists * np.exp(1j * 2 * np.pi * shift_angle * w)))


def cmap2d(intensity, lifetime, params):
    sz = intensity.shape

    Hp = np.minimum(np.maximum(lifetime, params["minTau"]), params["maxTau"])

    Hn = ((Hp - params["minTau"]) / (params["maxTau"] - params["minTau"])) * params[
        "satFactor"
    ]
    Sn = np.ones(Hp.shape)
    Vn = (intensity - params["minInt"]) / (params["maxInt"] - params["minInt"])

    HSV = np.empty((sz[0], sz[1], 3))

    if params["invertColormap"] == True:
        Hn = params["satFactor"] - Hn

    Hn[np.not_equal(lifetime, Hp)] = params["outOfBoundsHue"]

    HSV[:, :, 0] = Hn.astype("float64")
    HSV[:, :, 1] = Sn.astype("float64")
    HSV[:, :, 2] = Vn.astype("float64")

    RGB = hsv_to_rgb(HSV)

    return RGB


def linear_shift(data, shift_value, cyclic=True):
    xp = np.arange(0, data.shape[0])
    fp = data.copy()
    x = np.arange(0, data.shape[0]) - shift_value
    if cyclic:
        x = np.mod(x, data.shape[0])
    return np.interp(x, xp, fp)


def correction_phasor(laser, laser_irf):
    phasor_laser = phasor(laser)
    phasor_laser_irf = phasor(laser_irf)

    angle_laser = np.angle(phasor_laser)
    angle_laser_irf = np.angle(phasor_laser_irf)

    return np.exp(-1j * (angle_laser - angle_laser_irf))

def threshold_intensity(intensity_map, threshold=0.15):
    max_counts = intensity_map.max()
    idx = intensity_map > threshold * max_counts
    thresholded_intensity = intensity_map[idx]
    return thresholded_intensity


def threshold_phasor(intensity_map, phasor_map, threshold=0.15):
    max_counts = intensity_map.max()
    idx = intensity_map > threshold * max_counts
    thresholded_phasor = phasor_map[idx].ravel()

    return thresholded_phasor

def median_phasors(phasor_map, window=3):
    filtered_phasors = median_filter(phasor_map, size=window)

    return filtered_phasors


def calculate_irf_correction(hist_irf, n_bins):
    phasors_irf = calculate_phasor(hist_irf)
    angle_irf = np.angle(phasors_irf)
    correction_term = (angle_irf / (2 * np.pi)) * n_bins

    return correction_term


def m_phi_to_complex(m, phi):
    return m * np.exp(-1j * phi)


def find_shifts_irf_data(data_reference, data):
    '''
    It calculates the temporal shifts through phase cross-correlation (up to 1/100 of a time bin) between the decay histograms
    of the reference data used for IRF calibration and the actual data to be analyzed, channel by channel

    '''
    data_hist_irf = data_reference.sum((0, 1))
    data_hist_input = data.sum((0, 1))
    nch = data_hist_irf.shape[-1]

    shift_vec = np.empty(nch)
    for i in range(nch):
        shift_vec[i], *_ = phase_cross_correlation(data_hist_irf[:, i], data_hist_input[:, i], upsample_factor=100,
                                                   normalization=None)

    return shift_vec


def find_temporal_shifts(data_4D):
    '''
    It calculates the temporal shifts through phase cross-correlation (up to 1/100 of a time bin) between the decay histograms
    of the 12th SPAD channel (used as reference) and the decay histograms of the remaining channels

    '''
    data_hist = data_4D.sum((0, 1))
    nch = data_hist.shape[-1]

    shift_vec = np.empty(nch)

    for i in range(nch):
        shift_vec[i], *_ = phase_cross_correlation(data_hist[:, 12], data_hist[:, i], upsample_factor=100,
                                                   normalization=None)

    return shift_vec


def align_decays(data_4D, shift_vec):
    '''
    Shift the image's decay histograms in each pixel by the quantity defined in the parameter 'shift vector'. You
    can find the quantity shift vector by running the function 'find_temporal_shifts'.

    : param data_4D: input ISM image (x,y,t,ch) with mis-aligned channels' (ch) time-decay histograms (t)
    : param shift_vector: vector of 25 values representing the time shift of each channel with respect to the considered reference
    : return: ISM image with aligned channels' time-decay histograms
    '''
    nch = data_4D.shape[-1]
    data_hist = data_4D.sum((0, 1))
    data_hist_shift = np.empty_like(data_hist)
    for i in range(nch):
        data_hist_shift[:, i] = shift(data_hist[:, i], shift_vec[i], order=1, mode='grid-wrap')

    return data_hist_shift


def align_image(data_4D, shift_vector):
    '''
    Shift the image's decay histograms in each pixel by the quantity defined in the parameter shift vector.
    If you want to align the temporal decays of the different SPAD channels, you can find the quantity
    'shift vector' by running the function 'find_temporal_shifts'.
    If you want to shift the channels' temporal decays to match the shifts of a reference image, you can find
    the quantity 'shift vector' by running the function 'find_shifts_irf_data'.

    : param data_4D: input ISM image (x,y,t,ch) with mis-aligned channels' (ch) time-decay histograms (t)
    : param shift_vector: vector of 25 values representing the time shift of each channel with respect to the considered reference
    : return: ISM image with aligned channels' time-decay histograms
    '''
    nch = data_4D.shape[-1]
    image_4D_aligned = np.empty_like(data_4D)
    shift_dset = np.zeros((data_4D.ndim - 1, nch))
    shift_dset[-1] = shift_vector

    for i in range(nch):
        image_4D_aligned[..., i] = shift(data_4D[..., i], shift_dset[:, i], order=1, mode='grid-wrap')

    return image_4D_aligned


def calibrate_phasor(data, data_reference, ch=12, tau_m_reference=2.5 * 10 ** -9, tau_phi_reference=2.5 * 10 ** -9,
                     laser_MHz=40, h=1, processing='Open confocal'):
    '''
    Clean the measured phasor of the IRF

    : param data: input ISM raw image (x,y,t,ch) before IRF calibration. The image has mis-aligned channels' (ch) time-decay histograms (t)
    : param data_reference: ISM raw image (x,y,t,ch) of a reference dye with known mono-exponential decay used for extracting the phasor of the IRF
    : param ch: specific channel in the SPAD array to be processed if 'Singel channel' is selected
    : param tau_m_reference: expected lifetime of the reference sample in 'reference_data' extracted with tau_m
    : param tau_phi_reference: expected lifetime of the reference sample in 'reference_data' extracted with tau_phi
    : param laser_MHz: frequency of the pulsed laser source used for fluorescence excitation (in MHz)
    : param h: harmonic considered in the phasor computation

    : return: phasor_corrected: complex variable containing the phasor in each pixel of 'data' corrected of the IRF
    '''
    if processing == 'Single channel':
        shift_vec = find_shifts_irf_data(data_reference, data)
        data_aligned = align_image(data, shift_vec)

        data_hist_irf = data_reference.sum((0, 1))
        phasor_calibration = np.zeros(data_hist_irf.shape[-1], dtype=complex)

        for i in range(data_hist_irf.shape[-1]):
            phasor_calibration[i] = phasor(data_hist_irf[:, i])

        nch = data_hist_irf.shape[-1]
        m_phi_calibration = np.zeros((nch, 2))
        for i in range(nch):
            m_phi_calibration[i] = g0s0_to_m_phi(np.real(phasor_calibration[i]), np.imag(phasor_calibration[i]))

        laser_Hz = laser_MHz * 1e6
        k = 1 / (2 * np.pi * laser_Hz * h)
        phi_ref = math.atan2(tau_phi_reference, k)
        m_ref = np.sqrt(1 / (((tau_m_reference * k) ** 2) + 1))
        m_irf = 1
        if np.real(phasor_calibration[12]) < 0 and np.imag(phasor_calibration[12]) > 0:
           phi_irf = -(np.abs(m_phi_calibration[:, 1] - phi_ref))

        if np.real(phasor_calibration[12]) > 0 and np.imag(phasor_calibration[12]) > 0:
           phi_irf = -(np.abs(m_phi_calibration[:, 1] - phi_ref))

        if np.real(phasor_calibration[12]) > 0 and np.imag(phasor_calibration[12]) < 0:
           phi_irf = (np.abs(m_phi_calibration[:, 1] - phi_ref))

        if np.real(phasor_calibration[12]) < 0 and np.imag(phasor_calibration[12]) < 0:
           phi_irf = (np.abs(m_phi_calibration[:, 1] - phi_ref))

        phasor_irf = np.zeros((nch), dtype=complex)
        for i in range(nch):
            phasor_irf[i] = m_phi_to_complex(m_irf, phi_irf[i])

        phasor_data_pixels = phasor(data_aligned[:, :, :, ch])
        phasor_corrected = phasor_data_pixels / phasor_irf[ch]
        return phasor_corrected

    if processing == 'Open confocal':
        shift_vec = find_shifts_irf_data(data_reference, data)
        data_aligned = align_image(data, shift_vec)
        shift_vec_reference = find_temporal_shifts(data_reference)
        data_aligned_reference = align_image(data_reference, shift_vec_reference)
        shift_vec_interchannel = find_temporal_shifts(data_aligned)
        data_aligned_final = align_image(data_aligned, shift_vec_interchannel)

        data_hist_irf_shift_sum = data_aligned_reference.sum((0, 1, -1))
        phasor_calibration_sum = phasor(data_hist_irf_shift_sum)
        m_phi_calibration_sum = g0s0_to_m_phi(np.real(phasor_calibration_sum), np.imag(phasor_calibration_sum))
        laser_Hz = laser_MHz * 1e6
        k = 1 / (2 * np.pi * laser_Hz * h)
        phi_ref = math.atan2(tau_phi_reference, k)
        m_irf_sum = 1

        if np.real(phasor_calibration_sum) < 0 and np.imag(phasor_calibration_sum) > 0:
           phi_irf_sum = -(np.abs(m_phi_calibration_sum[-1] - phi_ref))

        if np.real(phasor_calibration_sum) > 0 and np.imag(phasor_calibration_sum) > 0:
           phi_irf_sum = -(np.abs(m_phi_calibration_sum[-1] - phi_ref))

        if np.real(phasor_calibration_sum) > 0 and np.imag(phasor_calibration_sum) < 0:
           phi_irf_sum = (np.abs(m_phi_calibration_sum[-1] - phi_ref))

        if np.real(phasor_calibration_sum) < 0 and np.imag(phasor_calibration_sum) < 0:
           phi_irf_sum = (np.abs(m_phi_calibration_sum[-1] - phi_ref))

        phasor_irf_sum = m_phi_to_complex(m_irf_sum, phi_irf_sum)

        data_aligned_3D = np.sum(data_aligned_final, axis=-1)

        phasor_pixels_aligned = phasor(data_aligned_3D)
        phasor_corrected_pix_aligned = phasor_pixels_aligned / phasor_irf_sum
        return phasor_corrected_pix_aligned

def show_lifetime_histogram(phasors_calibrated, method='tau_phi', interval=(-2, 6), bin_number=50, dfd_freq=40e6):
    '''
    Plot the histogram of the fluorescence lifetime values the analyzed image's pixels

    : param phasors_calibrated: array with phasors in each pixel of the image extracted with calibrate_phasor function
    : param method: method used for lifetime computation from the phasors
    : param interval: range of lifetime values shown in the histogram
    : param bin number: bins of the histogram
    : dfd_freq: frequency of the pulsed laser source used for fluorescence excitation (in Hz)

        '''
    if method == 'tau_phi':
        tau_phi = calculate_tau_phi(np.real(phasors_calibrated), np.imag(phasors_calibrated), dfd_freq=dfd_freq)
        tau_data = 1e9 * tau_phi.flatten()
        plt.figure()
        plt.hist(tau_data, range=interval, bins=bin_number)

    if method == 'tau_m':
        tau_m = calculate_tau_m(np.real(phasors_calibrated), np.imag(phasors_calibrated), dfd_freq=dfd_freq)
        tau_data = 1e9 * tau_m.flatten()
        plt.figure()
        plt.hist(tau_data, range=interval, bins=bin_number)


def plot_flim_image(data_4D, phasors_calibrated, method='tau_phi', pxsize=0.04, pxdwelltime=91,
                    lifetime_bounds=(1, 2.5), log_scale=False, dfd_freq=40e6):
    '''
    Plot FLIM map of raw data_4D in input, given its calibrated phasor in each pixel

    : param data_4D: input ISM raw image (x,y,t,ch) before IRF calibration. The image has mis-aligned channels' (ch) time-decay histograms (t)
    : param phasor_calibrated: array with phasors in each pixel of the image extracted with calibrate_phasor function
    : param method: method used for lifetime computation from the phasors
    : param pxsize: pixel size of the input image (in um)
    : pxdwelltime: dwell time used in the experiment (in us)
    : lifetime_bounds: range of lifetime values displayed in the FLIM map
    : param log_scale: if True phasor plot is shown in logarithmic scale
    : dfd_freq: frequency of the pulsed laser source used for fluorescence excitation (in Hz)
    '''
    if gra is None:
        raise ImportError("plot_flim_image requires brighteyes_ism.analysis.Graph_lib")

    if method == 'tau_phi':
        data_histograms = data_4D.sum((-2, -1))
        tau_phi = calculate_tau_phi(np.real(phasors_calibrated), np.imag(phasors_calibrated), dfd_freq=dfd_freq)
        fig = plt.figure(figsize=(9, 6))
        gs = fig.add_gridspec(4, 4)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax2 = fig.add_subplot(gs[2:4, 0:2])
        plot_phasor(phasors_calibrated, bins_2dplot=200, log_scale=log_scale, quadrant='first', fig=fig, ax=ax1)
        gra.show_flim(data_histograms, tau_phi * 1e9, pxsize=pxsize, pxdwelltime=pxdwelltime,
                      lifetime_bounds=lifetime_bounds, fig=fig, ax=ax2)
        fig.tight_layout()

    if method == 'tau_m':
        data_histograms = data_4D.sum((-2, -1))
        tau_m = calculate_tau_m(np.real(phasors_calibrated), np.imag(phasors_calibrated), dfd_freq=dfd_freq)
        fig = plt.figure(figsize=(9, 6))
        gs = fig.add_gridspec(4, 4)
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax2 = fig.add_subplot(gs[2:4, 0:2])
        plot_phasor(phasors_calibrated, bins_2dplot=200, log_scale=log_scale, quadrant='first', fig=fig, ax=ax1)
        gra.show_flim(data_histograms, tau_m * 1e9, pxsize=pxsize, pxdwelltime=pxdwelltime,
                      lifetime_bounds=lifetime_bounds, fig=fig, ax=ax2)
        fig.tight_layout()
