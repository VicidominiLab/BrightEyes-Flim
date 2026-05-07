"""Common plotting helpers for BrightEyes FLIM notebooks."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from . import tools_phasor as flim

try:
    import brighteyes_ism.analysis.Graph_lib as gra
except ImportError:  # pragma: no cover - optional dependency
    gra = None

__all__ = [
    "crop_2d",
    "normalize_histogram",
    "threshold_lifetime_map",
    "weighted_lifetime_stats",
    "plot_calibration_fit_traces",
    "plot_calibration_lifetime_summary",
    "plot_calibration_shift_summary",
    "plot_channel_skew_correction",
    "plot_lifetime_histogram",
    "plot_lifetime_summary",
    "plot_equalized_lifetime_summary",
]


def _require_graph_lib():
    if gra is None:
        raise ImportError("FLIM map plotting requires brighteyes_ism.analysis.Graph_lib")


def crop_2d(array, crop=0):
    """Crop a 2D array by the same number of pixels on each border."""
    array = np.asarray(array)
    crop = int(crop)
    if crop <= 0:
        return array
    if array.ndim < 2:
        raise ValueError("array must be at least 2D")
    if array.shape[-2] <= 2 * crop or array.shape[-1] <= 2 * crop:
        raise ValueError(f"crop={crop} is too large for shape {array.shape}")
    return array[..., crop:-crop, crop:-crop]


def normalize_histogram(histogram):
    """Return a unit-sum copy of a histogram, or zeros when it cannot be normalized."""
    histogram = np.asarray(histogram, dtype=float)
    total = np.sum(histogram)
    if not np.isfinite(total) or total <= 0:
        return np.zeros_like(histogram, dtype=float)
    return histogram / total


def threshold_lifetime_map(lifetime, intensity=None, threshold=0.05, finite_only=True):
    """
    Return thresholded lifetime/intensity vectors and the 2D mask.

    ``threshold`` is a fraction of the maximum intensity when ``intensity`` is
    provided. Without intensity, only finite lifetime values are retained.
    """
    lifetime = np.asarray(lifetime, dtype=float)
    if intensity is None:
        mask = np.ones(lifetime.shape, dtype=bool)
        intensity_values = None
    else:
        intensity = np.asarray(intensity, dtype=float)
        if intensity.shape != lifetime.shape:
            raise ValueError("intensity and lifetime must have the same shape")
        max_counts = np.nanmax(intensity)
        if not np.isfinite(max_counts) or max_counts <= 0:
            mask = np.zeros(lifetime.shape, dtype=bool)
        else:
            mask = intensity > float(threshold) * max_counts
        intensity_values = intensity[mask].ravel()

    if finite_only:
        mask &= np.isfinite(lifetime)
        if intensity is not None:
            mask &= np.isfinite(intensity)
            intensity_values = intensity[mask].ravel()

    return lifetime[mask].ravel(), intensity_values, mask


def weighted_lifetime_stats(lifetime, weights=None):
    """Return weighted mean and RMS for finite lifetime values."""
    lifetime = np.asarray(lifetime, dtype=float).ravel()
    mask = np.isfinite(lifetime)
    if weights is None:
        values = lifetime[mask]
        if values.size == 0:
            return np.nan, np.nan
        return float(np.mean(values)), float(np.std(values))

    weights = np.asarray(weights, dtype=float).ravel()
    if weights.size != lifetime.size:
        raise ValueError("weights must match lifetime size")
    mask &= np.isfinite(weights) & (weights > 0)
    values = lifetime[mask]
    weights = weights[mask]
    if values.size == 0 or np.sum(weights) <= 0:
        return np.nan, np.nan
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, float(np.sqrt(max(variance, 0.0)))


def _format_histogram_axis(ax, lifetime_axis="x", count_label="Pixel counts"):
    if lifetime_axis == "x":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_xlabel("Lifetime (ns)")
        ax.set_ylabel(count_label)
    elif lifetime_axis == "y":
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")
        ax.set_xlabel(count_label)
        ax.set_ylabel("Lifetime (ns)")
    else:
        raise ValueError("lifetime_axis must be 'x' or 'y'")

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((3, 3))
    if lifetime_axis == "x":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def _column(table, name):
    if hasattr(table, "__getitem__"):
        values = table[name]
        if hasattr(values, "to_numpy"):
            values = values.to_numpy()
        return np.asarray(values)
    raise TypeError("table must provide column access by name")


def _plot_vertical_value_histogram(
    values,
    ax,
    bins,
    color=None,
    label=None,
    alpha=0.22,
    linewidth=1.4,
    gaussian=False,
    show_stats=False,
):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    mean = float(np.mean(values))
    std = float(np.std(values))
    stats_label = None
    if show_stats:
        stats_prefix = label if label is not None else "data"
        stats_label = f"{stats_prefix}: mu={mean:.3g}, std={std:.3g}"

    hist, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    can_plot_gaussian = gaussian and values.size > 1 and np.isfinite(std) and std > 0
    hist_label = label
    if show_stats:
        hist_label = stats_label if not can_plot_gaussian else None

    hist_line = ax.plot(
        hist,
        centers,
        drawstyle="steps-mid",
        color=color,
        linewidth=linewidth,
        label=hist_label,
    )[0]
    color = hist_line.get_color()
    ax.fill_betweenx(centers, hist, step="mid", color=color, alpha=alpha)

    if can_plot_gaussian:
        curve = np.exp(-0.5 * ((centers - mean) / std) ** 2)
        if np.sum(curve) > 0:
            curve = curve * (np.sum(hist) / np.sum(curve))
        ax.plot(
            curve,
            centers,
            color=color,
            linewidth=2,
            label=stats_label,
        )


def plot_calibration_lifetime_summary(summary_table, fig=None, histogram_lifetime_axis="y"):
    """Plot fitted calibration lifetime and reference lifetime by channel."""
    channels = _column(summary_table, "channel")
    tau = _column(summary_table, "tau_ns").astype(float)
    tau_err = _column(summary_table, "tau_err_ns").astype(float)
    tau_ref = _column(summary_table, "tau_ref_ns").astype(float)
    fit_error = _column(summary_table, "fit_error").astype(float)

    if fig is None:
        fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.2], height_ratios=[1, 0.55], wspace=0.16)
    ax_tau = fig.add_subplot(gs[0, 0])
    share_lifetime_axis = ax_tau if histogram_lifetime_axis == "y" else None
    ax_hist = fig.add_subplot(gs[0, 1], sharey=share_lifetime_axis)
    ax_error = fig.add_subplot(gs[1, 0], sharex=ax_tau)
    ax_error_hist = fig.add_subplot(gs[1, 1], sharey=ax_error)

    ax_tau.errorbar(
        channels,
        tau,
        yerr=tau_err,
        fmt="o-",
        color="tab:blue",
        linewidth=2,
        markersize=5,
        capsize=3,
        label="Fitted lifetime",
    )
    if np.any(np.isfinite(tau_ref)):
        ax_tau.plot(channels, tau_ref, "o-", color="tab:orange", label="Reference lifetime")
    mean_tau = np.nanmean(tau)
    std_tau = np.nanstd(tau)
    if np.isfinite(mean_tau):
        ax_tau.axhline(mean_tau, color="tab:blue", alpha=0.7)
        if np.isfinite(std_tau) and std_tau > 0:
            ax_tau.axhspan(mean_tau - std_tau, mean_tau + std_tau, color="tab:blue", alpha=0.08)
    ax_tau.set_ylabel("Lifetime (ns)")
    ax_tau.set_title("Calibration lifetime by channel")
    ax_tau.grid(True, alpha=0.3)
    ax_tau.legend(loc="best")

    plot_lifetime_histogram(
        tau,
        lifetime_bounds=None,
        bins=min(20, max(5, len(tau))),
        ax=ax_hist,
        gaussian=True,
        lifetime_axis=histogram_lifetime_axis,
        count_label="Channel count",
    )
    ax_hist.set_title("Lifetime distribution")
    if histogram_lifetime_axis == "y":
        ax_hist.tick_params(axis="y", labelleft=False, labelright=False)
        ax_hist.set_ylabel("")

    ax_error.plot(channels, fit_error, "o-", color="tab:red", linewidth=1.8, markersize=5)
    ax_error.set_xlabel("Channel")
    ax_error.set_ylabel("Fit RMSE")
    ax_error.grid(True, alpha=0.3)

    fit_error_bins = min(20, max(5, np.count_nonzero(np.isfinite(fit_error))))
    _plot_vertical_value_histogram(
        fit_error,
        ax_error_hist,
        bins=fit_error_bins,
        color="tab:red",
    )
    ax_error_hist.set_xlabel("Channel count")
    ax_error_hist.set_title("RMSE distribution")
    ax_error_hist.grid(True, alpha=0.25)
    ax_error_hist.tick_params(axis="y", labelleft=False, labelright=False)
    ax_error_hist.set_ylabel("")

    return fig, (ax_tau, ax_hist, ax_error)


def plot_calibration_shift_summary(summary_tables, labels=None, reference_channel=None, fig=None):
    """Plot stored channel skew and common delay for one or more calibration groups."""
    if not isinstance(summary_tables, (list, tuple)):
        summary_tables = [summary_tables]
    if labels is None:
        labels = [f"group {idx + 1}" for idx in range(len(summary_tables))]

    if fig is None:
        fig = plt.figure(figsize=(16, 6.4), constrained_layout=True)
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[6, 1.15],
            height_ratios=[1.2, 1],
            wspace=0.08,
            hspace=0.18,
        )
        ax_shift = fig.add_subplot(gs[0, 0])
        ax_shift_hist = fig.add_subplot(gs[0, 1], sharey=ax_shift)
        ax_delay = fig.add_subplot(gs[1, 0])
        ax_delay_hist = fig.add_subplot(gs[1, 1], sharey=ax_delay)
    else:
        if len(fig.axes) >= 4:
            ax_shift, ax_shift_hist, ax_delay, ax_delay_hist = fig.axes[:4]
        else:
            fig.clear()
            gs = fig.add_gridspec(
                2,
                2,
                width_ratios=[6, 1.15],
                height_ratios=[1.2, 1],
                wspace=0.08,
                hspace=0.18,
            )
            ax_shift = fig.add_subplot(gs[0, 0])
            ax_shift_hist = fig.add_subplot(gs[0, 1], sharey=ax_shift)
            ax_delay = fig.add_subplot(gs[1, 0])
            ax_delay_hist = fig.add_subplot(gs[1, 1], sharey=ax_delay)

    for table, label in zip(summary_tables, labels):
        channels = _column(table, "channel")
        channel_skew = _column(table, "channel_skew").astype(float)
        channel_skew_err = _column(table, "channel_skew_est_err").astype(float)
        common_delay = _column(table, "common_delay_in_ns").astype(float)
        common_delay_err = _column(table, "common_delay_err_in_ns").astype(float)

        shift_container = ax_shift.errorbar(
            channels,
            channel_skew,
            yerr=channel_skew_err,
            fmt="o-",
            linewidth=2,
            markersize=5,
            capsize=3,
            label=label,
        )
        delay_container = ax_delay.errorbar(
            channels,
            common_delay,
            yerr=common_delay_err,
            fmt="o--",
            linewidth=2,
            markersize=5,
            capsize=3,
            label=label,
        )
        shift_color = shift_container.lines[0].get_color()
        delay_color = delay_container.lines[0].get_color()
        shift_bins = min(20, max(5, np.count_nonzero(np.isfinite(channel_skew))))
        delay_bins = min(20, max(5, np.count_nonzero(np.isfinite(common_delay))))
        _plot_vertical_value_histogram(
            channel_skew,
            ax_shift_hist,
            bins=shift_bins,
            color=shift_color,
            label=label,
            gaussian=True,
            show_stats=True,
        )
        _plot_vertical_value_histogram(
            common_delay,
            ax_delay_hist,
            bins=delay_bins,
            color=delay_color,
            label=label,
            gaussian=True,
            show_stats=True,
        )

    if reference_channel is not None:
        ax_shift.axvline(reference_channel, color="0.75", linestyle=":", linewidth=1.2)

    ax_shift.axhline(0, color="0.85", linestyle="--", linewidth=1)
    ax_shift.set_ylabel("Channel skew (bins)")
    ax_shift.set_title("Channel skew")
    ax_shift.grid(True, alpha=0.3)
    ax_shift.legend(loc="best")
    ax_shift_hist.axhline(0, color="0.85", linestyle="--", linewidth=1)
    ax_shift_hist.set_xlabel("Channel count")
    ax_shift_hist.set_title("Distribution")
    ax_shift_hist.grid(True, alpha=0.25)
    ax_shift_hist.tick_params(axis="y", labelleft=False, labelright=False)
    ax_shift_hist.set_ylabel("")
    handles, _ = ax_shift_hist.get_legend_handles_labels()
    if handles:
        ax_shift_hist.legend(loc="best", fontsize="small")

    ax_delay.axhline(0, color="0.85", linestyle="--", linewidth=1)
    ax_delay.set_xlabel("Channel")
    ax_delay.set_ylabel("Common delay (ns)")
    ax_delay.set_title("Fitted common delay")
    ax_delay.grid(True, alpha=0.3)
    ax_delay.legend(loc="best")
    ax_delay_hist.axhline(0, color="0.85", linestyle="--", linewidth=1)
    ax_delay_hist.set_xlabel("Channel count")
    ax_delay_hist.set_title("Distribution")
    ax_delay_hist.grid(True, alpha=0.25)
    ax_delay_hist.tick_params(axis="y", labelleft=False, labelright=False)
    ax_delay_hist.set_ylabel("")
    handles, _ = ax_delay_hist.get_legend_handles_labels()
    if handles:
        ax_delay_hist.legend(loc="best", fontsize="small")

    return fig, (ax_shift, ax_delay)


def plot_calibration_fit_traces(t, traces, title=None, ax=None, log_scale=True):
    """Plot normalized calibration histograms and fitted trace for one channel."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    t = np.asarray(t, dtype=float)

    default_styles = {
        "data_for_fit": ("tab:blue", 1.0, 2.0),
        "ref_for_fit": ("tab:purple", 0.35, 1.5),
        "ref_common_delay_realigned": ("tab:purple", 0.9, 1.8),
        "irf_for_fit": ("tab:green", 0.35, 1.5),
        "irf_common_delay_realigned": ("tab:green", 0.9, 1.8),
        "data_fitted": ("tab:red", 1.0, 2.0),
    }

    for name, values in traces.items():
        if values is None:
            continue
        color, alpha, linewidth = default_styles.get(name, (None, 0.9, 1.5))
        ax.plot(
            t,
            normalize_histogram(values),
            label=name,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Normalized counts")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    return ax


def plot_channel_skew_correction(
    irf_no_alignment,
    irf_aligned,
    data_no_alignment,
    data_aligned,
    irf_reversed=None,
    data_reversed=None,
    fig=None,
    axs=None,
):
    """Compare summed IRF and data traces before and after channel-skew correction."""
    if fig is None or axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(irf_no_alignment, label="No correction", color="tab:blue", linewidth=1)
    if irf_reversed is not None:
        axs[0].plot(irf_reversed, label="Reversed correction", color="tab:orange", linewidth=1)
    axs[0].plot(irf_aligned, label="Corrected", color="black", linewidth=2)
    axs[0].set_title("Summed IRF")
    axs[0].set_xlabel("Time bin")
    axs[0].set_ylabel("Counts")
    axs[0].grid(True, linestyle=":")
    axs[0].legend()

    axs[1].plot(data_no_alignment, label="No correction", color="tab:blue", linewidth=1)
    if data_reversed is not None:
        axs[1].plot(data_reversed, label="Reversed correction", color="tab:orange", linewidth=1)
    axs[1].plot(data_aligned, label="Corrected", color="black", linewidth=2)
    axs[1].set_title("Summed data")
    axs[1].set_xlabel("Time bin")
    axs[1].set_ylabel("Counts")
    axs[1].set_yscale("log")
    axs[1].grid(True, linestyle=":")
    axs[1].legend()

    fig.tight_layout()
    return fig, axs


def plot_lifetime_histogram(
    lifetime,
    weights=None,
    lifetime_bounds=None,
    bins=500,
    ax=None,
    color="yellowgreen",
    edgecolor="darkgreen",
    gaussian=True,
    label=None,
    lifetime_axis="x",
    count_label="Pixel counts",
):
    """Plot a lifetime histogram with optional intensity weights and Gaussian overlay."""
    if ax is None:
        _, ax = plt.subplots()
    if lifetime_axis not in {"x", "y"}:
        raise ValueError("lifetime_axis must be 'x' or 'y'")

    values = np.asarray(lifetime, dtype=float).ravel()
    mask = np.isfinite(values)
    if weights is not None:
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.size != values.size:
            raise ValueError("weights must match lifetime size")
        mask &= np.isfinite(weights) & (weights > 0)
        weights = weights[mask]
    values = values[mask]

    hist, edges = np.histogram(values, bins=bins, range=lifetime_bounds, weights=weights)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if lifetime_axis == "x":
        ax.plot(centers, hist, drawstyle="steps-mid", color=edgecolor, linewidth=1.5, label=label)
        ax.fill_between(centers, hist, step="mid", facecolor=color, alpha=0.6)
    else:
        ax.plot(
            hist,
            centers,
            drawstyle="steps-mid",
            color=edgecolor,
            linewidth=1.5,
            label=label,
        )
        ax.fill_betweenx(centers, hist, step="mid", facecolor=color, alpha=0.6)

    if gaussian and values.size > 1:
        mean, rms = weighted_lifetime_stats(values, weights=weights)
        if np.isfinite(mean) and np.isfinite(rms) and rms > 0:
            curve = np.exp(-0.5 * ((centers - mean) / rms) ** 2)
            if np.sum(curve) > 0:
                curve = curve * (np.sum(hist) / np.sum(curve))
            if lifetime_axis == "x":
                ax.plot(
                    centers,
                    curve,
                    color="red",
                    linewidth=2,
                    label=f"mu={mean:.2f} ns, sigma={rms:.2f} ns",
                )
                ax.axvline(mean, color="red", linestyle="--", linewidth=1)
            else:
                ax.plot(
                    curve,
                    centers,
                    color="red",
                    linewidth=2,
                    label=f"mu={mean:.2f} ns, sigma={rms:.2f} ns",
                )
                ax.axhline(mean, color="red", linestyle="--", linewidth=1)
            ax.legend(loc="upper right")

    _format_histogram_axis(ax, lifetime_axis=lifetime_axis, count_label=count_label)
    return ax


def plot_lifetime_summary(
    intensity,
    lifetime,
    pxsize,
    pxdwelltime,
    lifetime_bounds=(1.0, 5.0),
    crop=0,
    threshold=0.05,
    bins=500,
    colormap="turbo",
    weighted_histogram=True,
    fig=None,
):
    """Show a FLIM map together with a thresholded lifetime histogram."""
    _require_graph_lib()

    intensity = np.asarray(intensity, dtype=float)
    lifetime = np.asarray(lifetime, dtype=float)
    if intensity.shape != lifetime.shape:
        raise ValueError("intensity and lifetime must have the same shape")

    display_intensity = crop_2d(intensity, crop)
    display_lifetime = crop_2d(lifetime, crop)
    hist_lifetime, hist_intensity, _ = threshold_lifetime_map(
        lifetime,
        intensity=intensity,
        threshold=threshold,
    )
    hist_weights = hist_intensity if weighted_histogram else None

    if fig is None:
        fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.25)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])

    gra.show_flim(
        display_intensity,
        display_lifetime,
        pxsize,
        pxdwelltime,
        lifetime_bounds=lifetime_bounds,
        fig=fig,
        ax=ax_map,
        colormap=colormap,
    )
    ax_map.set_title("Lifetime map")
    plot_lifetime_histogram(
        hist_lifetime,
        weights=hist_weights,
        lifetime_bounds=lifetime_bounds,
        bins=bins,
        ax=ax_hist,
    )
    ax_hist.set_title("Thresholded lifetime histogram")

    fig.tight_layout()
    return fig, (ax_map, ax_hist)


def plot_equalized_lifetime_summary(
    intensity,
    lifetime,
    pxsize,
    pxdwelltime,
    lifetime_bounds=(1.0, 8.0),
    crop=0,
    threshold=0.05,
    bins=500,
    colormap="turbo",
    equalization_reference=None,
    equalization_strength=4.0,
    equalization_bins=4096,
    colorbar_ticks=12,
    fig=None,
):
    """Compare linear and equalized FLIM hue axes and show the lifetime histogram."""
    _require_graph_lib()

    intensity = np.asarray(intensity, dtype=float)
    lifetime = np.asarray(lifetime, dtype=float)
    if intensity.shape != lifetime.shape:
        raise ValueError("intensity and lifetime must have the same shape")

    display_intensity = crop_2d(intensity, crop)
    display_lifetime = crop_2d(lifetime, crop)
    hist_lifetime, _, _ = threshold_lifetime_map(lifetime, intensity=intensity, threshold=threshold)
    if equalization_reference is None:
        equalization_reference = hist_lifetime

    if fig is None:
        fig = plt.figure(figsize=(9, 18))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.35, 1.55, 0.75], hspace=0.35)

    ax_linear = fig.add_subplot(gs[0, 0])
    gra.show_flim(
        display_intensity,
        display_lifetime,
        pxsize,
        pxdwelltime,
        lifetime_bounds=lifetime_bounds,
        fig=fig,
        ax=ax_linear,
        colormap=colormap,
    )
    ax_linear.set_title("Linear hue axis")

    ax_equalized = fig.add_subplot(gs[1, 0])
    flim.show_flim_equalized(
        display_intensity,
        display_lifetime,
        pxsize,
        pxdwelltime,
        lifetime_bounds=lifetime_bounds,
        fig=fig,
        ax=ax_equalized,
        colormap=colormap,
        equalization_reference=equalization_reference,
        equalization_strength=equalization_strength,
        equalization_bins=equalization_bins,
        colorbar_ticks=colorbar_ticks,
    )
    ax_equalized.set_title(
        f"Equalized hue axis (strength={equalization_strength:.1f}, bins={equalization_bins})"
    )

    ax_hist = fig.add_subplot(gs[2, 0])
    plot_lifetime_histogram(
        hist_lifetime,
        lifetime_bounds=lifetime_bounds,
        bins=bins,
        ax=ax_hist,
        gaussian=False,
    )
    ax_hist.set_title("Thresholded lifetime histogram")

    fig.tight_layout()
    return fig, (ax_linear, ax_equalized, ax_hist)
