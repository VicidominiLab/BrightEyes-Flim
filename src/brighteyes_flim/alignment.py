"""Alignment, fitting, and IRF estimation utilities."""

import warnings

import numpy as np
from scipy.ndimage import shift
try:
    from scipy import optimize as scipy_optimize
except ImportError:  # pragma: no cover - optional dependency
    scipy_optimize = None
try:
    import torch
    from torch.fft import fftn, ifftn, ifftshift
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    fftn = ifftn = ifftshift = None

from .tools_phasor import (
    estimate_lifetime_from_birfi,
    estimate_lifetime_from_circmean,
    estimate_lifetime_from_log,
)

__all__ = ["Alignment"]


class Alignment:
    """Static helpers for fitting, shifts, and IRF estimation."""

    @staticmethod
    def _require_torch():
        if torch is None or fftn is None or ifftn is None or ifftshift is None:
            raise ImportError("torch is required for Alignment methods")

    @staticmethod
    def _require_scipy_optimize():
        if scipy_optimize is None:
            raise ImportError("scipy is required for Alignment fitting methods")

    @staticmethod
    def to_numpy_1d(x, dtype=None):
        if torch is not None and torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        x = np.asarray(x)
        if dtype is not None:
            x = x.astype(dtype, copy=False)
        return x

    @staticmethod
    def to_torch_1d(x, dtype=None, device=None):
        Alignment._require_torch()
        if torch.is_tensor(x):
            tensor = x.detach().clone()
            if device is not None:
                tensor = tensor.to(device)
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            return tensor
        return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)

    @staticmethod
    def _normalize_histogram_1d(x, name="histogram"):
        hist = Alignment.to_numpy_1d(x, dtype=float)
        total = hist.sum()
        if not np.isfinite(total) or total <= 0:
            warnings.warn(
                f"{name} has a non-positive or non-finite sum; returning zeros without normalization",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.zeros_like(hist, dtype=float)
        return hist / total

    @staticmethod
    def _wrap_to_period(x, period, center=0.0):
        period = float(period)
        center = float(center)
        return center + np.mod(np.asarray(x) - center + 0.5 * period, period) - 0.5 * period

    @staticmethod
    def _normalize_circular_params(circular_params, n_params, lb, ub, p0):
        if circular_params is None:
            return {}

        if isinstance(circular_params, dict):
            circular_items = circular_params.items()
        else:
            circular_items = circular_params

        normalized = {}
        for item in circular_items:
            if isinstance(item, tuple) and len(item) == 2:
                idx, period = item
            else:
                idx, period = item, None

            idx = int(idx)
            if idx < 0 or idx >= n_params:
                raise IndexError(f"circular parameter index out of range: {idx}")

            if period is None:
                if np.isfinite(lb[idx]) and np.isfinite(ub[idx]):
                    period = ub[idx] - lb[idx]
                else:
                    raise ValueError(
                        f"missing circular period for parameter {idx}; provide it explicitly "
                        "or use finite bounds so it can be inferred"
                    )

            if float(period) <= 0:
                raise ValueError(f"circular period must be positive for parameter {idx}")

            if np.isfinite(lb[idx]) and np.isfinite(ub[idx]):
                center = 0.5 * (lb[idx] + ub[idx])
            else:
                center = float(p0[idx])

            normalized[idx] = {"period": float(period), "center": float(center)}

        return normalized

    @staticmethod
    def curve_fit_circular(
        f,
        xdata,
        ydata,
        p0=None,
        sigma=None,
        absolute_sigma=False,
        bounds=(-np.inf, np.inf),
        circular_curve_period=None,
        circular_curve_center=0.0,
        circular_params=None,
        maxfev=None,
        method="trf",
        **kwargs,
    ):
        """
        ``curve_fit``-like helper aware of circular data and circular parameters.

        Parameters
        ----------
        f : callable
            Model function ``f(xdata, *params)``.
        xdata, ydata : array-like
            Input coordinates and observations.
        p0, sigma, absolute_sigma, bounds, maxfev :
            Same meaning as in ``scipy.optimize.curve_fit``.
        circular_curve_period : float, optional
            If provided, residuals are wrapped onto a circle with this period.
        circular_curve_center : float, default 0.0
            Center of the wrapped residual interval.
        circular_params : dict or iterable, optional
            Circular parameters. A dict maps ``param_index -> period``. If an
            iterable is used, each item can be either ``index`` or
            ``(index, period)``. When only the index is given, the period is
            inferred from finite bounds.

        Returns
        -------
        popt, pcov : ndarray
            Best-fit parameters and covariance, like ``curve_fit``.
        """
        Alignment._require_scipy_optimize()

        xdata = np.asarray(xdata)
        ydata = Alignment.to_numpy_1d(ydata, dtype=float)

        if p0 is None:
            raise ValueError("curve_fit_circular requires an explicit p0")
        p0 = Alignment.to_numpy_1d(p0, dtype=float)

        n_params = len(p0)

        lb, ub = bounds
        lb = np.broadcast_to(np.asarray(lb, dtype=float), (n_params,)).copy()
        ub = np.broadcast_to(np.asarray(ub, dtype=float), (n_params,)).copy()

        circular_params = Alignment._normalize_circular_params(circular_params, n_params, lb, ub, p0)

        if sigma is None:
            sigma_array = None
        else:
            sigma_array = np.asarray(sigma, dtype=float)

        def wrap_params(params):
            params_wrapped = np.asarray(params, dtype=float).copy()
            for idx, spec in circular_params.items():
                params_wrapped[idx] = Alignment._wrap_to_period(
                    params_wrapped[idx],
                    period=spec["period"],
                    center=spec["center"],
                )
            return params_wrapped

        def residuals(params):
            params_eval = wrap_params(params)
            y_model = Alignment.to_numpy_1d(f(xdata, *params_eval), dtype=float)
            resid = y_model - ydata
            if circular_curve_period is not None:
                resid = Alignment._wrap_to_period(
                    resid,
                    period=circular_curve_period,
                    center=circular_curve_center,
                )
            if sigma_array is not None:
                resid = resid / sigma_array
            return resid

        p0_internal = p0.copy()
        for idx, spec in circular_params.items():
            p0_internal[idx] = Alignment._wrap_to_period(
                p0_internal[idx],
                period=spec["period"],
                center=spec["center"],
            )

        for idx in range(n_params):
            if np.isfinite(lb[idx]) and p0_internal[idx] < lb[idx]:
                p0_internal[idx] = lb[idx]
            if np.isfinite(ub[idx]) and p0_internal[idx] > ub[idx]:
                p0_internal[idx] = ub[idx]

        max_nfev = maxfev if maxfev is not None else kwargs.pop("max_nfev", None)
        result = scipy_optimize.least_squares(
            residuals,
            p0_internal,
            bounds=(lb, ub),
            method=method,
            max_nfev=max_nfev,
            **kwargs,
        )

        if not result.success:
            raise RuntimeError(f"circular curve fit failed: {result.message}")

        popt = wrap_params(result.x)

        _, s, vt = np.linalg.svd(result.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(result.jac.shape) * s[0] if s.size else 0.0
        s = s[s > threshold]
        vt = vt[: s.size]
        pcov = np.dot(vt.T / (s ** 2), vt) if s.size else np.full((n_params, n_params), np.inf)

        if not absolute_sigma and sigma_array is not None:
            dof = max(0, len(ydata) - n_params)
            if dof > 0:
                cost = 2.0 * result.cost
                pcov *= cost / dof

        return popt, pcov

    @staticmethod
    def model_data(
        t: np.ndarray,
        C: float,
        tau: float,
        period: float,
        shift_bins: float = 0.,
        mode: str = "binned",
    ) -> np.ndarray:
        """
        Periodic mono-exponential decay model.

        Units:
        - ``t`` and ``period`` are in nanoseconds.
        - ``tau`` is in nanoseconds.
        - ``C`` is the model amplitude.
        - ``shift_bins`` is applied in histogram-bin units.
        - ``mode`` selects either the center-sampled or bin-integrated model.
        """
        t_ns = Alignment.to_numpy_1d(t, dtype=float)
        C_norm = float(C)
        tau_ns = float(tau)
        period_ns = float(period)
        mode = str(mode).lower()
        shift_ns = float(shift_bins * (period_ns / len(t_ns)))

        # Center the decay on the middle bin before applying the sub-bin shift.
        t_local_ns = t_ns - shift_ns - period_ns - (period_ns / 2)

        if mode == "binned":
            if len(t_ns) < 2:
                raise ValueError("mode='binned' requires at least two time samples")

            dt_ns = float(t_ns[1] - t_ns[0])
            if not np.allclose(np.diff(t_ns), dt_ns):
                raise ValueError("mode='binned' requires uniformly spaced t values")

            t_start_ns = t_local_ns - 0.5 * dt_ns
            t_end_ns = t_start_ns + dt_ns
            u0_ns = np.mod(t_start_ns, period_ns)
            u1_ns = u0_ns + dt_ns
            denom = 1 - np.exp(-period_ns / tau_ns)

            model_hist = np.empty_like(t_ns, dtype=float)
            same_period = u1_ns <= period_ns

            model_hist[same_period] = (
                tau_ns
                * (
                    np.exp(-u0_ns[same_period] / tau_ns)
                    - np.exp(-u1_ns[same_period] / tau_ns)
                )
                / denom
            )

            if np.any(~same_period):
                wrapped_u1_ns = u1_ns[~same_period] - period_ns
                first_leg = (
                    tau_ns
                    * (
                        np.exp(-u0_ns[~same_period] / tau_ns)
                        - np.exp(-period_ns / tau_ns)
                    )
                    / denom
                )
                second_leg = (
                    tau_ns
                    * (
                        1.0 - np.exp(-wrapped_u1_ns / tau_ns)
                    )
                    / denom
                )
                model_hist[~same_period] = first_leg + second_leg
        elif mode == "sampled":
            model_hist = (
                np.exp(-(np.mod(t_local_ns, period_ns)) / tau_ns)
                / (1 - np.exp(-period_ns / tau_ns))
            )
        else:
            raise ValueError(f"Unsupported model_data mode: {mode}. Supported sampled, binned")

        model_hist = C_norm * model_hist / model_hist.sum()

        return model_hist

    @staticmethod
    def rectangular_IRF(t, dt):
        t_ns = Alignment.to_numpy_1d(t, dtype=float)
        dt_ns = float(dt)
        offset_ns = (t_ns.max() - t_ns.min()) / 2
        return np.where((t_ns >= offset_ns - dt_ns) & (t_ns <= offset_ns + dt_ns), 1.0, 0.0)

    @staticmethod
    def pad_tensor(x: torch.Tensor, pad_left: int, pad_right: int, dim: int, mode: str = "reflect"):
        Alignment._require_torch()
        if pad_left == 0 and pad_right == 0:
            return x

        length = x.shape[dim]

        if mode == "reflect":
            left_idx = torch.arange(pad_left, 0, -1, device=x.device)
            right_idx = torch.arange(length - 2, length - pad_right - 2, -1, device=x.device)
        elif mode == "replicate":
            left_idx = torch.zeros(pad_left, dtype=torch.long, device=x.device)
            right_idx = torch.full((pad_right,), length - 1, dtype=torch.long, device=x.device)
        elif mode == "constant":
            pad_shape = list(x.shape)
            pad_shape[dim] = pad_left + pad_right
            constant_pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
            return torch.cat(
                [
                    constant_pad.narrow(dim, 0, pad_left),
                    x,
                    constant_pad.narrow(dim, pad_left, pad_right),
                ],
                dim=dim,
            )
        else:
            raise ValueError(f"Unsupported padding mode: {mode}")

        pad_left_tensor = x.index_select(dim, left_idx)
        pad_right_tensor = x.index_select(dim, right_idx)
        return torch.cat([pad_left_tensor, x, pad_right_tensor], dim=dim)

    @staticmethod
    def median_filter(x: torch.Tensor, window_size=3, dims=None, mode="reflect"):
        Alignment._require_torch()
        if dims is None:
            dims = list(range(x.ndim))

        if isinstance(window_size, int):
            window_size = [window_size] * len(dims)
        elif len(window_size) != len(dims):
            raise ValueError("window_size must be scalar or match len(dims)")

        for w in window_size:
            if w % 2 == 0:
                raise ValueError(f"All window sizes must be odd, got {w}")

        out = x
        for d, w in zip(dims, window_size):
            pad_left = (w - 1) // 2
            pad_right = w // 2
            out = Alignment.pad_tensor(out, pad_left, pad_right, d, mode=mode)
            out = out.unfold(d, w, 1).median(dim=-1).values

        return out

    @staticmethod
    def partial_convolution_fft(volume: torch.Tensor, kernel: torch.Tensor, dim1: str = 'ijk', dim2: str = 'jkl',
                                axis: str = 'jk', fourier: tuple = (False, False)):
        Alignment._require_torch()
        dim3 = dim1 + dim2
        dim3 = ''.join(sorted(set(dim3), key=dim3.index))

        dims = [dim1, dim2, dim3]
        axis_list = [[d.find(c) for c in axis] for d in dims]

        volume_fft = fftn(volume, dim=axis_list[0]) if not fourier[0] else volume
        kernel_fft = fftn(kernel, dim=axis_list[1]) if not fourier[1] else kernel

        conv = torch.einsum(f'{dim1},{dim2}->{dim3}', volume_fft, kernel_fft)
        conv = ifftn(conv, dim=axis_list[2])
        conv = ifftshift(conv, dim=axis_list[2])
        return torch.real(conv)

    @staticmethod
    def IRF_from_data_deconvolution(ref_data, t, C_R, tau_R, period, iterations=30, eps=1e-8, regularization=3):
        """
        Estimate the IRF from a reference histogram.

        Units:
        - ``t`` and ``period`` are in nanoseconds.
        - ``tau_R`` is in nanoseconds.
        - ``C_R`` is the reference-model amplitude.
        - ``period`` is the excitation period in the same time units as ``t``.
        """
        Alignment._require_torch()
        ref_hist = Alignment.to_torch_1d(ref_data, dtype=torch.float64)
        t_ns = Alignment.to_numpy_1d(t, dtype=float)
        C_ref = float(C_R)
        tau_ref_ns = float(tau_R)
        period_ns = float(period)

        irf_est = torch.ones_like(ref_hist)

        kernel = Alignment.to_torch_1d(Alignment.model_data(t=t_ns, C=C_ref, tau=tau_ref_ns, period=period_ns))

        kernel_t = kernel.clone().flip(0)

        kernel = fftn(kernel, dim=0)
        kernel_t = fftn(kernel_t, dim=0)

        y = torch.clamp(ref_hist, min=0)

        for _ in range(iterations):
            conv = Alignment.partial_convolution_fft(irf_est, kernel, dim1="t", dim2="t", axis="t", fourier=(False, True))
            conv = torch.clamp(conv, min=eps)
            relative_blur = y / conv
            correction = Alignment.partial_convolution_fft(
                relative_blur, kernel_t, dim1='t', dim2='t', axis='t', fourier=(0, 1)
            )
            irf_est = irf_est * correction
            irf_est = torch.clamp(irf_est, min=0)
            if regularization > 1:
                irf_est = Alignment.median_filter(irf_est, window_size=regularization, dims=[0], mode='replicate')

        return irf_est

    @staticmethod
    def linear_shift(data, shift_value, cyclic=True):
        xp = np.arange(0, data.shape[0])
        fp = data.copy()
        x = np.arange(0, data.shape[0]) - shift_value
        if cyclic:
            x = np.mod(x, data.shape[0])
        return np.interp(x, xp, fp)

    @staticmethod
    def fit_data_with_ref_or_irf(
        t,
        data,
        period,
        ref=None,
        tau_ref=None,
        irf=None,
        C_ref=1.0,
        irf_output="original",
        shift_output=None,
        fit_type="circular",
        mode="irf_shift",
        initial_tau=None,
        initial_dT=None,
        initial_C=None,
        force_C_normalized=False,
        irf_iterations=30,
        eps=1e-8,
        regularization=3,
    ):
        """
        Fit ``data`` using either ``ref`` + ``tau_ref`` or a directly provided ``irf``.

        Parameters
        ----------
        t, data, period :
            Same units and meaning as in ``perform_fit_data``.
        ref : array-like, optional
            Reference decay used to estimate the IRF through
            ``IRF_from_data_deconvolution``. When ``irf`` is not given,
            ``tau_ref`` can be provided explicitly or estimated from ``ref``.
        tau_ref : float or str, optional
            Lifetime of ``ref`` in nanoseconds. If omitted, it is estimated
            from ``ref`` with ``estimate_lifetime_from_log``. String selectors
            are also accepted:
            - ``"circmean"`` uses ``estimate_lifetime_from_circmean``
            - ``"birfi"`` uses ``estimate_lifetime_from_birfi``
            - ``"log"`` uses ``estimate_lifetime_from_log``
        irf : array-like, optional
            Directly provided IRF. When this is given, ``tau_ref`` is ignored.
        C_ref : float, default 1.0
            Reference amplitude used during IRF estimation from ``ref``.
        irf_output : {"original", "shifted"}, default "original"
            Controls the returned ``irf`` in the result dictionary.
            - ``"original"`` returns the estimated IRF (if ``ref`` was used) or
              the provided IRF (if ``irf`` was used).
            - ``"shifted"`` returns that IRF after ``linear_shift(..., dT)``.
        shift_output : {None, "ref", "reference", "data"}, default None
            Optionally returns an additional shifted histogram:
            - ``"ref"`` / ``"reference"`` returns ``ref_shifted`` using ``+dT``.
            - ``"data"`` returns ``data_shifted`` using ``-dT``.

        Returns
        -------
        dict
            Result dictionary with at least:
            - ``C``: fitted normalized amplitude
            - ``tau_ref``: reference lifetime actually used for IRF estimation,
              in nanoseconds, or ``None`` when no reference lifetime was used
            - ``dT``: fitted shift in bins
            - ``dT_ns``: fitted shift in nanoseconds
            - ``tau``: fitted lifetime in nanoseconds
            - ``irf``: returned IRF according to ``irf_output``
            - ``fit``: fitted histogram
            - ``cov``: covariance matrix from ``perform_fit_data``
            - ``irf_source``: ``"estimated_from_ref"`` or ``"provided"``
            When requested, the dictionary also includes ``ref_shifted`` or
            ``data_shifted``.

            All histogram outputs returned by this helper are normalized to
            unit sum. This function does not return unnormalized ``data``,
            ``ref``, or ``irf`` histograms.
        """
        t_ns = Alignment.to_numpy_1d(t, dtype=float)
        data_hist = Alignment.to_numpy_1d(data, dtype=float)

        if len(t_ns) == 0:
            raise ValueError("t must contain at least one sample")
        if len(t_ns) != len(data_hist):
            raise ValueError("t and data must have the same 1D length")

        period_ns = float(period)
        dt_ns = float(period_ns / len(t_ns))

        data_hist_norm = Alignment._normalize_histogram_1d(data_hist, name="data")

        if (ref is None) == (irf is None):
            raise ValueError("provide exactly one of ref or irf")

        irf_output = str(irf_output).lower()
        if irf_output in {"estimated", "input", "provided", "unshifted"}:
            irf_output = "original"
        if irf_output not in {"original", "shifted"}:
            raise ValueError("irf_output must be 'original' or 'shifted'")

        if shift_output is not None:
            shift_output = str(shift_output).lower()
        if shift_output == "reference":
            shift_output = "ref"
        if shift_output not in {None, "ref", "data"}:
            raise ValueError("shift_output must be None, 'ref', 'reference', or 'data'")

        used_tau_ref_ns = None

        if irf is None:
            ref_hist = Alignment.to_numpy_1d(ref, dtype=float)
            if len(ref_hist) != len(t_ns):
                raise ValueError("t and ref must have the same 1D length")

            if tau_ref is None:
                tau_ref_mode = "log"
            elif isinstance(tau_ref, str):
                tau_ref_mode = tau_ref.strip().lower()
            else:
                tau_ref_mode = None

            if tau_ref_mode is None:
                tau_ref_ns = float(tau_ref)
            elif tau_ref_mode in {"log", "estimate_lifetime_from_log"}:
                tau_ref_result = estimate_lifetime_from_log(
                    data_hist=ref_hist,
                    t_ns=t_ns,
                    dt_ns=dt_ns,
                    nbin=len(t_ns),
                    period_ns=period_ns,
                )
                tau_ref_ns = tau_ref_result[0]
            elif tau_ref_mode in {"circmean", "estimate_lifetime_from_circmean"}:
                repetition_rate_MHz = 1e3 / period_ns
                t0_ns = float((int(np.argmax(ref_hist)) + 0.5) * dt_ns)
                tau_ref_ns = estimate_lifetime_from_circmean(
                    ref_hist,
                    t0_ns=t0_ns,
                    repetition_rate_MHz=repetition_rate_MHz,
                )
            elif tau_ref_mode in {"birfi", "estimate_lifetime_from_birfi"}:
                tau_ref_ns = estimate_lifetime_from_birfi(t_ns, ref_hist)
            else:
                try:
                    tau_ref_ns = float(tau_ref)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "tau_ref must be a float, None, 'log', 'circmean', or 'birfi'"
                    ) from exc

            if tau_ref_ns is None:
                raise ValueError("unable to estimate tau_ref from ref")
            tau_ref_ns_array = np.asarray(tau_ref_ns, dtype=float)
            if tau_ref_ns_array.size != 1:
                raise ValueError("tau_ref estimator must return a scalar lifetime")
            tau_ref_ns = float(tau_ref_ns_array.reshape(-1)[0])
            if not np.isfinite(tau_ref_ns) or tau_ref_ns <= 0:
                raise ValueError("unable to estimate a valid positive tau_ref from ref")
            used_tau_ref_ns = tau_ref_ns

            ref_hist_norm = Alignment._normalize_histogram_1d(ref_hist, name="ref")
            irf_hist = np.asarray(
                Alignment.IRF_from_data_deconvolution(
                    ref_hist_norm,
                    t_ns,
                    C_ref,
                    tau_ref_ns,
                    period_ns,
                    iterations=irf_iterations,
                    eps=eps,
                    regularization=regularization,
                ),
                dtype=float,
            )
            irf_source = "estimated_from_ref"
        else:
            ref_hist_norm = None
            irf_hist = Alignment.to_numpy_1d(irf, dtype=float)
            if len(irf_hist) != len(t_ns):
                raise ValueError("t and irf must have the same 1D length")
            irf_source = "provided"

        irf_hist_norm = Alignment._normalize_histogram_1d(irf_hist, name="irf")

        fit_result, fit_cov = Alignment.perform_fit_data(
            t_ns,
            data_hist,
            irf_hist_norm,
            period_ns,
            initial_tau=initial_tau,
            initial_dT=initial_dT,
            initial_C=initial_C,
            mode=mode,
            fit_type=fit_type,
            force_C_normalized=force_C_normalized,
        )

        dT_bins = float(fit_result["dT"])
        tau_ns = float(fit_result["tau"])
        C_value = float(fit_result["C"])
        fit_is_valid = np.isfinite(C_value) and np.isfinite(dT_bins) and np.isfinite(tau_ns)

        returned_irf = irf_hist_norm.copy()
        if irf_output == "shifted" and fit_is_valid:
            returned_irf = Alignment._normalize_histogram_1d(
                Alignment.linear_shift(returned_irf, dT_bins, cyclic=True),
                name="shifted irf",
            )
        elif irf_output == "shifted":
            returned_irf = np.zeros_like(irf_hist_norm, dtype=float)

        if fit_is_valid:
            fitted_hist = Alignment._normalize_histogram_1d(
                Alignment.fit_model_data(
                    t_ns,
                    C_value,
                    dT_bins,
                    tau_ns,
                    irf=irf_hist_norm,
                    period=period_ns,
                    mode=mode,
                ),
                name="fit",
            )
        else:
            fitted_hist = np.zeros_like(data_hist_norm, dtype=float)

        result = {
            "C": C_value,
            "tau_ref": used_tau_ref_ns,
            "dT": dT_bins,
            "dT_ns": dT_bins * dt_ns,
            "tau": tau_ns,
            "irf": returned_irf,
            "fit": fitted_hist,
            "cov": fit_cov,
            "irf_source": irf_source,
        }

        if shift_output == "ref":
            if ref_hist_norm is None:
                raise ValueError("shift_output='ref' requires ref to be provided")
            if fit_is_valid:
                result["ref_shifted"] = Alignment._normalize_histogram_1d(
                    Alignment.linear_shift(ref_hist_norm, dT_bins, cyclic=True),
                    name="shifted ref",
                )
            else:
                result["ref_shifted"] = np.zeros_like(ref_hist_norm, dtype=float)
        elif shift_output == "data":
            if fit_is_valid:
                result["data_shifted"] = Alignment._normalize_histogram_1d(
                    Alignment.linear_shift(data_hist_norm, -dT_bins, cyclic=True),
                    name="shifted data",
                )
            else:
                result["data_shifted"] = np.zeros_like(data_hist_norm, dtype=float)

        return result

    @staticmethod
    def fit_model_data(t, C, dT, tau, irf, period, mode="irf_shift"):
        """
        Convolve the mono-exponential model with a shifted IRF.

        Units:
        - ``t`` and ``period`` are in nanoseconds.
        - ``tau`` is in nanoseconds.
        - ``dT`` is in histogram bins, because it is applied through
          ``scipy.ndimage.shift(..., mode='grid-wrap')``.
        """
        t_ns = Alignment.to_numpy_1d(t, dtype=float)
        irf_hist = Alignment.to_numpy_1d(irf, dtype=float)
        C_norm = float(C)
        dT_bins = float(dT) 
        tau_ns = float(tau)
        period_ns = float(period)

        if mode=="model_shift":
            pure_model_hist = Alignment.model_data(t=t_ns, C=C_norm, tau=tau_ns, period=period_ns, shift_bins=dT_bins)                                                  
            irf_shifted_hist = irf_hist   
        elif mode=="irf_shift":
            pure_model_hist = Alignment.model_data(t=t_ns, C=C_norm, tau=tau_ns, period=period_ns)                                                  
            irf_shifted_hist = shift(irf_hist, dT_bins, order=1, mode='grid-wrap')
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported model_shift, irf_shift")
        
        pure_model_hist = pure_model_hist / pure_model_hist.sum()
        irf_shifted_hist = irf_shifted_hist / irf_shifted_hist.sum()


        pure_model_hist = Alignment.to_torch_1d(pure_model_hist)
        irf_shifted_hist = Alignment.to_torch_1d(irf_shifted_hist)
        return Alignment.partial_convolution_fft(
            pure_model_hist, irf_shifted_hist, dim1='t', dim2='t', axis='t', fourier=(0, 0)
        )

    @staticmethod
    def estimate_peak_dT_bins(data, irf):
        """
        Estimate a direct circular shift seed from the data/IRF peak locations.

        The returned shift is wrapped to the public ``[-nbin/2, nbin/2)`` bin
        convention used by the fitting helpers.
        """
        data_hist = Alignment.to_numpy_1d(data, dtype=float)
        irf_hist = Alignment.to_numpy_1d(irf, dtype=float)

        if len(data_hist) != len(irf_hist):
            raise ValueError("data and irf must have the same 1D length")

        nbin = len(data_hist)
        data_peak_bin = int(np.argmax(data_hist))
        irf_peak_bin = int(np.argmax(irf_hist))

        return float(
            Alignment._wrap_to_period(
                data_peak_bin - irf_peak_bin,
                period=float(nbin),
                center=0.0,
            )
        )

    @staticmethod
    def perform_fit_data(
        t,
        data,
        irf,
        period,
        initial_tau=None,
        initial_dT=None,
        initial_C=None,
        irf_min=1e-5,
        mode="irf_shift",
        fit_type="circular",
        force_C_normalized=False,
    ):
        """
        Fit ``data`` with ``fit_model_data``.

        Unit contract:
        - ``t`` and ``period`` are in nanoseconds.
        - ``tau`` / ``initial_tau`` are in nanoseconds.
        - ``dT`` / ``initial_dT`` are in bins.
        - ``fit_type`` can be ``"circular"`` or ``"curve_fit"``.
        - ``force_C_normalized=True`` keeps ``C`` fixed to ``1.0`` and only
          fits ``dT`` and ``tau``.
        - when ``initial_dT is None``, the fitter first pre-shifts the IRF by a
          direct peak-based seed and then fits only the residual shift around
          zero. This avoids the circular-boundary trap near ``+-nbin/2`` while
          keeping the public returned ``dT`` in the same convention.
        - when ``initial_dT`` is provided, it is used directly as the initial
          guess for the public ``dT`` parameter.
        - returned ``C`` is a normalized 0..1 amplitude because the data is
          normalized by its sum and the IRF by its sum.
        """
        t_ns = Alignment.to_numpy_1d(t, dtype=float)
        data_hist = Alignment.to_numpy_1d(data, dtype=float)
        irf_hist = Alignment.to_numpy_1d(irf, dtype=float)




        if len(t_ns) != len(data_hist) or len(t_ns) != len(irf_hist):
            raise ValueError("t, data, and irf must have the same 1D length")

        data_sum = data_hist.sum()
        irf_sum = irf_hist.sum()

        if not np.isfinite(data_sum) or data_sum <= 0:
            warnings.warn(
                "data histogram has a non-positive or non-finite sum; skipping fit and returning NaNs",
                RuntimeWarning,
                stacklevel=2,
            )
            return {
                "C": np.nan,
                "dT": np.nan,
                "tau": np.nan,
            }, np.full((3, 3), np.nan, dtype=float)
        if not np.isfinite(irf_sum) or irf_sum <= 0:
            warnings.warn(
                "irf histogram has a non-positive or non-finite sum; skipping fit and returning NaNs",
                RuntimeWarning,
                stacklevel=2,
            )
            return {
                "C": np.nan,
                "dT": np.nan,
                "tau": np.nan,
            }, np.full((3, 3), np.nan, dtype=float)

        data_hist_norm = data_hist / data_sum
        irf_hist_norm = irf_hist / irf_sum
        Alignment._require_scipy_optimize()
        nbin = len(t_ns)
        dT_seed_bins = None
        fit_irf_hist_norm = irf_hist_norm
        fit_initial_dT = initial_dT

        if initial_dT is None:
            dT_seed_bins = Alignment.estimate_peak_dT_bins(data_hist_norm, irf_hist_norm)
            fit_irf_hist_norm = np.asarray(
                shift(irf_hist_norm, dT_seed_bins, order=1, mode="grid-wrap"),
                dtype=float,
            )
            fit_irf_hist_norm = fit_irf_hist_norm / fit_irf_hist_norm.sum()
            fit_initial_dT = 0.0

        fit_lambda = lambda t_ns_fit, C_norm, dT_bins, tau_ns: Alignment.fit_model_data(
            t_ns_fit,
            C_norm,
            dT_bins,
            tau_ns,
            irf=fit_irf_hist_norm,
            period=period,
            mode=mode,
        )
        fit_lambda_numpy = lambda t_ns_fit, C_norm, dT_bins, tau_ns: Alignment.to_numpy_1d(
            fit_lambda(t_ns_fit, C_norm, dT_bins, tau_ns),
            dtype=float,
        )


        initial_guess = [1.0, 0.0, 1.0]

        if initial_C is not None:
            initial_guess[0] = initial_C
        if fit_initial_dT is not None:
            initial_guess[1] = fit_initial_dT
        if initial_tau is not None:
            initial_guess[2] = initial_tau

        
        sigma = np.sqrt(data_hist)
        tau_lower_bound = float(irf_min)
        if tau_lower_bound <= 0:
            raise ValueError("irf_min must be positive")
        fit_type = str(fit_type).lower()

        if force_C_normalized:
            fit_lambda_fixed_c = lambda t_ns_fit, dT_bins, tau_ns: fit_lambda_numpy(
                t_ns_fit, 1.0, dT_bins, tau_ns
            )
            initial_guess_fixed_c = initial_guess[1:]
            fit_bounds_fixed_c = ([-nbin / 2, tau_lower_bound], [nbin / 2, t_ns.max()])

            if fit_type == "circular":
                popt_fixed_c, conv_fixed_c = Alignment.curve_fit_circular(
                    fit_lambda_fixed_c,
                    t_ns,
                    data_hist_norm,
                    sigma=sigma,
                    p0=initial_guess_fixed_c,
                    bounds=fit_bounds_fixed_c,
                    circular_params={0: float(nbin)},
                    circular_curve_period=float(nbin),
                    maxfev=600000,
                )
            elif fit_type == "curve_fit":
                popt_fixed_c, conv_fixed_c = scipy_optimize.curve_fit(
                    fit_lambda_fixed_c,
                    t_ns,
                    data_hist_norm,
                    sigma=sigma,
                    p0=initial_guess_fixed_c,
                    bounds=fit_bounds_fixed_c,
                    maxfev=600000,
                )
            else:
                raise ValueError(f"Unsupported fit_type: {fit_type}. Supported circular, curve_fit")

            popt = np.array([1.0, popt_fixed_c[0], popt_fixed_c[1]], dtype=float)
            conv = np.full((3, 3), np.nan, dtype=float)
            conv[1:, 1:] = conv_fixed_c
        else:
            fit_bounds = ([0.0, -nbin / 2, tau_lower_bound], [np.inf, nbin / 2, t_ns.max()])

            if fit_type == "circular":
                popt, conv = Alignment.curve_fit_circular(
                    fit_lambda_numpy,
                    t_ns,
                    data_hist_norm,
                    sigma=sigma,
                    p0=initial_guess,
                    bounds=fit_bounds,
                    circular_params={1: float(nbin)},
                    maxfev=600000,
                )
            elif fit_type == "curve_fit":
                popt, conv = scipy_optimize.curve_fit(
                    fit_lambda_numpy,
                    t_ns,
                    data_hist_norm,
                    sigma=sigma,
                    p0=initial_guess,
                    bounds=fit_bounds,
                    maxfev=600000,
                )
            else:
                raise ValueError(f"Unsupported fit_type: {fit_type}. Supported circular, curve_fit")

        if dT_seed_bins is not None:
            popt = np.asarray(popt, dtype=float).copy()
            popt[1] = float(
                Alignment._wrap_to_period(
                    dT_seed_bins + popt[1],
                    period=float(nbin),
                    center=0.0,
                )
            )

        return {"C": popt[0], "dT": popt[1], "tau": popt[2]}, conv

    @staticmethod
    def phasor_delay_from_hist(hist, period_ns, harmonic=1):
        hist = Alignment.to_numpy_1d(hist, dtype=float)
        phasor_value = np.fft.fft(hist / hist.sum())[harmonic].conj()
        phase_rad = np.mod(np.angle(phasor_value), 2 * np.pi)
        delay_ns = phase_rad / (2 * np.pi) * period_ns
        return phasor_value, phase_rad, delay_ns

    @staticmethod
    def hist_for_plot(hist):
        return Alignment.to_numpy_1d(hist)
