from matplotlib import colors
import numpy as np
from matplotlib.pyplot import gca
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
try:
    from scipy import optimize as scipy_optimize
except ImportError:  # pragma: no cover - optional dependency
    scipy_optimize = None
from tqdm.auto import tqdm
import math
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import brighteyes_ism.analysis.Graph_lib as gra
import h5py
import matplotlib.pyplot as plt
import os
try:
    import torch
    from torch.fft import fftn, ifftn, ifftshift
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    fftn = ifftn = ifftshift = None

import brighteyes_ism.dataio.mcs as mcs


class FlimData:

    def __init__(self, data_path: str = None, data_path_irf: str = None,
                 freq_exc: float = 41.48e6, correction_coeff: complex = None, step_size: int = None,
                 sub_image_dim: int = 100, pre_filter: str = None):
        '''

        :param data_path:
        :param freq_exc:
        :param correction_coeff:
        :param pre_filter: None, "normalize", float = noise removal threshold on normalized data - Usefully for IRF
        '''

        #

        self.phasor_laser = 0.0j
        self.phasor_laser_irf = 0.0j
        self.freq_exc = freq_exc

        self.sub_image_dim = sub_image_dim
        # The source data
        self.data = None

        self.data_irf = None

        # sliced data along xy dimensions
        self.sliced_data = None

        self.bin_number = 81

        if data_path is not None:
            with h5py.File(data_path, "r") as hf:
                if "data" in hf.keys():
                    img = hf["data"]
                    self.bin_number = img.shape[-2]
                    self.channel_number = img.shape[-1]
                elif "dataset_1" in hf.keys():
                    img = hf["dataset_1"]
                    self.bin_number = img.shape[-2]
                    self.channel_number = img.shape[-1]

        self.data_hist = np.zeros((self.bin_number, self.channel_number))

        self.data_hist_irf = np.zeros((self.bin_number, self.channel_number))

        # The laser data histogram (26th channel)
        self.data_laser_hist = np.zeros(self.bin_number)
        self.data_laser_hist_irf = np.zeros(self.bin_number)

        # initialize dataset to store histograms aligned for each pixel

        # The sliced laser data histogram (26th channel)
        self.data_laser_hist_non_sliced = None

        # The metadata
        self.metadata = None

        self.metadata_irf = None

        # The Calibration reference phasor (complex notation)
        self.calib_ref = None

        # Full phasors
        self.phasors = None

        # Phasor of the global histogram (for each ch.)
        self.phasors_global = None

        self.phasors_global_irf = None

        # Shift term for re-alignment in calibration
        self.shift_term = 0

        self.correction_coeff = None

        self.step_size = step_size

        if data_path is not None:
            self.load_data_irf(data_path_irf)
            self.load_data(data_path)
            if pre_filter is not None:

                if isinstance(pre_filter, str):
                    if pre_filter == "normalize":
                        self.data_hist = ((self.data_hist - np.nanmin(self.data_hist.T, axis=1)) /
                                          (np.nanmax(self.data_hist.T, axis=1) - np.nanmin(self.data_hist.T, axis=1)))
                        self.data_hist_irf = ((self.data_hist_irf - np.nanmin(self.data_hist_irf.T, axis=1)) /
                                              (np.nanmax(self.data_hist_irf.T, axis=1) - np.nanmin(self.data_hist_irf.T,
                                                                                                   axis=1)))
                elif type(pre_filter) is int or type(pre_filter) is float:
                    threshold = pre_filter
                    nnn = self.data_hist / np.max(self.data_hist, axis=0)
                    nnn[nnn < threshold] = 1. / np.max(self.data_hist[nnn < threshold], axis=0)
                    self.data_hist = nnn
                    mmm = self.data_hist_irf / np.max(self.data_hist_irf, axis=0)
                    mmm[mmm < threshold] = 1. / np.max(self.data_hist_irf[mmm < threshold], axis=0)
                    self.data_hist_irf = mmm
                    print("used pre_filter ", threshold)
                else:
                    pre_filter = None
                    print("Wrong pre_filter parameter", pre_filter, type(pre_filter))
            self.calculate_phasor_global_irf()
            self.calculate_phasor_global()
            print("phasor global", self.phasors_global)
            print("phasor global irf", self.phasors_global_irf)
            self.calculate_phasor_laser()
            self.calculate_phasor_laser_irf()
            print("phasor laser", self.phasor_laser)
            print("phasor laser irf", self.phasor_laser_irf)

        if correction_coeff is not None:
            if isinstance(correction_coeff, list) or \
                    isinstance(correction_coeff, np.ndarray):
                self.correction_coeff = correction_coeff[0] + 1j * correction_coeff[1]
            if isinstance(correction_coeff, complex):
                self.correction_coeff = correction_coeff

        # print("Global", self.phasors_global)
        # print("Laser", self.phasor_laser)

    def load_data(self, data_path: str):
        self.data = h5py.File(data_path)
        self.metadata = mcs.metadata_load(data_path)  # data_format = 'h5'
        data_extra, _ = mcs.load(data_path, key="data_channels_extra")
        data_laser = data_extra[:, :, :, :, :, 1]
        image = self.data["data"]
        self.data_laser_hist = np.sum(data_laser, axis=(0, 1, 2, 3))
        self.data_hist = np.sum(image, axis=(0, 1, 2, 3))

    def load_data_irf(self, data_path_irf: str):
        self.data_irf = h5py.File(data_path_irf)
        self.metadata_irf = mcs.metadata_load(data_path_irf)  # data_format = 'h5'
        data_extra_irf, _ = mcs.load(data_path_irf, key="data_channels_extra")
        data_laser_irf = data_extra_irf[:, :, :, :, :, 1]
        image_irf = self.data_irf["data"]
        self.data_hist_irf = np.sum(image_irf, axis=(0, 1, 2, 3))
        self.data_laser_hist_irf = np.sum(data_laser_irf, axis=(0, 1, 2, 3))

    def calculate_phasor_global(self):
        self.phasors_global = calculate_phasor(data_hist=self.data_hist)
        return self.phasors_global

    def calculate_phasor_global_irf(self):
        self.phasors_global_irf = calculate_phasor(data_hist=self.data_hist_irf)
        return self.phasors_global_irf

    def calculate_phasor_laser(self):
        self.phasor_laser = calculate_phasor(self.data_laser_hist)
        return self.phasor_laser

    def calculate_phasor_laser_irf(self):
        self.phasor_laser_irf = calculate_phasor(self.data_laser_hist_irf)
        return self.phasor_laser_irf

    # def apply_laser_phasor(self):
    #     if self.phasor_laser is None:
    #         self.calculate_phasor_laser()
    #     normalized_phase = self.phasor_laser / abs(self.phasor_laser)
    #     self.phasors = self.phasors / normalized_phase
    #
    #     if self.phasors_global is None:
    #         self.phasors_global = calculate_phasor_global()
    #     self.phasors_global = self.phasors_global / normalized_phase
    #
    #     print("applied_laser_phasor")
    #
    #     return self.phasors

    def calculate_correction_coeff(self, dfd_freq=41.48e6, tau=10e-9):
        dfd_time = 1. / dfd_freq

        ppp = self.phasors_global / abs(self.phasors_global)

        theta = 2 * np.pi * tau / dfd_time
        DeltaM = np.abs(self.phasors_global) / np.abs(np.cos(theta))

        DeltaPhi = np.exp(1j * theta)
        coeff = 1. / (ppp * DeltaM * DeltaPhi)

        self.correction_coeff = coeff

        return self.correction_coeff

    def phasor_global_corrected(self, coeff=1., laser_correction=True):
        if laser_correction:
            return self.phasors_global * coeff * np.exp(1j * (-np.angle(self.phasor_laser)))
        else:
            return self.phasors_global * coeff

    def phasor_global_corrected_irf(self, coeff=1., laser_correction=True):
        if laser_correction:
            return self.phasors_global_irf * coeff * np.exp(1j * (-np.angle(self.phasor_laser)))
        else:
            return self.phasors_global_irf * coeff


class Alignment:

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
        if total <= 0:
            raise ValueError(f"{name} must have a positive sum")
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

    # @staticmethod
    # def model_dataGGG(t, C, tau, T):
    #     offset = t[len(t) // 2]
    #     x = t - offset

    #     b = np.exp(-T / tau)
    #     denom = 1.0 - b

    #     model = np.empty_like(t, dtype=float)
    #     mask = x >= 0
    #     model[mask] = C * np.exp(-x[mask] / tau) / denom
    #     model[~mask] = C * np.exp(-(x[~mask] + T) / tau) / denom
    #     return model

    @staticmethod    
    def model_data(
        t: np.ndarray,
        C: float,
        tau: float,
        period: float,
        shift_bins: float = 0.,
        mode: str = ""
        "binned",
    ) -> np.ndarray:
        """
        Periodic mono-exponential decay model.

        Units:
        - ``t`` and ``period`` are in nanoseconds.
        - ``tau`` is in nanoseconds.
        - ``C`` is the model amplitude. When the fit helpers are used, it is a
        - ``shift_bins`` is in histogram bins, because it is applied through
          normalized 0..1 scale factor.
        - ``mode`` can be ``"sampled"`` for the current center-sampled model or
          ``"binned"`` to integrate the model over each histogram bin.
        """
        t_ns = Alignment.to_numpy_1d(t, dtype=float)
        C_norm = float(C)
        tau_ns = float(tau)
        period_ns = float(period)
        mode = str(mode).lower()
        shift_ns = float(shift_bins * (period_ns / len(t_ns)))

        t_local_ns = t_ns - shift_ns - period - (period_ns / 2)
        '''
        This shifts the time axis so the model is centered on the middle bin,
        then applies the shift_bins as a sub-bin shift, and finally shifts back
        by one period to ensure the model is zero at the start of the histogram.
        '''

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

        model_hist = C_norm * model_hist / model_hist.sum()  # normalize to unit area so C is a simple amplitude factor

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
            #conv = Alignment.partial_convolution_fft(irf_est, kernel, dim1='t', dim2='t', axis='t', fourier=(0, 1))
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
            if tau_ref is not None:
                print("tau_ref ignored because irf was provided directly")
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

        returned_irf = irf_hist_norm.copy()
        if irf_output == "shifted":
            returned_irf = Alignment._normalize_histogram_1d(
                Alignment.linear_shift(returned_irf, dT_bins, cyclic=True),
                name="shifted irf",
            )

        fitted_hist = Alignment._normalize_histogram_1d(
            Alignment.fit_model_data(
                t_ns,
                fit_result["C"],
                fit_result["dT"],
                fit_result["tau"],
                irf=irf_hist_norm,
                period=period_ns,
                mode=mode,
            ),
            name="fit",
        )

        result = {
            "C": float(fit_result["C"]),
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
            result["ref_shifted"] = Alignment._normalize_histogram_1d(
                Alignment.linear_shift(ref_hist_norm, dT_bins, cyclic=True),
                name="shifted ref",
            )
        elif shift_output == "data":
            result["data_shifted"] = Alignment._normalize_histogram_1d(
                Alignment.linear_shift(data_hist_norm, -dT_bins, cyclic=True),
                name="shifted data",
            )

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
            irf_shifted_hist = shift(irf_hist, dT_bins, order=1, mode='grid-wrap') #use scipy.ndimage.shift for sub-bin shifts with cyclic wrapping, using linear interpolation
            #irf_shifted_hist = Alignment.linear_shift(irf_hist, shift_value=dT_bins, cyclic=True) # use linear interpolation for sub-bin shifts, with cyclic wrapping
        else:
            raise ValueError(f"Unsupported mode: {mode}. Supported model_shift, irf_shift")
        
        pure_model_hist = pure_model_hist / pure_model_hist.sum()  # normalize model to unit area so C is a simple amplitude factor
        irf_shifted_hist = irf_shifted_hist / irf_shifted_hist.sum()  # normalize IRF to unit area  


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

        if data_sum <= 0:
            raise ValueError("data must contain at least one positive value")
            data_max = 1.0  # avoid division by zero, though the fit will likely fail with non-positive data
        if irf_sum <= 0:
            raise ValueError("irf must have a positive sum")

        data_hist_norm = data_hist / data_sum
        irf_hist_norm = irf_hist / irf_sum
        #data_hist = np.log(data_hist + 1)  # add small constant to avoid log(0)
        #irf_hist = np.log(irf_hist + 1)  # add small constant to avoid log(0)    
        
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
                print("Using circular fit with curve_fit_circular and C fixed to 1.0")
                print("initial_guess", initial_guess_fixed_c)
                print("bounds", fit_bounds_fixed_c)
                print("circular_params", {0: float(nbin)})
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
                print("Using standard curve_fit with C fixed to 1.0")
                print("initial_guess", initial_guess_fixed_c)
                print("bounds", fit_bounds_fixed_c)
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
                print("Using circular fit with curve_fit_circular")
                print("initial_guess", initial_guess)
                print("bounds", fit_bounds)
                print("circular_params", {1: float(nbin)})
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
                print("Using standard curve_fit")
                print("initial_guess", initial_guess)
                print("bounds", fit_bounds)
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

    # @staticmethod
    # def perform_fit_data_ng(t, data, irf, period, initial_tau=None, initial_dT=None, initial_C=None):
    #     """
    #     Same fit as ``perform_fit_data()``, but ``data`` and ``irf`` are rolled
    #     so the data peak is near the central bin before fitting.

    #     The returned ``dT`` is converted back to the original, unrolled bin
    #     coordinates.
    #     """
    #     Alignment._require_scipy_optimize()

    #     t_ns = Alignment.to_numpy_1d(t, dtype=float)
    #     data_hist = Alignment.to_numpy_1d(data, dtype=float)
    #     irf_hist = Alignment.to_numpy_1d(irf, dtype=float)

    #     if len(t_ns) != len(data_hist) or len(t_ns) != len(irf_hist):
    #         raise ValueError("t, data, and irf must have the same 1D length")

    #     data_hist = data_hist / data_hist.max()
    #     irf_hist = irf_hist / irf_hist.sum()

    #     nbin = len(t_ns)
    #     roll_to_center_bins = np.mod(nbin // 2 - int(np.argmax(data_hist)), nbin)

    #     data_hist_rolled = np.roll(data_hist, roll_to_center_bins)
    #     irf_hist_rolled = np.roll(irf_hist, roll_to_center_bins)

    #     fit_lambda = lambda t_ns_fit, C_norm, dT_bins, tau_ns: Alignment.fit_model_data(
    #         t_ns_fit, C_norm, dT_bins, tau_ns, irf=irf_hist_rolled, period=period
    #     )

    #     initial_guess = [1.0, 0.0, 1.0]

    #     if initial_C is not None:
    #         initial_guess[0] = initial_C
    #     if initial_dT is not None:
    #         initial_guess[1] = initial_dT + roll_to_center_bins
    #     if initial_tau is not None:
    #         initial_guess[2] = initial_tau

    #     popt, conv = scipy_optimize.curve_fit(
    #         fit_lambda,
    #         t_ns,
    #         data_hist_rolled,
    #         p0=initial_guess,
    #         bounds=([0.0, -nbin / 2, 1e-5], [np.inf, nbin / 2, t_ns.max()]),
    #         maxfev=600000,
    #     )

    #     dT_bins = popt[1] - roll_to_center_bins
    #     dT_bins = ((dT_bins + nbin / 2) % nbin) - nbin / 2

    #     return {"C": popt[0], "dT": dT_bins, "tau": popt[2]}, conv

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


# class FlimCalibrateData:
#
#     def __init__(self, data_path: str, calibration_path: str, freq_exc: float = 21e6):
#         self.freq_exc = freq_exc
#         self.load_data(data_path)
#         self.data = None
#         self.data_ref = None
#         self.data_calib = None
#         self.metadata = None
#         self.calib_ref = None
#         self.metadata_calib = None
#
#         self.load_data(data_path)
#         self.load_calibration_data(calibration_path)
#
#     def load_data(self, data_path: str):
#         self.data, self.metadata = mcs.load(data_path)
#         data_extra, _ = mcs.load(data_path, key="data_channels_extra")
#         self.data_ref = data_extra[..., 1].sum(axis=np.arange(0, 4))
#
#     def load_calibration_data(self, calibration_path: str):
#         self.data_calib, self.metadata_calib = mcs.load(calibration_path)
#         data_calib_extra, _ = mcs.load(calibration_path, key="data_channels_extra")
#         self.calib_ref = data_calib_extra[..., 1].sum(axis=np.arange(0, 4))
#
#     def load_calibration_parameters(self, calibration_path: str):
#         pass
#
#     def calculate_calib_phasor(self, laser_registration=True):
#         calculate_phasor(self.calib_ref)

def correct_phasor(phasor_data, laser_phasor, coeff=1., laser_correction=True):
    if phasor_data is None:
        raise (ValueError("call before calculate_phasor_on_img_ch(...)"))

    if laser_correction:
        return phasor_data * coeff * np.exp(1j * (-np.angle(laser_phasor)))
    else:
        return phasor_data * coeff


def sum_adjacent_pixel(data, n=4):
    if n <= 1:
        return data
    print("original data.shape", data.shape)
    if len(data.shape) == 2:
        data = data[:, 0:((data.shape[0] // n) * n)]
        data = data[0:((data.shape[1] // n) * n), :]
        print("reduced data.shape", data.shape)
        assert (data.shape[0] % n == 0)
        assert (data.shape[1] % n == 0)
        d = data[0::n, ::n]
        for i in range(1, n):
            d = d + data[i::n, i::n]
        print("merged data.shape", data.shape)
        return d
    elif len(data.shape) == 6:
        data = data[:, :, :, 0:((data.shape[2] // n) * n), :, :]
        data = data[:, :, 0:((data.shape[3] // n) * n), :, :, :]
        print("reduced data.shape", data.shape)
        # rzyxtc
        assert (data.shape[2] % n == 0)
        assert (data.shape[3] % n == 0)
        d = data[:, :, 0::n, 0::n, :, :]
        for i in range(1, n):
            d = d + data[:, :, i::n, i::n, :, :]
        print("merged data.shape", data.shape)
        return d


def _require_torch():
    if torch is None:
        raise ImportError(
            "estimate_irf() requires PyTorch. Please install `torch` to use the IRF deconvolution utilities."
        )


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

    _require_torch()

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


def torch_median_filter(x, window_size=3, dims=None, mode="reflect"):
    """
    Apply a median filter to a torch tensor along selected dimensions.
    """

    _require_torch()

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
        out = pad_tensor(out, pad_left, pad_right, d, mode=mode)
        out = out.unfold(d, w, 1).median(dim=-1).values

    return out


def partial_convolution_fft(volume, kernel, dim1: str = "ijk", dim2: str = "jkl",
                            axis: str = "jk", fourier: tuple = (False, False)):
    """
    Convolution through FFT with einsum-based dimension bookkeeping.
    """

    _require_torch()

    dim3 = dim1 + dim2
    dim3 = ''.join(sorted(set(dim3), key=dim3.index))
    dims = [dim1, dim2, dim3]
    axis_list = [[d.find(c) for c in axis] for d in dims]

    volume_fft = fftn(volume, dim=axis_list[0]) if not fourier[0] else volume
    kernel_fft = fftn(kernel, dim=axis_list[1]) if not fourier[1] else kernel

    conv = torch.einsum(f"{dim1},{dim2}->{dim3}", volume_fft, kernel_fft)
    conv = ifftn(conv, dim=axis_list[2])
    conv = ifftshift(conv, dim=axis_list[2])
    conv = torch.real(conv)
    return conv


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
        #print_dec("FIT: invalid peak value")
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
        #print_dec("FIT: y_start, y_end, idx_start_candidates, idx_end_candidates", y_start, y_end, idx_start_candidates, idx_end_candidates)
        #print_dec("FIT: no candidates for start or end indices")
        return None, None, peak_idx, trace_sum_peak0, trace_x_peak0_ns

    #print_dec("FIT: start_level, selected_end_level", start_level, selected_end_level)

    start_idx = int(idx_start_candidates[0])
    end_idx = int(idx_end_candidates[0])
    if end_idx <= start_idx:
        #print_dec("FIT: end_idx <= start_idx")
        return None, None, peak_idx, trace_sum_peak0, trace_x_peak0_ns

    y_section = trace_sum_peak0[start_idx : end_idx + 1]
    x_section_ns = trace_x_peak0_ns[start_idx : end_idx + 1]
    positive = y_section > 0
    if np.count_nonzero(positive) < 4:
        #print_dec("FIT: np.count_nonzero(positive) < 4")
        return None, None, peak_idx, trace_sum_peak0, trace_x_peak0_ns

    x_fit_bins = np.arange(start_idx, end_idx + 1, dtype=float)[positive]
    x_fit_ns = x_section_ns[positive]
    y_fit_input = y_section[positive]
    slope, intercept = np.polyfit(x_fit_bins, np.log(y_fit_input), 1)
    #print_dec("FIT: slope and intercept", slope, intercept)
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

    # Bin centers
    t_ns = (np.arange(n) + 0.5) * dt_ns

    # CYCLIC forward delay from t0
    delay_ns = (t_ns - t0_ns) % Tcycle_ns

    # Background subtraction
    w = np.clip(y - background_per_bin, 0.0, None)

    # Optional truncation in the cyclic-delay domain
    if truncate_ns is not None:
        keep = delay_ns < truncate_ns
        w = w[keep]
        delay_ns = delay_ns[keep]

    s0 = np.sum(w)
    if s0 <= 0:
        return np.nan

    tau_ns = np.sum(w * delay_ns) / s0
    return tau_ns






def plot_universal_circle(ax=None):
    '''
    draw the universal circle in the last plot
    '''
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
    '''
    ravel the two array and remove the rows where at least one of the two array is nan
    :param a: input array
    :param b: input array (or None)
    :return: 1D array, 1D array
    '''

    a = a.flatten()
    if b is None:
        cond = np.isfinite(a)
        return a[cond]
    if b is not None:
        b = b.flatten()
        cond = np.logical_not(np.logical_or(np.isnan(a), np.isnan(b)))
        return a[cond], b[cond]


def g0s0_to_m_phi(g0, s0):
    '''
    :param g0: array or number
    :param s0: array or number
    :return: modulo, phase array or number
    '''
    phi = np.arctan2(s0, g0)
    m = np.sqrt(s0 ** 2 + g0 ** 2)
    return m, phi


def m_phi_to_g0s0(m, phi):
    '''
    :param m: modulo, array or number
    :param phi: fase, array or number
    :return: g0, s0 array or number
    '''
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
    '''
    :param data_hist: the histogram 1D
    :param threshold: the minimum of entry of the histogram
    :return: phasor: g0 + 1j * s0
    '''

    data = data_hist.copy()
    if len(data.shape) == 1:
        # data_thresholded = np.where(data < 10, data, 0)
        N = np.sum(data)
        if N < threshold:
            return np.nan + 1j * np.nan
        else:
            f = np.fft.fft(data / N)  # FFT of normalized data
            return f[harmonic].conj()  # 1st harmonic
    elif len(data.shape) == 2:
        # data_thresholded = np.where(data < 10, data, 0)
        N = np.sum(data, axis=0)
        print(N.shape, data.shape)
        f = np.fft.fft(data / N, axis=0)  # FFT of normalized data
        print(f.shape)
        out = f[harmonic]
        out[N < threshold * np.ones(data.shape[1])] = np.nan + 1j * np.nan
        return out.conj()
    elif len(data.shape) == 3:
        flux = data.sum(-1)
        transform = np.fft.fft(data, axis=-1)[..., harmonic].conj()
        out = transform / flux
        out[flux < threshold * np.ones(data.shape[1])] = np.nan + 1j * np.nan
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

        # Create an empty dataset with the specified dimensions in h5_dim

        h5_dim = (x_dim, y_dim, channel_dim)
        h5_dataset_p = fi.create_dataset('h5_dataset_p', shape=h5_dim, dtype=np.complex128)
        # h5_dataset_np = np.zeros(h5_dim, dtype=np.complex128)

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

    # Extract directory and filename from the provided data_path
    directory, filename = os.path.split(data_path)
    # Remove file extension
    filename = os.path.splitext(filename)[0]

    # Generate new filename with "aligned" appended
    new_filename = filename + "_phasors_matrix.h5"
    # Construct full path for the new file
    new_file_path = os.path.join(directory, new_filename)
    if len(data_input_shape) < 3:
        raise ValueError("data_input must have 3 or more dimensions")
    elif len(data_input_shape) >= 6:
        raise ValueError("use calculate_phasor_on_img_ch() instead")

    with h5py.File(new_file_path, 'w') as fil:

        # Create an empty dataset with the specified dimensions in dataset_shape
        x_dim, y_dim = data_input_3d.shape[0], data_input_3d.shape[1]
        h5_dim = (x_dim, y_dim)
        # ze = np.zeros(h5_dim)
        h5_dataset_phasor_pix = fil.create_dataset('h5_dataset_phasor_pix', shape=h5_dim, dtype=np.complex128)
        # h5_dataset_phasor_pix[:] = np.zeros(h5_dim, dtype=np.complex128)

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

    # Extract directory and filename from the provided data_path
    directory, filename = os.path.split(data_path)
    # Remove file extension
    filename = os.path.splitext(filename)[0]

    # Generate new filename with "aligned" appended
    new_filename = filename + "_phasors_matrix.h5"
    # Construct full path for the new file
    new_file_path = os.path.join(directory, new_filename)
    if len(data_input_shape) < 3:
        raise ValueError("data_input must have 3 or more dimensions")
    elif len(data_input_shape) >= 6:
        raise ValueError("use calculate_phasor_on_img_ch() instead")

    with h5py.File(new_file_path, 'w') as fil:

        # Create an empty dataset with the specified dimensions in dataset_shape
        x_dim, y_dim = data_input_3d.shape[0], data_input_3d.shape[1]
        h5_dim = (x_dim, y_dim)
        h5_dataset_phasor_pix = fil.create_dataset('h5_dataset_phasor_pix', shape=h5_dim, dtype=np.complex128)
        # h5_dataset_phasor_pix[:] = np.zeros(h5_dim, dtype=np.complex128)

        for x_start in range(0, x_dim, phasor_pix_data_size):
            for y_start in range(0, y_dim, phasor_pix_data_size):
                x_stop = min(x_start + phasor_pix_data_size, x_dim)
                y_stop = min(y_start + phasor_pix_data_size, y_dim)

                slice_term_pix = np.s_[x_start:x_stop, y_start:y_stop, :]
                sub_image_pix = data_input_3d[slice_term_pix]

                aligned_phasor_pix = np.zeros(sub_image_pix.shape[:-1], dtype=np.complex128)

                # for cc in np.arange(sub_image_pix.shape[-1]):
                # for rr in tqdm(np.arange(data_input.shape[-6]), leave=False):
                #   for zz in tqdm(np.arange(data_input.shape[-5]), leave=False):
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
        # plot(x,y,".-k")
        # text(x0,y0," %d ns"%(i*1e9))

        m = 1.
        ttt = y0 / x0
        x0 = 1 / (1 + ttt ** 2)
        y0 = np.sqrt(0.25 - (x0 - 0.5) ** 2)
        # print(i,x0,y0)
        x = [0, x0]
        y = [0, y0]

        m, phi = g0s0_to_m_phi(x0, y0)
        ax.plot(x, y, ".k")

        gamma = np.arctan2(y0, x0 - 0.5)  # the angle in the small circle used for add label

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

    # the HSV representation
    Hn = ((Hp - params["minTau"]) / (params["maxTau"] - params["minTau"])) * params[
        "satFactor"
    ]
    Sn = np.ones(Hp.shape)
    Vn = (intensity - params["minInt"]) / (params["maxInt"] - params["minInt"])

    HSV = np.empty((sz[0], sz[1], 3))

    if params["invertColormap"] == True:
        Hn = params["satFactor"] - Hn

    # set to violet color of pixels outside lifetime bounds
    # BG = intensity < ( np.max(intensity * params.bgIntPerc )
    Hn[np.not_equal(lifetime, Hp)] = params["outOfBoundsHue"]

    HSV[:, :, 0] = Hn.astype("float64")
    HSV[:, :, 1] = Sn.astype("float64")
    HSV[:, :, 2] = Vn.astype("float64")

    # convert to RGB
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
