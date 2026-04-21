"""HDF5 calibration and structure inspection helpers."""

from html import escape
import json
from pathlib import Path
import shutil
import warnings

import h5py
import numpy as np
from tqdm.auto import tqdm

import brighteyes_ism.dataio.mcs as mcs

from .alignment import Alignment

__all__ = [
    "H5DataCalibrator",
    "calibrate_h5_file",
    "show_h5_structure",
    "show_h5_structure_html",
]

DEFAULT_DATA_KEY = ("data", "data_channels_extra")
DEFAULT_REFERENCE_KEY = ("data", "data_channels_extra")
DEFAULT_TAU_REF = None
DEFAULT_REFERENCE_TYPE = "ref"
DEFAULT_FIT_MODE = "model_shift"
DEFAULT_FIT_TYPE = "circular"
DEFAULT_C_REF = 1.0
DEFAULT_IRF_ITERATIONS = 300
DEFAULT_REGULARIZATION = 0
DEFAULT_OVERWRITE = True


class H5DataCalibrator:
    """
    Calibrate per-channel FLIM histograms stored in HDF5 files.

    Parameters
    ----------
    data_path : str or path-like
        HDF5 file containing the data to calibrate.
    reference_path : str or path-like
        HDF5 file containing the reference histogram or IRF source.
    data_key : str or iterable of str, default ``("data", "data_channels_extra")``
        Dataset key or keys to calibrate in ``data_path``. When ``None``, the
        class falls back to ``DEFAULT_DATA_KEYS``.
    reference_key : None, str, iterable of str, or dict, default ``None``
        Dataset key selection for the reference file. If ``None``, each data
        key is mapped to itself. A single string is reused for all data keys, an
        iterable can provide one key per data key, and a dict can map each data
        key explicitly.
    reference_type : {"ref", "irf"}, default ``"ref"``
        Type of reference data. Aliases such as ``"reference"`` are normalized
        internally to ``"ref"``.
    tau_ref : float or None, default ``None``
        Reference lifetime in ns. When ``None``, the reference lifetime is
        estimated from the reference data when needed by the chosen fit mode.
    fit_mode : str, default ``"model_shift"``
        Fitting mode forwarded to the alignment routines.
    fit_type : str, default ``"circular"``
        Fit geometry forwarded to the alignment routines.
    C_ref : float, default ``1.0``
        Reference amplitude scaling factor.
    output_path : str or path-like or None, default ``None``
        Output HDF5 path. When ``None``, a default calibrated output path is
        derived from ``data_path``.
    overwrite : bool, default ``True``
        If ``True``, overwrite an existing output file.
    channels : iterable of int or None, default ``None``
        Optional subset of channel indices to calibrate. When ``None``, all
        available channels are processed.
    calibration_key : str, default ``"calibration"``
        Group name used to store calibration outputs in the destination file.
    period_ns : float or None, default ``None``
        Laser period in ns. When ``None``, the value is inferred from metadata
        when possible.
    initial_tau : float or None, default ``None``
        Optional initial guess for the lifetime fit.
    initial_dT : float or None, default ``None``
        Optional initial guess for the temporal shift fit.
    initial_C : float or None, default ``None``
        Optional initial guess for the amplitude fit.
    force_C_normalized : bool, default ``False``
        If ``True``, force the fitted amplitude term to remain normalized.
    irf_iterations : int, default ``300``
        Number of iterations used when estimating the IRF from the reference
        data.
    eps : float, default ``1e-8``
        Numerical stability constant passed to the fitting routines.
    regularization : float, default ``0``
        Regularization strength used during IRF estimation.

    Notes
    -----
    The input datasets are expected to have shape
    ``[repetition, z, y, x, t, ch]``. Only one channel histogram at a time is
    materialized in memory, so the whole 6D dataset is never converted to a
    NumPy array up front.
    """

    DEFAULT_DATA_KEYS = DEFAULT_DATA_KEY

    def __init__(
        self,
        data_path,
        reference_path,
        data_key=DEFAULT_DATA_KEY,
        reference_key=None,
        reference_type=DEFAULT_REFERENCE_TYPE,
        tau_ref=DEFAULT_TAU_REF,
        fit_mode=DEFAULT_FIT_MODE,
        fit_type=DEFAULT_FIT_TYPE,
        C_ref=DEFAULT_C_REF,
        output_path=None,
        overwrite=DEFAULT_OVERWRITE,
        channels=None,
        calibration_key="calibration",
        period_ns=None,
        initial_tau=None,
        initial_dT=None,
        initial_C=None,
        force_C_normalized=False,
        irf_iterations=DEFAULT_IRF_ITERATIONS,
        eps=1e-8,
        regularization=DEFAULT_REGULARIZATION,
    ):
        self.data_path = Path(data_path)
        self.reference_path = Path(reference_path)
        self.data_keys = self._normalize_key_sequence(data_key, "data_key")
        self.reference_key_map = self._normalize_reference_keys(reference_key, self.data_keys)
        self.reference_type = self._normalize_reference_type(reference_type)
        self.tau_ref = tau_ref
        self.fit_mode = fit_mode
        self.fit_type = fit_type
        self.C_ref = C_ref
        self.output_path = Path(output_path) if output_path is not None else self._default_output_path()
        self.overwrite = overwrite
        self.channels = channels
        self.calibration_key = calibration_key
        self.period_ns = period_ns
        self.initial_tau = initial_tau
        self.initial_dT = initial_dT
        self.initial_C = initial_C
        self.force_C_normalized = force_C_normalized
        self.irf_iterations = irf_iterations
        self.eps = eps
        self.regularization = regularization

    @classmethod
    def _normalize_key_sequence(cls, keys, param_name):
        if keys is None:
            keys = cls.DEFAULT_DATA_KEYS

        if isinstance(keys, (str, Path)):
            normalized = [str(keys)]
        else:
            try:
                normalized = [str(key) for key in keys]
            except TypeError as exc:
                raise TypeError(f"{param_name} must be a string or an iterable of strings") from exc

        normalized = [key for key in normalized if key]
        if not normalized:
            raise ValueError(f"{param_name} must contain at least one dataset key")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{param_name} must not contain duplicates")
        return normalized

    @staticmethod
    def _normalize_reference_keys(reference_key, data_keys):
        if reference_key is None:
            return {data_key: data_key for data_key in data_keys}

        if isinstance(reference_key, dict):
            mapping = {}
            for data_key in data_keys:
                if data_key not in reference_key:
                    raise KeyError(f"missing reference_key entry for data key {data_key!r}")
                mapping[data_key] = str(reference_key[data_key])
            return mapping

        if isinstance(reference_key, (str, Path)):
            reference_key_str = str(reference_key)
            return {data_key: reference_key_str for data_key in data_keys}

        try:
            reference_keys = [str(key) for key in reference_key]
        except TypeError as exc:
            raise TypeError(
                "reference_key must be None, a string, a mapping, or an iterable of strings"
            ) from exc

        if len(reference_keys) == 1:
            return {data_key: reference_keys[0] for data_key in data_keys}
        if len(reference_keys) != len(data_keys):
            raise ValueError(
                "reference_key iterable must have length 1 or match the number of data keys"
            )
        return {data_key: ref_key for data_key, ref_key in zip(data_keys, reference_keys)}

    @staticmethod
    def _normalize_reference_type(reference_type):
        normalized = str(reference_type).strip().lower()
        aliases = {
            "reference": "ref",
            "reference_histogram": "ref",
            "ref_histogram": "ref",
            "fluorescence_reference": "ref",
            "irf_histogram": "irf",
            "input_irf": "irf",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"ref", "irf"}:
            raise ValueError("reference_type must be 'ref' or 'irf'")
        return normalized

    def _default_output_path(self):
        suffix = self.data_path.suffix or ".h5"
        return self.data_path.with_name(f"{self.data_path.stem}_calib{suffix}")

    @staticmethod
    def _metadata_get(metadata, key, default=None):
        if metadata is None:
            return default
        if isinstance(metadata, dict):
            return metadata.get(key, default)
        if hasattr(metadata, "get"):
            try:
                return metadata.get(key, default)
            except TypeError:
                pass
        return getattr(metadata, key, default)

    @staticmethod
    def _metadata_items(metadata):
        if metadata is None:
            return []
        if isinstance(metadata, dict):
            return list(metadata.items())
        if hasattr(metadata, "items"):
            try:
                return list(metadata.items())
            except TypeError:
                pass
        if hasattr(metadata, "__dict__") and vars(metadata):
            return [(key, value) for key, value in vars(metadata).items() if not key.startswith("_")]

        items = []
        for key in dir(metadata):
            if key.startswith("_"):
                continue
            try:
                value = getattr(metadata, key)
            except Exception:
                continue
            if callable(value):
                continue
            items.append((key, value))
        return items

    @staticmethod
    def build_time_axis(metadata, nbin=None, period_ns=None):
        if nbin is None:
            metadata_nbin = H5DataCalibrator._metadata_get(metadata, "dfd_nbins")
            if metadata_nbin is None:
                raise ValueError("metadata must provide dfd_nbins or nbin must be passed explicitly")
            nbin = int(metadata_nbin)
        else:
            nbin = int(nbin)

        if nbin <= 0:
            raise ValueError("nbin must be positive")

        if period_ns is None:
            dfd_freq_MHz = H5DataCalibrator._metadata_get(metadata, "dfd_freq")
            if dfd_freq_MHz is None:
                raise ValueError("metadata must provide dfd_freq or period_ns must be passed explicitly")
            dfd_freq_MHz = float(dfd_freq_MHz)
            if not np.isfinite(dfd_freq_MHz) or dfd_freq_MHz <= 0:
                raise ValueError("metadata.dfd_freq must be a positive finite value")
            period_ns = 1e3 / dfd_freq_MHz
        else:
            period_ns = float(period_ns)

        if not np.isfinite(period_ns) or period_ns <= 0:
            raise ValueError("period_ns must be a positive finite value")

        dt_ns = period_ns / nbin
        t_ns = np.arange(nbin, dtype=float) * dt_ns
        return nbin, dt_ns, period_ns, t_ns

    @staticmethod
    def _open_dataset(handle, key=None):
        if key is not None:
            if key not in handle:
                raise KeyError(f"dataset key {key!r} not found in {handle.filename!r}")
            dataset = handle[key]
        else:
            dataset = None
            for candidate in H5DataCalibrator.DEFAULT_DATA_KEYS:
                if candidate in handle:
                    dataset = handle[candidate]
                    break
            if dataset is None:
                raise KeyError(
                    f"no default dataset found in {handle.filename!r}; "
                    f"tried {H5DataCalibrator.DEFAULT_DATA_KEYS}"
                )

        if not isinstance(dataset, h5py.Dataset):
            raise TypeError(f"{key!r} in {handle.filename!r} is not an HDF5 dataset")
        return dataset

    @staticmethod
    def _validate_dataset_layout(dataset, name):
        if dataset.ndim != 6:
            raise ValueError(
                f"{name} dataset must have shape [repetition, z, y, x, t, ch], got {dataset.shape}"
            )
        if dataset.shape[-2] <= 0 or dataset.shape[-1] <= 0:
            raise ValueError(f"{name} dataset must contain positive time and channel dimensions")

    @staticmethod
    def _resolve_channels(channels, channel_count):
        if channels is None:
            return list(range(int(channel_count)))

        resolved = []
        for channel in channels:
            channel_index = int(channel)
            if channel_index < 0 or channel_index >= channel_count:
                raise IndexError(f"channel {channel_index} out of range for {channel_count} channels")
            resolved.append(channel_index)

        if len(set(resolved)) != len(resolved):
            raise ValueError("channels must not contain duplicates")
        return resolved

    @staticmethod
    def _resolve_reference_channel_map(data_dataset, reference_dataset):
        data_channel_count = int(data_dataset.shape[-1])
        reference_channel_count = int(reference_dataset.shape[-1])

        if reference_channel_count == data_channel_count:
            return {channel: channel for channel in range(data_channel_count)}
        if reference_channel_count == 1:
            return {channel: 0 for channel in range(data_channel_count)}

        raise ValueError(
            "reference dataset must have either the same number of channels as data "
            "or exactly one channel"
        )

    @staticmethod
    def _sum_histogram_for_channel(dataset, channel_index):
        nbin = int(dataset.shape[-2])
        histogram = np.zeros(nbin, dtype=np.float64)

        for repetition_index in range(int(dataset.shape[0])):
            for z_index in range(int(dataset.shape[1])):
                histogram += np.asarray(
                    dataset[repetition_index, z_index, :, :, :, channel_index],
                    dtype=np.float64,
                ).sum(axis=(0, 1))

        return histogram

    @staticmethod
    def _prepare_attr_value(value):
        if value is None:
            return "None"
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (str, bytes)):
            return value
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return float(value)
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return H5DataCalibrator._prepare_attr_value(value.item())
            if value.dtype.kind in {"i", "u", "f", "b"}:
                return value
            return json.dumps(value.tolist(), default=str)
        if isinstance(value, (list, tuple, dict, set)):
            return json.dumps(value, default=str)
        return str(value)

    @classmethod
    def _set_group_attrs(cls, group, attrs):
        for key, value in attrs.items():
            group.attrs[str(key)] = cls._prepare_attr_value(value)

    @classmethod
    def _write_metadata_group(cls, parent_group, group_name, metadata):
        metadata_group = parent_group.create_group(group_name)
        for key, value in cls._metadata_items(metadata):
            metadata_group.attrs[str(key)] = cls._prepare_attr_value(value)
        return metadata_group

    @staticmethod
    def _replace_dataset(group, name, data):
        if name in group:
            del group[name]
        array = np.asarray(data)
        kwargs = {}
        if array.ndim > 0 and array.size > 0:
            kwargs["compression"] = "gzip"
        return group.create_dataset(name, data=array, **kwargs)

    @staticmethod
    def _format_tau_ref_input(tau_ref):
        if tau_ref is None:
            return "None"
        if isinstance(tau_ref, str):
            return tau_ref
        try:
            return float(tau_ref)
        except (TypeError, ValueError):
            return str(tau_ref)

    @staticmethod
    def _empty_fit_payload(nbin, reference_type, data_histogram, reference_histogram, irf_source):
        zero_hist = np.zeros(int(nbin), dtype=float)
        payload = {
            "C": np.nan,
            "tau": np.nan,
            "tau_ref": np.nan,
            "dT": np.nan,
            "dT_ns": np.nan,
            "cov": np.full((3, 3), np.nan, dtype=float),
            "data_histogram": np.asarray(data_histogram, dtype=float),
            "irf": zero_hist.copy(),
            "irf_calibrated": zero_hist.copy(),
            "fit": zero_hist.copy(),
            "irf_source": str(irf_source),
        }
        if reference_type == "ref":
            payload["ref_original"] = np.asarray(reference_histogram, dtype=float)
            payload["ref_calibrated"] = zero_hist.copy()
        return payload

    def _prepare_output_file(self):
        if self.output_path.resolve() == self.data_path.resolve():
            raise ValueError("output_path must be different from data_path")
        if self.output_path.exists():
            if not self.overwrite:
                raise FileExistsError(f"output file already exists: {self.output_path}")
            self.output_path.unlink()
        shutil.copy2(self.data_path, self.output_path)
        return self.output_path

    def _get_target_group(self, calibration_group, data_key):
        if len(self.data_keys) == 1:
            return calibration_group
        if data_key in calibration_group:
            del calibration_group[data_key]
        return calibration_group.create_group(data_key)

    def calibrate(self):
        data_metadata = mcs.metadata_load(str(self.data_path))
        reference_metadata = mcs.metadata_load(str(self.reference_path))
        output_path = self._prepare_output_file()

        with h5py.File(self.data_path, "r") as data_handle, \
             h5py.File(self.reference_path, "r") as reference_handle, \
             h5py.File(output_path, "a") as output_handle:

            if self.calibration_key in output_handle:
                del output_handle[self.calibration_key]
            calibration_group = output_handle.create_group(self.calibration_key)

            self._set_group_attrs(
                calibration_group,
                {
                    "source_data_file": str(self.data_path),
                    "source_reference_file": str(self.reference_path),
                    "output_file": str(output_path),
                    "data_keys": json.dumps(self.data_keys),
                    "reference_keys": json.dumps(self.reference_key_map),
                    "reference_type": self.reference_type,
                    "tau_ref_input": self._format_tau_ref_input(self.tau_ref),
                    "fit_mode": self.fit_mode,
                    "fit_type": self.fit_type,
                    "C_ref": self.C_ref,
                    "irf_iterations": self.irf_iterations,
                    "eps": self.eps,
                    "regularization": self.regularization,
                    "initial_tau": self.initial_tau,
                    "initial_dT": self.initial_dT,
                    "initial_C": self.initial_C,
                    "force_C_normalized": self.force_C_normalized,
                    "data_key_count": len(self.data_keys),
                },
            )
            self._write_metadata_group(calibration_group, "input_data_metadata", data_metadata)
            self._write_metadata_group(calibration_group, "input_reference_metadata", reference_metadata)

            data_key_iterator = tqdm(
                self.data_keys,
                desc="Calibrating data keys",
                unit="key",
                disable=len(self.data_keys) <= 1,
            )
            for data_key in data_key_iterator:
                reference_key = self.reference_key_map[data_key]
                data_dataset = self._open_dataset(data_handle, data_key)
                reference_dataset = self._open_dataset(reference_handle, reference_key)

                self._validate_dataset_layout(data_dataset, f"data[{data_key}]")
                self._validate_dataset_layout(reference_dataset, f"reference[{reference_key}]")

                if data_dataset.shape[-2] != reference_dataset.shape[-2]:
                    raise ValueError(
                        "data and reference datasets must have the same number of time bins "
                        f"(got {data_dataset.shape[-2]} and {reference_dataset.shape[-2]}) "
                        f"for data key {data_key!r}"
                    )

                channel_indices = self._resolve_channels(self.channels, int(data_dataset.shape[-1]))
                reference_channel_map = self._resolve_reference_channel_map(data_dataset, reference_dataset)
                if len(channel_indices) == 0:
                    raise ValueError("channels must contain at least one channel index")

                nbin, dt_ns, period_ns, t_ns = self.build_time_axis(
                    data_metadata,
                    nbin=int(data_dataset.shape[-2]),
                    period_ns=self.period_ns,
                )

                target_group = self._get_target_group(calibration_group, data_key)
                self._set_group_attrs(
                    target_group,
                    {
                        "data_key": data_key,
                        "reference_key": reference_key,
                        "reference_type": self.reference_type,
                        "tau_ref_input": self._format_tau_ref_input(self.tau_ref),
                        "fit_mode": self.fit_mode,
                        "fit_type": self.fit_type,
                        "C_ref": self.C_ref,
                        "irf_iterations": self.irf_iterations,
                        "eps": self.eps,
                        "regularization": self.regularization,
                        "initial_tau": self.initial_tau,
                        "initial_dT": self.initial_dT,
                        "initial_C": self.initial_C,
                        "force_C_normalized": self.force_C_normalized,
                        "nbin": nbin,
                        "dt_ns": dt_ns,
                        "period_ns": period_ns,
                        "channel_count": int(data_dataset.shape[-1]),
                        "channel_count_calibrated": len(channel_indices),
                        "data_shape": list(data_dataset.shape),
                        "reference_shape": list(reference_dataset.shape),
                        "channel_axis": -1,
                        "stacked_histogram_layout": "(t, ch)",
                        "stacked_covariance_layout": "(param, param, ch)",
                    },
                )

                stacked_channel_index = []
                stacked_reference_channel_index = []
                stacked_C = []
                stacked_tau = []
                stacked_tau_ref = []
                stacked_dT_bins = []
                stacked_dT_ns = []
                stacked_irf_source = []
                stacked_covariance = []
                stacked_data_histogram = []
                stacked_irf_original = []
                stacked_irf_calibrated = []
                stacked_fit = []
                stacked_ref_original = []
                stacked_ref_calibrated = []

                channel_iterator = tqdm(
                    channel_indices,
                    desc=f"Calibrating {data_key}",
                    unit="ch",
                    leave=False,
                )
                for channel_index in channel_iterator:
                    reference_channel_index = reference_channel_map[channel_index]
                    data_histogram = self._sum_histogram_for_channel(data_dataset, channel_index)
                    reference_histogram = self._sum_histogram_for_channel(reference_dataset, reference_channel_index)
                    data_sum = float(np.sum(data_histogram))
                    reference_sum = float(np.sum(reference_histogram))

                    if not np.isfinite(data_sum) or data_sum <= 0:
                        warnings.warn(
                            (
                                f"Skipping calibration for data key {data_key!r}, channel {channel_index}: "
                                "data histogram has a non-positive or non-finite sum"
                            ),
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        fit_payload = self._empty_fit_payload(
                            nbin=nbin,
                            reference_type=self.reference_type,
                            data_histogram=data_histogram,
                            reference_histogram=reference_histogram,
                            irf_source="skipped_zero_sum_data",
                        )
                    elif not np.isfinite(reference_sum) or reference_sum <= 0:
                        warnings.warn(
                            (
                                f"Skipping calibration for data key {data_key!r}, channel {channel_index}: "
                                f"reference channel {reference_channel_index} histogram has a non-positive "
                                "or non-finite sum"
                            ),
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        fit_payload = self._empty_fit_payload(
                            nbin=nbin,
                            reference_type=self.reference_type,
                            data_histogram=data_histogram,
                            reference_histogram=reference_histogram,
                            irf_source="skipped_zero_sum_reference",
                        )
                    else:
                        fit_kwargs = {
                            "t": t_ns,
                            "data": data_histogram,
                            "period": period_ns,
                            "C_ref": self.C_ref,
                            "irf_output": "original",
                            "shift_output": "ref" if self.reference_type == "ref" else None,
                            "fit_type": self.fit_type,
                            "mode": self.fit_mode,
                            "initial_tau": self.initial_tau,
                            "initial_dT": self.initial_dT,
                            "initial_C": self.initial_C,
                            "force_C_normalized": self.force_C_normalized,
                            "irf_iterations": self.irf_iterations,
                            "eps": self.eps,
                            "regularization": self.regularization,
                        }
                        if self.reference_type == "ref":
                            fit_kwargs["ref"] = reference_histogram
                            fit_kwargs["tau_ref"] = self.tau_ref
                        else:
                            fit_kwargs["irf"] = reference_histogram

                        try:
                            fit_result = Alignment.fit_data_with_ref_or_irf(**fit_kwargs)
                            irf_original = np.asarray(fit_result["irf"], dtype=float)
                            irf_calibrated = Alignment._normalize_histogram_1d(
                                Alignment.linear_shift(irf_original, fit_result["dT"], cyclic=True),
                                name="irf_calibrated",
                            )
                            fit_payload = {
                                "C": float(fit_result["C"]),
                                "tau": float(fit_result["tau"]),
                                "tau_ref": (
                                    np.nan
                                    if fit_result["tau_ref"] is None
                                    else float(fit_result["tau_ref"])
                                ),
                                "dT": float(fit_result["dT"]),
                                "dT_ns": float(fit_result["dT_ns"]),
                                "cov": np.asarray(fit_result["cov"], dtype=float),
                                "data_histogram": np.asarray(data_histogram, dtype=float),
                                "irf": np.asarray(irf_original, dtype=float),
                                "irf_calibrated": np.asarray(irf_calibrated, dtype=float),
                                "fit": np.asarray(fit_result["fit"], dtype=float),
                                "irf_source": str(fit_result["irf_source"]),
                            }
                            if self.reference_type == "ref":
                                fit_payload["ref_original"] = np.asarray(reference_histogram, dtype=float)
                                fit_payload["ref_calibrated"] = np.asarray(
                                    fit_result["ref_shifted"],
                                    dtype=float,
                                )
                        except Exception as exc:
                            warnings.warn(
                                (
                                    f"Calibration fit failed for data key {data_key!r}, channel "
                                    f"{channel_index}: {exc}"
                                ),
                                RuntimeWarning,
                                stacklevel=2,
                            )
                            fit_payload = self._empty_fit_payload(
                                nbin=nbin,
                                reference_type=self.reference_type,
                                data_histogram=data_histogram,
                                reference_histogram=reference_histogram,
                                irf_source="fit_failed",
                            )

                    stacked_channel_index.append(channel_index)
                    stacked_reference_channel_index.append(reference_channel_index)
                    stacked_C.append(float(fit_payload["C"]))
                    stacked_tau.append(float(fit_payload["tau"]))
                    stacked_tau_ref.append(float(fit_payload["tau_ref"]))
                    stacked_dT_bins.append(float(fit_payload["dT"]))
                    stacked_dT_ns.append(float(fit_payload["dT_ns"]))
                    stacked_irf_source.append(str(fit_payload["irf_source"]))
                    stacked_covariance.append(np.asarray(fit_payload["cov"], dtype=float))
                    stacked_data_histogram.append(np.asarray(fit_payload["data_histogram"], dtype=float))
                    stacked_irf_original.append(np.asarray(fit_payload["irf"], dtype=float))
                    stacked_irf_calibrated.append(np.asarray(fit_payload["irf_calibrated"], dtype=float))
                    stacked_fit.append(np.asarray(fit_payload["fit"], dtype=float))
                    if self.reference_type == "ref":
                        stacked_ref_original.append(np.asarray(fit_payload["ref_original"], dtype=float))
                        stacked_ref_calibrated.append(
                            np.asarray(fit_payload["ref_calibrated"], dtype=float)
                        )

                self._replace_dataset(
                    target_group,
                    "channel_index",
                    np.asarray(stacked_channel_index, dtype=int),
                )
                self._replace_dataset(
                    target_group,
                    "reference_channel_index",
                    np.asarray(stacked_reference_channel_index, dtype=int),
                )
                self._replace_dataset(target_group, "C", np.asarray(stacked_C, dtype=float))
                self._replace_dataset(target_group, "tau_ns", np.asarray(stacked_tau, dtype=float))
                self._replace_dataset(target_group, "tau_ref_ns", np.asarray(stacked_tau_ref, dtype=float))
                self._replace_dataset(
                    target_group,
                    "delta_t_correction_bins",
                    np.asarray(stacked_dT_bins, dtype=float),
                )
                self._replace_dataset(
                    target_group,
                    "delta_t_correction_ns",
                    np.asarray(stacked_dT_ns, dtype=float),
                )
                self._replace_dataset(
                    target_group,
                    "covariance",
                    np.stack(stacked_covariance, axis=-1),
                )
                self._replace_dataset(
                    target_group,
                    "data_histogram",
                    np.stack(stacked_data_histogram, axis=-1),
                )
                self._replace_dataset(
                    target_group,
                    "irf_original",
                    np.stack(stacked_irf_original, axis=-1),
                )
                self._replace_dataset(
                    target_group,
                    "irf_calibrated",
                    np.stack(stacked_irf_calibrated, axis=-1),
                )
                self._replace_dataset(
                    target_group,
                    "fit",
                    np.stack(stacked_fit, axis=-1),
                )
                if self.reference_type == "ref":
                    self._replace_dataset(
                        target_group,
                        "ref_original",
                        np.stack(stacked_ref_original, axis=-1),
                    )
                    self._replace_dataset(
                        target_group,
                        "ref_calibrated",
                        np.stack(stacked_ref_calibrated, axis=-1),
                    )
                string_dtype = h5py.string_dtype(encoding="utf-8")
                if "irf_source" in target_group:
                    del target_group["irf_source"]
                target_group.create_dataset(
                    "irf_source",
                    data=np.asarray(stacked_irf_source, dtype=object),
                    dtype=string_dtype,
                )

                if len(self.data_keys) == 1:
                    target_group.attrs["data_key_group_mode"] = "flat_root"
                else:
                    target_group.attrs["data_key_group_mode"] = "nested_under_calibration"

        return str(output_path)


def calibrate_h5_file(
    data_path,
    reference_path,
    data_key=DEFAULT_DATA_KEY,
    reference_key=DEFAULT_REFERENCE_KEY,
    reference_type=DEFAULT_REFERENCE_TYPE,
    tau_ref=DEFAULT_TAU_REF,
    fit_mode=DEFAULT_FIT_MODE,
    fit_type=DEFAULT_FIT_TYPE,
    C_ref=DEFAULT_C_REF,
    irf_iterations=DEFAULT_IRF_ITERATIONS,
    regularization=DEFAULT_REGULARIZATION,
    overwrite=DEFAULT_OVERWRITE,
    **kwargs,
):
    """
    Calibrate an HDF5 FLIM file against a reference file.

    This is a convenience wrapper around :class:`H5DataCalibrator` that exposes
    the most commonly used calibration parameters directly and forwards any
    extra keyword arguments to the class constructor.

    Parameters
    ----------
    data_path : str or path-like
        HDF5 file containing the data to calibrate.
    reference_path : str or path-like
        HDF5 file containing the reference histogram or IRF source.
    data_key : str or iterable of str, default ``("data", "data_channels_extra")``
        Dataset key or keys to calibrate from ``data_path``.
    reference_key : str or iterable of str or dict, default ``("data", "data_channels_extra")``
        Dataset key or keys to read from ``reference_path``. A dict can also be
        used to map each data key to a different reference key.
    reference_type : {"ref", "irf"}, default ``"ref"``
        Type of reference input used during calibration.
    tau_ref : float or None, default ``None``
        Reference lifetime in ns. If ``None``, it is estimated from the
        reference data when needed.
    fit_mode : str, default ``"model_shift"``
        Fitting mode forwarded to the alignment routines.
    fit_type : str, default ``"circular"``
        Fit geometry forwarded to the alignment routines.
    C_ref : float, default ``1.0``
        Reference amplitude scaling factor.
    irf_iterations : int, default ``300``
        Number of iterations used when estimating the IRF.
    regularization : float, default ``0``
        Regularization strength used during IRF estimation.
    overwrite : bool, default ``True``
        If ``True``, overwrite an existing output file.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`H5DataCalibrator`, including ``output_path=None``,
        ``channels=None``, ``calibration_key="calibration"``,
        ``period_ns=None``, ``initial_tau=None``, ``initial_dT=None``,
        ``initial_C=None``, ``force_C_normalized=False``, and ``eps=1e-8``.

    Returns
    -------
    str
        Path to the calibrated output HDF5 file.
    """
    return H5DataCalibrator(
        data_path,
        reference_path,
        data_key=data_key,
        reference_key=reference_key,
        reference_type=reference_type,
        tau_ref=tau_ref,
        fit_mode=fit_mode,
        fit_type=fit_type,
        C_ref=C_ref,
        irf_iterations=irf_iterations,
        regularization=regularization,
        overwrite=overwrite,
        **kwargs,
    ).calibrate()


def show_h5_structure(file_path, include_attrs=True, attrs_inline=False):
    """
    Return and print a readable tree view of an HDF5 file structure.

    Parameters
    ----------
    file_path : str or path-like
        HDF5 file to inspect.
    include_attrs : bool, default True
        If ``True``, include group and dataset attributes in the output.
    attrs_inline : bool, default False
        If ``True``, print attributes as indented ``.attrs.<name> = value``
        lines directly below each group or dataset.
    """
    lines = []

    def append_attrs(node, level, node_name=None):
        if not include_attrs:
            return
        attrs_items = list(node.attrs.items())
        if not attrs_items:
            return
        indent = "  " * level
        if attrs_inline:
            prefix = f"{node_name}.attrs" if node_name else ".attrs"
            joined = ", ".join(f"{key}={value!r}" for key, value in attrs_items)
            lines.append(f"{indent}{prefix}: {joined}")
            return
        for key, value in attrs_items:
            lines.append(f"{indent}@{key} = {value!r}")

    def visit(name, obj):
        if name == "":
            lines.append("/")
            append_attrs(obj, 1)
            return

        level = name.count("/")
        prefix = "  " * level
        node_name = name.split("/")[-1]
        if isinstance(obj, h5py.Group):
            lines.append(f"{prefix}{node_name}/")
            append_attrs(obj, level + 1, node_name=node_name)
        elif isinstance(obj, h5py.Dataset):
            lines.append(
                f"{prefix}{node_name} shape={obj.shape} dtype={obj.dtype}"
            )
            append_attrs(obj, level + 1, node_name=node_name)

    with h5py.File(file_path, "r") as handle:
        visit("", handle)
        handle.visititems(visit)

    structure = "\n".join(lines)
    print(structure)
    return structure


def show_h5_structure_html(file_path, include_attrs=True, attrs_inline=True, display_output=True):
    """
    Return an HTML tree view of an HDF5 file structure.

    Parameters
    ----------
    file_path : str or path-like
        HDF5 file to inspect.
    include_attrs : bool, default True
        If ``True``, include group and dataset attributes in the output.
    attrs_inline : bool, default True
        If ``True``, render all attributes for a node on one line.
    display_output : bool, default True
        If ``True`` and IPython is available, display the HTML immediately.
    """

    def format_value(value):
        value_repr = escape(repr(value))
        if isinstance(value, np.ndarray):
            return (
                f"{value_repr} "
                f"<span class='h5-array-shape'>shape={escape(str(value.shape))}</span>"
            )
        return value_repr

    def render_attrs(node, node_name):
        if not include_attrs or len(node.attrs) == 0:
            return ""

        items = list(node.attrs.items())
        if attrs_inline:
            joined = ", ".join(
                f"<span class='h5-attr-key'>{escape(str(key))}</span>="
                f"<span class='h5-attr-value'>{format_value(value)}</span>"
                for key, value in items
            )
            return (
                f"<div class='h5-attrs-inline'>"
                f"<span class='h5-attrs-prefix'>"
                f"<span class='h5-node-ref'>{escape(node_name)}.attrs</span>:"
                f"</span>"
                f"<span class='h5-attrs-content'>{joined}</span>"
                f"</div>"
            )

        parts = ["<ul class='h5-attrs-list'>"]
        for key, value in items:
            parts.append(
                "<li>"
                f"<span class='h5-node-ref'>{escape(node_name)}.attrs.</span>"
                f"<span class='h5-attr-key'>{escape(str(key))}</span>"
                f" = <span class='h5-attr-value'>{format_value(value)}</span>"
                "</li>"
            )
        parts.append("</ul>")
        return "".join(parts)

    def render_node(name, obj):
        node_name = name.split("/")[-1] if name else "/"
        if isinstance(obj, h5py.Dataset):
            label = (
                f"<span class='h5-dataset'>{escape(node_name)}</span> "
                f"<span class='h5-meta'>shape={escape(str(obj.shape))} "
                f"dtype={escape(str(obj.dtype))}</span>"
            )
            attrs_html = render_attrs(obj, node_name)
            return f"<li>{label}{attrs_html}</li>"

        label = f"<span class='h5-group'>{escape(node_name)}</span>/"
        attrs_html = render_attrs(obj, node_name)
        children = []
        for child_name in obj.keys():
            children.append(render_node(child_name, obj[child_name]))
        children_html = ""
        if children:
            children_html = f"<ul>{''.join(children)}</ul>"
        return (
            "<li>"
            "<details class='h5-branch' open>"
            f"<summary>{label}</summary>"
            f"{attrs_html}"
            f"{children_html}"
            "</details>"
            "</li>"
        )

    with h5py.File(file_path, "r") as handle:
        children = [render_node(name, handle[name]) for name in handle.keys()]
        root_attrs_html = render_attrs(handle, "/")

    html_output = f"""
<div class="h5-tree">
  <style>
    .h5-tree {{
      color-scheme: light dark;
      font-family: "Menlo", "Consolas", "DejaVu Sans Mono", monospace;
      font-size: 13px;
      line-height: 1.5;
      color: var(--h5-fg);
      --h5-fg: #1f2937;
      --h5-muted: #6b7280;
      --h5-border: #d1d5db;
      --h5-group: #0f766e;
      --h5-dataset: #1d4ed8;
      --h5-attrs: #7c2d12;
      --h5-node-ref: #7c3aed;
      --h5-attr-key: #b45309;
      --h5-attr-value: #374151;
      --h5-root: #111827;
    }}
    @media (prefers-color-scheme: dark) {{
      .h5-tree {{
        --h5-fg: #e5e7eb;
        --h5-muted: #9ca3af;
        --h5-border: #4b5563;
        --h5-group: #5eead4;
        --h5-dataset: #93c5fd;
        --h5-attrs: #fdba74;
        --h5-node-ref: #c4b5fd;
        --h5-attr-key: #fbbf24;
        --h5-attr-value: #f3f4f6;
        --h5-root: #f9fafb;
      }}
    }}
    .h5-tree ul {{
      list-style: none;
      margin: 0.2rem 0 0.2rem 1.1rem;
      padding-left: 1rem;
      border-left: 1px solid var(--h5-border);
    }}
    .h5-tree li {{
      margin: 0.2rem 0;
    }}
    .h5-tree summary {{
      cursor: pointer;
      list-style: none;
    }}
    .h5-tree summary::-webkit-details-marker {{
      display: none;
    }}
    .h5-branch > summary::before {{
      content: "▾";
      color: var(--h5-muted);
      display: inline-block;
      width: 1rem;
    }}
    .h5-branch:not([open]) > summary::before {{
      content: "▸";
    }}
    .h5-group {{
      color: var(--h5-group);
      font-weight: 700;
    }}
    .h5-dataset {{
      color: var(--h5-dataset);
      font-weight: 700;
    }}
    .h5-meta {{
      color: var(--h5-muted);
      font-weight: 500;
    }}
    .h5-attrs-inline, .h5-attrs-list {{
      margin-top: 0.15rem;
      color: var(--h5-attrs);
    }}
    .h5-attrs-inline {{
      display: grid;
      grid-template-columns: max-content 1fr;
      column-gap: 0.45rem;
      align-items: start;
    }}
    .h5-attrs-prefix {{
      white-space: nowrap;
    }}
    .h5-attrs-content {{
      min-width: 0;
      overflow-wrap: anywhere;
    }}
    .h5-node-ref {{
      color: var(--h5-node-ref);
      font-weight: 700;
    }}
    .h5-attr-key {{
      color: var(--h5-attr-key);
      font-weight: 700;
    }}
    .h5-attr-value {{
      color: var(--h5-attr-value);
    }}
    .h5-array-shape {{
      color: var(--h5-muted);
      font-weight: 500;
    }}
    .h5-root {{
      color: var(--h5-root);
      font-weight: 800;
    }}
  </style>
  <div class="h5-root">/</div>
  {root_attrs_html}
  <ul>
    {''.join(children)}
  </ul>
</div>
""".strip()

    if display_output:
        try:
            from IPython.display import HTML, display
            display(HTML(html_output))
        except ImportError:
            pass

    return html_output
