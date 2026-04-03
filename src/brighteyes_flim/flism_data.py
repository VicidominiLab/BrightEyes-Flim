"""Legacy FLIM data containers."""

import warnings

import h5py
import numpy as np

import brighteyes_ism.dataio.mcs as mcs

from .tools_phasor import calculate_phasor

__all__ = ["FlimData"]


class FlimData:
    """Container for global FLIM histograms and phasor-based calibration helpers."""

    def __init__(self, data_path: str = None, data_path_irf: str = None,
                 freq_exc: float = 41.48e6, correction_coeff: complex = None, step_size: int = None,
                 sub_image_dim: int = 100, pre_filter: str = None):
        """Load data, aggregate histograms, and initialize phasor state."""
        self.phasor_laser = 0.0j
        self.phasor_laser_irf = 0.0j
        self.freq_exc = freq_exc

        self.sub_image_dim = sub_image_dim
        self.data = None
        self.data_irf = None
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
        self.data_laser_hist = np.zeros(self.bin_number)
        self.data_laser_hist_irf = np.zeros(self.bin_number)
        self.data_laser_hist_non_sliced = None
        self.metadata = None
        self.metadata_irf = None
        self.calib_ref = None
        self.phasors = None
        self.phasors_global = None
        self.phasors_global_irf = None
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
                elif isinstance(pre_filter, (int, float)):
                    threshold = pre_filter
                    nnn = self.data_hist / np.max(self.data_hist, axis=0)
                    nnn[nnn < threshold] = 1. / np.max(self.data_hist[nnn < threshold], axis=0)
                    self.data_hist = nnn
                    mmm = self.data_hist_irf / np.max(self.data_hist_irf, axis=0)
                    mmm[mmm < threshold] = 1. / np.max(self.data_hist_irf[mmm < threshold], axis=0)
                    self.data_hist_irf = mmm
                else:
                    warnings.warn(
                        f"Ignoring unsupported pre_filter value {pre_filter!r}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            self.calculate_phasor_global_irf()
            self.calculate_phasor_global()
            self.calculate_phasor_laser()
            self.calculate_phasor_laser_irf()

        if correction_coeff is not None:
            if isinstance(correction_coeff, list) or \
                    isinstance(correction_coeff, np.ndarray):
                self.correction_coeff = correction_coeff[0] + 1j * correction_coeff[1]
            if isinstance(correction_coeff, complex):
                self.correction_coeff = correction_coeff

    def load_data(self, data_path: str):
        self.data = h5py.File(data_path)
        self.metadata = mcs.metadata_load(data_path)
        data_extra, _ = mcs.load(data_path, key="data_channels_extra")
        data_laser = data_extra[:, :, :, :, :, 1]
        image = self.data["data"]
        self.data_laser_hist = np.sum(data_laser, axis=(0, 1, 2, 3))
        self.data_hist = np.sum(image, axis=(0, 1, 2, 3))

    def load_data_irf(self, data_path_irf: str):
        self.data_irf = h5py.File(data_path_irf)
        self.metadata_irf = mcs.metadata_load(data_path_irf)
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
