import numpy as np
from matplotlib import colors
from matplotlib.pyplot import gca
from matplotlib.colors import hsv_to_rgb
from matplotlib_scalebar.scalebar import ScaleBar

from tqdm.auto import tqdm

import h5py
import matplotlib.pyplot as plt

import brighteyes_ism.dataio.mcs as mcs
from skimage.filters import threshold_otsu as otsu


class FlimData:

    def __init__(self, data_path: str = None, data_path_irf: str = None, data_path_ttm: str = None,
                 freq_exc: float = 21e6, correction_coeff: complex = None,
                 step_size: int = None, pre_filter: str = None, ignore_laser: bool = False):
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
        self.ignore_laser = ignore_laser
        # The source data
        self.data = None

        self.data_irf = None

        # sliced data along xy dimensions
        self.sliced_data = None

        # The global histogram (for each ch.)
        self.data_hist = np.zeros((120, 25))

        self.data_hist_irf = np.zeros((120, 25))

        # The laser data histogram (26th channel)
        self.data_laser_hist = np.zeros(120)
        self.data_laser_hist_irf = np.zeros(120)

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

        if data_path is not None and not ignore_laser:
            self.load_data_irf(data_path_irf, sub_image_size=100)
            self.load_data(data_path, sub_image_size=100)
            self.calculate_phasor_global_irf()
            self.calculate_phasor_global()
            self.calculate_phasor_laser()
            self.calculate_phasor_laser_irf()
            self.save_aligned_histogram_per_pixel(data_path, sub_image_size=100)

        else:
            print("CIAO TTM")
            self.save_histogram_decay_ttm(data_path_ttm, sub_image_dim=256)

        #  self.hf = h5py.File('dataset_of_pixel_histogram_aligned', "r")
        # self.image_with_histogram_realigned_in_each_pixel = self.hf["h5_dataset"]
        # self.calculate_phasor_on_img_pixel(self.image_with_histogram_realigned_in_each_pixel)

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

        if correction_coeff is not None:
            if isinstance(correction_coeff, list) or \
                    isinstance(correction_coeff, np.ndarray):
                self.correction_coeff = correction_coeff[0] + 1j * correction_coeff[1]
            if isinstance(correction_coeff, complex):
                self.correction_coeff = correction_coeff

        # print("Global", self.phasors_global)
        # print("Laser", self.phasor_laser)

    def load_data(self, data_path: str, sub_image_size: int = 100):
        self.data = h5py.File(data_path)
        self.metadata = mcs.metadata_load(data_path)  # data_format = 'h5'
        data_extra, _ = mcs.load(data_path, key="data_channels_extra")
        data_laser = data_extra[:, :, :, :, :, 1]
        image = self.data["data"]

        x_size, y_size = image.shape[2], image.shape[3]

        for x_start in range(0, x_size, sub_image_size):
            for y_start in range(0, y_size, sub_image_size):
                x_stop = min(x_start + sub_image_size, x_size)
                y_stop = min(y_start + sub_image_size, y_size)

                slice_term = np.s_[:, :, x_start:x_stop:self.step_size, y_start:y_stop:self.step_size, :, :]
                slice_term_laser = np.s_[:, :, x_start:x_stop:self.step_size, y_start:y_stop:self.step_size, :]
                sub_image = image[slice_term]
                sub_image_laser = data_laser[slice_term_laser]

                # Calculate data_hist for the current sub-image
                sub_data_hist = np.sum(sub_image, axis=(0, 1, 2, 3))
                sub_data_hist_laser = np.sum(sub_image_laser, axis=(0, 1, 2, 3))
                self.data_laser_hist += sub_data_hist_laser
                self.data_hist += np.array([sub_data_hist[:, i] for i in range(0, image.shape[5])]).T

    # else:
    #   self.data, self.metadata = mcs.load(data_path)

    #  x_size, y_size = self.data.shape[0], self.data.shape[1]

    # for x_start in range(0, x_size, sub_image_size):
    #    for y_start in range(0, y_size, sub_image_size):
    #       x_stop = min(x_start + sub_image_size, x_size)
    #      y_stop = min(y_start + sub_image_size, y_size)

    #     slice_term = np.s_[:, :, x_start:x_stop:self.step_size, y_start:y_stop:self.step_size, :, :]
    #    sub_image = self.data[slice_term]

    # Calculate data_hist for the current sub-image
    #   sub_data_hist = np.sum(sub_image, axis=(0, 1))

    #  self.data_hist_ttm += np.array([sub_data_hist[:, i] for i in range(0, self.data.shape[3])]).T

    def load_data_irf(self, data_path_irf: str, sub_image_size: int = 100):
        self.data_irf = h5py.File(data_path_irf)
        self.metadata_irf = mcs.metadata_load(data_path_irf)  # data_format = 'h5'
        data_extra_irf, _ = mcs.load(data_path_irf, key="data_channels_extra")
        data_laser_irf = data_extra_irf[:, :, :, :, :, 1]
        image_irf = self.data_irf["data"]

        x_size_irf, y_size_irf = image_irf.shape[2], image_irf.shape[3]

        for x_start in range(0, x_size_irf, sub_image_size):
            for y_start in range(0, y_size_irf, sub_image_size):
                x_stop = min(x_start + sub_image_size, x_size_irf)
                y_stop = min(y_start + sub_image_size, y_size_irf)

                slice_term_irf = np.s_[:, :, x_start:x_stop:self.step_size, y_start:y_stop:self.step_size, :, :]
                slice_term_laser_irf = np.s_[:, :, x_start:x_stop:self.step_size, y_start:y_stop:self.step_size, :]
                sub_image_irf = image_irf[slice_term_irf]
                sub_image_laser_irf = data_laser_irf[slice_term_laser_irf]

                # Calculate data_hist for the current sub-image
                sub_data_hist_irf = np.sum(sub_image_irf, axis=(0, 1, 2, 3))
                sub_data_hist_laser_irf = np.sum(sub_image_laser_irf, axis=(0, 1, 2, 3))
                self.data_laser_hist_irf += sub_data_hist_laser_irf
                self.data_hist_irf += np.array([sub_data_hist_irf[:, i] for i in range(0, image_irf.shape[5])]).T

    # else:
    #    self.data_irf, self.metadata_irf = mcs.load(data_path_irf)

    #   x_size, y_size = self.data_irf.shape[0], self.data_irf.shape[1]

    #  for x_start in range(0, x_size, sub_image_size):
    #     for y_start in range(0, y_size, sub_image_size):
    #        x_stop = min(x_start + sub_image_size, x_size)
    #       y_stop = min(y_start + sub_image_size, y_size)

    #      slice_term_irf = np.s_[:, :, x_start:x_stop:self.step_size, y_start:y_stop:self.step_size, :, :]
    #     sub_image_irf = self.data_irf[slice_term_irf]

    # Calculate data_hist for the current sub-image
    #    sub_data_hist_irf = np.sum(sub_image_irf, axis=(0, 1))

    #   self.data_hist_irf_ttm += np.array([sub_data_hist_irf[:, i] for i in range(0, self.data_irf.shape[3])]).T

    def calculate_irf_correction(self):
        phasors_shifted = self.phasor_global_corrected_irf()
        phasor_forced = 1.
        phasor_total = self.phasor_global_corrected(phasor_forced / phasors_shifted)
        irf_term = np.angle(phasor_forced / phasors_shifted / phasor_total) / (2 * np.pi)
        return [irf_term, phasor_total, phasors_shifted]

    def save_aligned_histogram_per_pixel(self, data_path: str, sub_image_size: int, cyclic=True):
        [shift, phasor_on_channels, phasors_irf] = self.calculate_irf_correction()
        self.data = h5py.File(data_path)
        self.metadata = mcs.metadata_load(data_path)  # data_format = 'h5'
        image = self.data["data"]
        x_size, y_size, bin_size, channel_size = image.shape[2], image.shape[3], image.shape[4], image.shape[5]

        with h5py.File('dataset_of_pixel_histogram_aligned', 'w') as f:

            # Create an empty dataset with the specified dimensions in dataset_shape
            dataset_shape = (x_size, y_size, bin_size, channel_size)
            h5_dataset = f.create_dataset('h5_dataset', shape=dataset_shape, dtype=np.float32)
            h5_dataset[:] = np.zeros(dataset_shape, dtype=np.float32)

            for x_start in range(0, x_size, sub_image_size):
                for y_start in range(0, y_size, sub_image_size):
                    x_stop = min(x_start + sub_image_size, x_size)
                    y_stop = min(y_start + sub_image_size, y_size)

                    slice_term = np.s_[:, :, x_start:x_stop:self.step_size, y_start:y_stop:self.step_size, :, :]
                    sub_image = image[slice_term]
                    sub_image = sub_image[0, 0, :, :, :, :]

                    aligned_sub_image = np.zeros(sub_image.shape, dtype=np.float32)

                    for i in range(sub_image.shape[0]):
                        for j in range(sub_image.shape[1]):
                            aligned_sub_image[i, j, :, :] = np.array([linear_shift(sub_image[i, j, :, ch],
                                                                                   shift[ch], cyclic) for ch in
                                                                      range(0, image.shape[5])]).T

                    h5_dataset[x_start:x_stop:self.step_size, y_start:y_stop:self.step_size, :,
                    :] = aligned_sub_image
        f.close()
        # hf = h5py.File('dataset_of_pixel_histogram_aligned', "r")
        # image_realigned = hf["h5_dataset"]  # image with histograms realigned in each pixel
        # self.phasors = calculate_phasor_on_img_ch(image_realigned)
        # phasor_processed = self.phasors_corrected(1 / phasors_irf)
        # plot_phasor(phasor_processed, bins_2dplot=200, log_scale=False)
        # for i in range(0, 25):
        #   plt.plot(np.nanmean(np.real(phasor_on_channels[i])), np.nanmean(np.imag(phasor_on_channels[i])), ".r")
        #   m, phi, tau, tau_m = calculate_m_phi_tau_phi_tau_m(phasor_on_channels[i], dfd_freq=21e6)
        #  print("%d\tphi: %.2e\tm: %.2e\t\ttau_phi: %.2e \ttau_m: %.2e" % (i, phi, m, tau, tau_m))

        # plt.xlim(-1., 1.)
        # plt.ylim(-1., 1.)

    #  def calculate_phasor_on_img_ch(self, merge_adjacent_pixel=1):
    #     self.phasors = calculate_phasor_on_img_ch(self.data, merge_pixels=merge_adjacent_pixel)
    #    return self.phasors

    def save_histogram_decay_ttm(self, data_path_ttm: str, sub_image_dim: int):
        self.data = h5py.File(data_path_ttm)
        image = self.data["dataset_1"]
        x_size, y_size, bin_size, channel_size = image.shape[0], image.shape[1], image.shape[2], \
            image.shape[3]

        with h5py.File('dataset_of_pixel_histogram_aligned_ttm', 'w') as f:

            # Create an empty dataset with the specified dimensions in dataset_shape
            dataset_shape = (x_size, y_size, bin_size, channel_size)
            h5_dataset_ttm = f.create_dataset('h5_dataset_ttm', shape=dataset_shape, dtype=np.float32)
            h5_dataset_ttm[:] = np.zeros(dataset_shape, dtype=np.float32)

            for x_start in range(0, x_size, sub_image_dim):
                for y_start in range(0, y_size, sub_image_dim):
                    x_stop = min(x_start + sub_image_dim, x_size)
                    y_stop = min(y_start + sub_image_dim, y_size)

                    slice_term = np.s_[x_start:x_stop, y_start:y_stop, :, :]
                    sub_image = image[slice_term]
                    h5_dataset_ttm[x_start:x_stop, y_start:y_stop, :, :] = sub_image

        f.close()

    def calculate_phasor_global(self):
        self.phasors_global = calculate_phasor(self.data_hist)
        return self.phasors_global

    def calculate_phasor_global_irf(self):
        self.phasors_global_irf = calculate_phasor(self.data_hist_irf)
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

    def calculate_correction_coeff(self, dfd_freq=21e6, tau=10e-9):
        dfd_time = 1. / dfd_freq

        ppp = self.phasors_global / abs(self.phasors_global)

        theta = 2 * np.pi * tau / dfd_time
        DeltaM = np.abs(self.phasors_global) / np.abs(np.cos(theta))

        DeltaPhi = np.exp(1j * theta)
        coeff = 1. / (ppp * DeltaM * DeltaPhi)

        self.correction_coeff = coeff

        return self.correction_coeff

    # def phasors_corrected(self, coeff=1., laser_correction=True):
    #    if self.phasors is None:
    #       raise (ValueError("call before calculate_phasor_on_img_ch(...)"))

    #  if laser_correction:
    #     return self.phasors * coeff * np.exp(1j * (-np.angle(self.phasor_laser)))
    # else:
    #    return self.phasors * coeff

    def phasor_global_corrected(self, coeff=1., laser_correction=True):
        if laser_correction:
            return self.phasors_global * coeff * np.exp(1j * (-np.angle(self.phasor_laser)))
        else:
            return self.phasors_global * coeff

    def phasor_global_corrected_irf(self, coeff=1., laser_correction=True):
        if laser_correction:
            return self.phasors_global_irf * coeff * np.exp(1j * (-np.angle(self.phasor_laser_irf)))
        else:
            return self.phasors_global_irf * coeff

    # def calculate_phasor_for_image_channels(self):
    # [shift, phasor_on_channels, phasors_irf] = self.calculate_irf_correction()
    #

    # plt.show()


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


def calculate_phasor(data_hist, threshold=1, harmonic=1):
    '''
    :param data_hist: the histogram 1D
    :param threshold: the minimum of entry of the histogram
    :return: phasor: g0 + 1j * s0
    '''

    data = data_hist
    if len(data.shape) == 1:
        data_thresholded = np.where(data < 10, data, 0)
        N = np.sum(data_thresholded)
        if N < threshold:
            return np.nan + 1j * np.nan
        else:
            f = np.fft.fft(data_thresholded / N)  # FFT of normalized data
            return f[harmonic].conj()  # 1st harmonic
    elif len(data.shape) == 2:
        data_thresholded = np.where(data < 10, data, 0)
        N = np.sum(data_thresholded, axis=0)
        print(N.shape, data_thresholded.shape)
        f = np.fft.fft(data_thresholded / N, axis=0)  # FFT of normalized data
        print(f.shape)
        out = f[harmonic]
        out[N < threshold * np.ones(data_thresholded.shape[1])] = np.nan + 1j * np.nan
        return out.conj()
    else:
        raise ValueError("Wrong input dimension")


def calculate_tau_phi(g0_or_complex, s0=None, dfd_freq=21e6):
    if s0 is None:
        phi = np.angle(g0_or_complex)
        m = np.abs(g0_or_complex)
    else:
        g0 = g0_or_complex
        phi = np.arctan2(s0, g0)
        m = np.sqrt(s0 ** 2 + g0 ** 2)

    tau_phi = np.tan(phi) / (2 * np.pi * dfd_freq)

    return tau_phi


def calculate_tau_m(g0_or_complex, s0=None, dfd_freq=21e6):
    if s0 is None:
        phi = np.angle(g0_or_complex)
        m = np.abs(g0_or_complex)
    else:
        g0 = g0_or_complex
        phi = np.arctan2(s0, g0)
        m = np.sqrt(s0 ** 2 + g0 ** 2)

    tau_m = np.sqrt((1. / (m ** 2)) - 1) / (2 * np.pi * dfd_freq)

    return tau_m


def calculate_m_phi_tau_phi_tau_m(g0_or_complex, s0=None, dfd_freq=21e6):
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
        h5_dataset_p[:] = np.zeros(h5_dim, dtype=np.complex128)

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
                            aligned_phasor[yy, xx, cc] = calculate_phasor(sub_image_p[yy, xx, :, cc],
                                                                          threshold,
                                                                          harmonic)
                h5_dataset_p[x_start:x_stop, y_start:y_stop,
                :] = aligned_phasor
    fi.close()


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
                    out[rr, zz, yy, xx] = calculate_phasor(data_input[rr, zz, yy, xx, :], threshold, harmonic)

    if len(data_input_shape) == 3:
        out = out[0, 0, :, :, :]
    elif len(data_input_shape) == 4:
        out = out[0, :, :, :, :]

    return out


def calculate_phasor_on_img_pixels(data_input, threshold=1, harmonic=1, phasor_pix_data_size=100):
    '''

    :param data_input: data_input [y, x, t] or [z, y, x, t] or [r, z, y, x, t]
    :param threshold:
    :param harmonic:

    :return: out G=[r,z,y,x,y,0] S=[r,z,y,x,y,1]
    '''
    data_input_3d = np.sum(data_input, axis=3)
    data_input_shape = data_input_3d.shape
    if len(data_input_shape) < 3:
        raise ValueError("data_input must have 3 or more dimensions")
    elif len(data_input_shape) >= 6:
        raise ValueError("use calculate_phasor_on_img_ch() instead")

    with h5py.File('dataset_of_phasor_per_pixel_ttm', 'w') as fil:

        # Create an empty dataset with the specified dimensions in dataset_shape
        x_dim, y_dim = data_input_3d.shape[0], data_input_3d.shape[1]
        h5_dim = (x_dim, y_dim)
        h5_dataset_phasor_pix = fil.create_dataset('h5_dataset_phasor_pix', shape=h5_dim, dtype=np.complex128)
        h5_dataset_phasor_pix[:] = np.zeros(h5_dim, dtype=np.complex128)

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
                        aligned_phasor_pix[yyy, xxx] = calculate_phasor(sub_image_pix[yyy, xxx, :],
                                                                        threshold,
                                                                        harmonic)
                h5_dataset_phasor_pix[x_start:x_stop, y_start:y_stop] = aligned_phasor_pix
    fil.close()


def plot_tau(list_value=None, dfd_freq=21e6, ax=None):
    '''
    :param list_value: values to draw on the universal circle, default = np.arange(10) * 1e-9
    :param dfd_freq:
    :return:
    '''

    if ax is None:
        ax = gca()

    if list_value is None:
        list_value = np.array([1, 2, 3, 4, 5, 7, 9, 12, 18]) * 1e-9

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
        ax.plot(x, y, ".-y")

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


def plot_phasor(phasors, bins_2dplot=50, log_scale=True, draw_universal_circle=True, tau_labels=True, ax=None,
                dfd_freq=21e6, cmap='viridis'):
    '''
    :param phasors:
    :param log_scale:
    :param bins_2dplot:
    :return:
    '''
    if ax is None:
        ax = gca()
    phasors_flat = flatten_and_remove_nan(phasors)

    if draw_universal_circle == True:
        plot_universal_circle(ax)

    if log_scale:
        _ = ax.hist2d(np.real(phasors_flat), np.imag(phasors_flat), range=[[-1, 1], [-1, 1]], bins=bins_2dplot,
                      norm=colors.LogNorm(), cmap=cmap)
    else:
        _ = ax.hist2d(np.real(phasors_flat), np.imag(phasors_flat), range=[[-1, 1], [-1, 1]], bins=bins_2dplot,
                      cmap=cmap)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    if tau_labels == True:
        plot_tau(ax=ax, dfd_freq=dfd_freq)


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


def showFLIM(
        intensity,
        lifetime,
        bounds_tau=None,
        bounds_int=None,
        satFactor=0.657,
        outOfBoundsHue=0.8,
        invertColormap=False,
        fig=None,
):
    """
    Display together the lifetime and intensity images with a proper colormap.
    Referring to the HSV color model:
    Intensity values are mapped in Value
    Lifetime values are mapped in Hue
    If the lifetime of a pixel is outside the interval [minInt, maxInt],
    that pixel is rendered with Hue outOfBoundsHue (default: violet)

    Input parameters
    intensityIm:        Pixel values are photon counts.
    lifetimeIm:         Pixel values are lifetime values.

    bounds_Tau
        minTau:             minimum lifetime value of the colorbar
        maxTau:             maximum lifetime value of the colorbar
    bounds_Int
        minInt:             minimum intensity value of the colorbar
        maxInt:             maximum intensity value of the colorbar
    invertColormap:     (False)
    outOfBoundsHue:      Hue to render the out of bounds pixels (0.8)
    satFactor:           The span of the Hue space (0.657)

    Last Modified May 2022 by Alessandro Zunino
    Based on the MATLAB function written by Giorgio Tortarolo
    """
    histograms_filtered = otsu(image=intensity, nbins=50)
    hist_indexes = np.argwhere(intensity > histograms_filtered)

    if bounds_tau is None:
        tau_denoised = lifetime[hist_indexes[:, 0], hist_indexes[:, 1]]
        bounds_tau = {
            "minTau": np.min(tau_denoised),
            "maxTau": np.max(tau_denoised),
        }

    if bounds_int is None:
        histogram_denoised = intensity[hist_indexes[:, 0], hist_indexes[:, 1]]
        bounds_int = {"minInt": np.min(histogram_denoised), "maxInt": np.max(histogram_denoised )}

    params = bounds_tau.copy()
    params.update(bounds_int)

    params["invertColormap"] = invertColormap
    params["satFactor"] = satFactor
    params["outOfBoundsHue"] = outOfBoundsHue

    sz = intensity.shape
    N = sz[0]

    # Image

    RGB = cmap2d(intensity, lifetime, params)

    # Colorbar

    LG = np.linspace(params["minTau"], params["maxTau"], num=sz[0])
    LifeTimeGradient = np.tile(LG, (N, 1))

    IG = np.linspace(params["minInt"], params["maxInt"], num=N)
    IntensityGradient = np.transpose(np.tile(IG, (sz[0], 1)))

    RGB_colormap = cmap2d(IntensityGradient, LifeTimeGradient, params)
    RGB_colormap = np.moveaxis(RGB_colormap, 0, 1)

    # Show combined image with colorbar

    extent = (params["minInt"], params["maxInt"], params["minTau"], params["maxTau"])

    if fig is None:
        fig = plt.figure(figsize=(9, 8))

    widths = [0.05, 1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=2, nrows=1, width_ratios=widths, height_ratios=heights
    )

    ax = fig.add_subplot(spec[0, 1])

    ax.imshow(RGB)
    ax.axis("off")

    ax = fig.add_subplot(spec[0, 0])

    ax.imshow(RGB_colormap, origin="lower", aspect="auto", extent=extent)
    ax.set_xticks([params["minInt"], params["maxInt"]])
    ax.set_xlabel("Counts")
    ax.set_ylabel("Lifetime (ns)")

    if fig is None:
        plt.tight_layout()

    return RGB, RGB_colormap


def linear_shift(data, shift, cyclic=True):
    xp = np.arange(0, data.shape[0])
    fp = data
    x = np.arange(0, data.shape[0]) - shift
    if cyclic:
        x = np.mod(x, data.shape[0])
    return np.interp(x, xp, fp)
