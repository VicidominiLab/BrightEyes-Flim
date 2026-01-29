from matplotlib import colors
from matplotlib.pyplot import gca
from matplotlib.colors import hsv_to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import median_filter
from tqdm.auto import tqdm
import math

import h5py
import matplotlib.pyplot as plt
import os

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








