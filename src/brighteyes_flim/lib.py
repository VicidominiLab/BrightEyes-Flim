import numpy as np
from matplotlib import colors
from matplotlib.pyplot import gca


from tqdm.auto import tqdm
import h5py

import brighteyes_ism.dataio.mcs as mcs


class FlimData:

    def __init__(self, data_path: str = None, freq_exc: float = 21e6, correction_coeff: complex = None,
                 pre_filter: str = None):
        '''

        :param data_path:
        :param freq_exc:
        :param correction_coeff:
        :param pre_filter: None, "normalize", float = noise removal threshold on normalized data - Usefully for IRF
        '''

        self.phasor_laser = 0.0j
        self.freq_exc = freq_exc

        # The source data
        self.data = None

        # The global histogram (for each ch.)
        self.data_hist = None

        # The laser data
        self.data_laser = None

        # The metadata
        self.metadata = None

        # The Calibration reference phasor (complex notation)
        self.calib_ref = None

        # Full phasors
        self.phasors = None

        # Phasor of the global histogram (for each ch.)
        self.phasors_global = None

        # Full phasors corrected
        # self.phasors_corrected = None

        self.correction_coeff = None

        if data_path is not None:
            self.load_data(data_path)

        if pre_filter is not None:

            if isinstance(pre_filter, str):
                if pre_filter == "normalize":
                    self.data_hist = ((self.data_hist - np.nanmin(self.data_hist.T, axis=1)) /
                                      (np.nanmax(self.data_hist.T, axis=1) - np.nanmin(self.data_hist.T, axis=1)))
            elif type(pre_filter) is int or type(pre_filter) is float:
                threshold = pre_filter
                nnn = self.data_hist / np.max(self.data_hist, axis=0)
                nnn[nnn < threshold] = 1. / np.max(self.data_hist[nnn < threshold], axis=0)
                self.data_hist = nnn
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

        self.calculate_phasor_global()
        self.calculate_phasor_laser()

        print("Global", self.phasors_global)
        print("Laser", self.phasor_laser)

    def load_data(self, data_path: str):
        self.data, self.metadata = mcs.load(data_path)
        data_extra, _ = mcs.load(data_path, key="data_channels_extra")
        data_laser = data_extra[:, :, :, :, :, 1]
        self.data_laser = np.sum(data_laser, axis=(0, 1, 2, 3))
        self.data_hist = np.sum(self.data, axis=(0, 1, 2, 3))

    def calculate_phasor_on_img_ch(self, merge_adjacent_pixel=1):
        self.phasors = calculate_phasor_on_img_ch(self.data, merge_pixels=merge_adjacent_pixel)
        return self.phasors

    def calculate_phasor_global(self):
        self.phasors_global = calculate_phasor(self.data_hist)
        return self.phasors_global

    def calculate_phasor_laser(self):
        self.phasor_laser = calculate_phasor(self.data_laser)
        return self.phasor_laser

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

    def phasors_corrected(self, coeff=1., laser_correction=True):
        if self.phasors is None:
            raise (ValueError("call before calculate_phasor_on_img_ch(...)"))

        if laser_correction:
            return self.phasors * coeff * np.exp(1j*(-np.angle(self.phasor_laser)))
        else:
            return self.phasors * coeff

    def phasor_global_corrected(self, coeff=1., laser_correction=True):
        if laser_correction:
            return self.phasors_global * coeff * np.exp(1j*(-np.angle(self.phasor_laser)))
        else:
            return self.phasors_global * coeff

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


def sum_adjacent_pixel(data, n=4):
    if n<=1:
        return data
    print("original data.shape", data.shape)
    if len(data.shape) == 2:
        data = data[:,0:((data.shape[0]//n)*n)]
        data = data[0:((data.shape[1] // n) * n),:]
        print("reduced data.shape", data.shape)
        assert (data.shape[0] % n == 0)
        assert (data.shape[1] % n == 0)
        d = data[0::n, ::n]
        for i in range(1, n):
            d = d + data[i::n, i::n]
        print("merged data.shape", data.shape)
        return d
    elif len(data.shape) == 6:
        data = data[:,:,:,0:((data.shape[2]//n)*n),:,:]
        data = data[:,:,0:((data.shape[3] // n) * n),:,:,:]
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
        N = np.sum(data)
        if N < threshold:
            return np.nan + 1j * np.nan
        else:
            f = np.fft.fft(data / N)  # FFT of normalized data
            return f[harmonic].conj()  # 1st harmonic
    elif len(data.shape) == 2:
        N = np.sum(data, axis=0)
        print(N.shape, data.shape)
        f = np.fft.fft(data / N, axis=0)  # FFT of normalized data
        print(f.shape)
        out = f[harmonic]
        out[N < threshold * np.ones(data.shape[1])] = np.nan + 1j * np.nan
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


def calculate_phasor_on_img_ch(data_input, threshold=1, harmonic=1, merge_pixels=1):
    '''

    :param data_input: data_input [r, z, y, x, t, ch]
    :param threshold:
    :param harmonic:
    :param merge_pixels:

    :return: out G=[r,z,y,x,y,ch,0] S=[r,z,y,x,y,ch,1]
    '''

    if merge_pixels > 1:
        data_input = sum_adjacent_pixel(data_input, merge_pixels)
    else:
        pass

    data_input_shape = data_input.shape
    if len(data_input_shape) != 6:
        raise ValueError("The shape data_input [r, z, y, x, t, ch]")

    print("data_input.shape", data_input.shape)

    out = np.zeros(data_input.shape[:-2] + (data_input.shape[-1],), dtype=complex)
    print("out.shape", out.shape)
    for cc in tqdm(np.arange(data_input.shape[-1])):
        for rr in tqdm(np.arange(data_input.shape[-6]), leave=False):
            for zz in tqdm(np.arange(data_input.shape[-5]), leave=False):
                for yy in tqdm(np.arange(data_input.shape[-4]), leave=False):
                    for xx in np.arange(data_input.shape[-3]):
                        out[rr, zz, yy, xx, cc] = calculate_phasor(data_input[rr, zz, yy, xx, :, cc],
                                                                   threshold,
                                                                   harmonic)

    if len(data_input_shape) == 3:
        out = out[0, 0, :, :, :]
    elif len(data_input_shape) == 4:
        out = out[0, :, :, :, :]

    return out


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


def plot_tau(list_value=None, dfd_freq=21e6,ax=None):
    '''
    :param list_value: values to draw on the universal circle, default = np.arange(10) * 1e-9
    :param dfd_freq:
    :return:
    '''

    if ax is None:
        ax = gca()

    if list_value is None:
        list_value = np.array([1,2,3,4,5,7,9,12,18]) * 1e-9

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


def plot_phasor(phasors, bins_2dplot=50, log_scale=True, draw_universal_circle=True, tau_labels=True, ax=None, dfd_freq=21e6, cmap='viridis'):
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


def linear_shift(data, shift, cyclic=True):
    xp = np.arange(0, data.shape[0])
    fp = data
    x = np.arange(0, data.shape[0])-shift
    if cyclic:
        x = np.mod(x, data.shape[0])
    return np.interp(x, xp, fp)

