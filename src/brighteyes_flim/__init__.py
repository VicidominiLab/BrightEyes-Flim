"""Public package API for BrightEyes FLIM utilities."""

from .alignment import Alignment
from .flism_data import FlimData
from .h5_data_calibrator import (
    H5DataCalibrator,
    calibrate_h5_file,
    show_h5_structure,
    show_h5_structure_html,
)
from .tools_phasor import (
    calculate_m_phi_tau_phi_tau_m,
    calculate_tau_m,
    calculate_tau_phi,
    estimate_lifetime_from_birfi,
    estimate_lifetime_from_circmean,
    estimate_lifetime_from_log,
    linear_shift,
    plot_phasor,
    plot_tau,
    plot_universal_circle,
)
from brighteyes_ism.dataio import mcs


IRF_from_data_deconvolution = Alignment.IRF_from_data_deconvolution
curve_fit_circular = Alignment.curve_fit_circular
fit_data_with_ref_or_irf = Alignment.fit_data_with_ref_or_irf
fit_model_data = Alignment.fit_model_data
hist_for_plot = Alignment.hist_for_plot
model_data = Alignment.model_data
perform_fit_data = Alignment.perform_fit_data
phasor_delay_from_hist = Alignment.phasor_delay_from_hist
rectangular_IRF = Alignment.rectangular_IRF

__all__ = [
    "Alignment",
    "FlimData",
    "H5DataCalibrator",
    "IRF_from_data_deconvolution",
    "calculate_m_phi_tau_phi_tau_m",
    "calculate_tau_m",
    "calculate_tau_phi",
    "estimate_lifetime_from_birfi",
    "estimate_lifetime_from_circmean",
    "estimate_lifetime_from_log",
    "calibrate_h5_file",
    "curve_fit_circular",
    "fit_data_with_ref_or_irf",
    "fit_model_data",
    "hist_for_plot",
    "linear_shift",
    "mcs",
    "model_data",
    "perform_fit_data",
    "phasor_delay_from_hist",
    "plot_phasor",
    "plot_tau",
    "plot_universal_circle",
    "rectangular_IRF",
    "show_h5_structure",
    "show_h5_structure_html",
]
