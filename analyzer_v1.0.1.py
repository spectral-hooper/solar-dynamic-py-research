#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hinode SP Sigma-V Analyzer

This module provides a pipeline for analyzing Hinode Spectro-Polarimeter (SP) data.
It estimates the longitudinal magnetic field strength using the "Sigma-V" method
(separation of Stokes V peaks) with a fallback to the Weak Field Approximation (WFA).

Features:
    - Cross-calibration of wavelength (offset + linear scale) using reference lines.
    - Detection of Zeeman sigma-components in Stokes V profiles.
    - Monte-Carlo error estimation for magnetic field calculations.
    - Generation of diagnostic plots and CSV reports.

Original Architecture Preserved.
Refactored for readability and PEP 8 compliance.
"""

import os
import time
import argparse
import subprocess
import warnings
from typing import Tuple, Dict, Optional, Union, Any, List

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.signal import savgol_filter, find_peaks, fftconvolve
from scipy.stats import median_abs_deviation

import matplotlib
# Set backend to 'Agg' for headless environments (servers/CI)
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

DEFAULT_FITS_FILE = "SP3D20231104_210115.0C.fits"
DEFAULT_OUTPUT_PREFIX = "тестовые_результаты"

# Analysis Parameters
SPATIAL_AVERAGING_WINDOW = 1  # Number of spatial pixels to average
REFERENCE_WAVELENGTH_0 = 6302.5
LINE_LAB_1 = 6301.5
LINE_LAB_2 = 6302.5
LANDÉ_FACTOR_EFF = 2.5
ZEEMAN_CONSTANT_K = 4.67e-13

# Signal Processing Parameters
SMOOTHING_WINDOW_SIZE = 9
SMOOTHING_POLY_ORDER = 3
SEARCH_WINDOW_HALF_PIX = 14
SIGMA_TIGHT_WINDOW_PIX = 8
MIN_RELATIVE_AMPLITUDE = 0.01
NOISE_THRESHOLD_FACTOR = 4.0
EDGE_SAFETY_MARGIN_PIX = 2
MAX_REALISTIC_B_GAUSS = 5000.0

# Monte Carlo Parameters
DEFAULT_MC_ITERATIONS = 100

# External Milne-Eddington Solver Configuration
USE_ATLAS_REF = False
REF_ATLAS_WAV_PATH = None
REF_ATLAS_INTENSITY_PATH = None
ME_SOLVER_CMD_TEMPLATE = None
ME_TEMP_DIR_SUFFIX = "_ME_input"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_odd(number: int) -> int:
    """Ensures the given number is odd and at least 3 (useful for filter windows)."""
    n = int(number)
    if n % 2 == 0:
        n -= 1
    if n < 3:
        n = 3
    return n


def build_wavelength_axis(header: fits.Header, n_pixels: int) -> Tuple[np.ndarray, float, float, float]:
    """
    Constructs the wavelength array from FITS header WCS keywords.
    
    Returns:
        wavelengths (np.ndarray): Array of wavelength values.
        dispersion (float): Delta lambda per pixel.
        ref_val (float): Reference wavelength value.
        ref_pix (float): Reference pixel index.
    """
    crval = header.get('CRVAL1', header.get('CRVAL', None))
    crpix = header.get('CRPIX1', header.get('CRPIX', 1.0))
    cdelt = header.get('CDELT1', header.get('CD1_1', header.get('CDELT', None)))

    if crval is None or cdelt is None:
        # Fallback if WCS is missing: create a dummy centered axis
        wavelengths = np.linspace(REFERENCE_WAVELENGTH_0 - 0.5, REFERENCE_WAVELENGTH_0 + 0.5, n_pixels)
        cdelt = wavelengths[1] - wavelengths[0]
        crval = wavelengths[0]
        crpix = 1.0
    else:
        pixel_indices = np.arange(n_pixels, dtype=float)
        wavelengths = float(crval) + (pixel_indices + 1.0 - float(crpix)) * float(cdelt)

    return np.array(wavelengths), float(cdelt), float(crval if crval is not None else wavelengths[0]), float(crpix)


def calculate_mad_std(data: np.ndarray) -> float:
    """Calculates the standard deviation estimate using Median Absolute Deviation (MAD)."""
    if data is None or len(data) == 0:
        return 0.0
    # 1.4826 is the scaling factor for normal distribution consistency
    return 1.4826 * np.median(np.abs(data - np.median(data)))


def compute_cross_correlation_shift(observed: np.ndarray, reference: np.ndarray) -> int:
    """
    Computes the pixel shift between observed and reference arrays using FFT cross-correlation.
    """
    obs_centered = np.asarray(observed) - np.nanmean(observed)
    ref_centered = np.asarray(reference) - np.nanmean(reference)
    
    # Reverse reference for convolution to act as correlation
    correlation = fftconvolve(obs_centered, ref_centered[::-1], mode='same')
    shift = int(np.argmax(correlation) - (len(correlation) // 2))
    return shift


def robust_find_line_center(wavelengths: np.ndarray, intensity: np.ndarray, 
                            target_lambda: float, window_angstrom: float = 0.6) -> Optional[int]:
    """
    Finds the pixel index of the minimum intensity (line center) near a target wavelength.
    """
    n_pixels = len(wavelengths)
    if n_pixels < 3:
        return None

    delta_lambda = abs(wavelengths[1] - wavelengths[0])
    half_window_pix = max(2, int(round(window_angstrom / delta_lambda)))
    
    # Initial guess based on wavelength axis
    pixel_guess = int(np.argmin(np.abs(wavelengths - target_lambda)))
    
    # Define search region boundaries
    start_idx = max(0, pixel_guess - half_window_pix)
    end_idx = min(n_pixels, pixel_guess + half_window_pix + 1)

    if end_idx - start_idx < 3:
        return None

    region_intensity = intensity[start_idx:end_idx]
    local_min_idx = int(np.argmin(region_intensity))
    
    return start_idx + local_min_idx


def solve_linear_wavelength_scale(observed_positions: List[float], 
                                  lab_positions: List[float]) -> Tuple[float, float]:
    """
    Solves for linear scaling coefficients (slope 'a' and intercept 'b') 
    such that: lambda_calibrated = a * lambda_raw + b
    """
    obs_arr = np.asarray(observed_positions)
    lab_arr = np.asarray(lab_positions)
    
    if len(obs_arr) < 2:
        return 1.0, 0.0
        
    # Least squares solution: [obs, 1] @ [a, b].T = lab
    design_matrix = np.vstack([obs_arr, np.ones_like(obs_arr)]).T
    solution, _, _, _ = np.linalg.lstsq(design_matrix, lab_arr, rcond=None)
    
    scale_factor = float(solution[0])
    offset = float(solution[1])
    return scale_factor, offset


def calculate_parabolic_centroid(y_data: np.ndarray, peak_index: int) -> float:
    """
    Refines the peak position to sub-pixel accuracy using parabolic interpolation
    around the integer peak index.
    """
    idx = int(round(peak_index))
    if idx <= 0 or idx >= len(y_data) - 1:
        return float(idx)

    y_minus = y_data[idx - 1]
    y_center = y_data[idx]
    y_plus = y_data[idx + 1]

    denominator = (y_minus - 2 * y_center + y_plus)
    if denominator == 0:
        return float(idx)

    correction = 0.5 * (y_minus - y_plus) / denominator
    return float(idx) + correction


# =============================================================================
# CORE ALGORITHM: SIGMA-V DETECTION
# =============================================================================

def analyze_sigma_v_on_spectrum(wavelengths: np.ndarray, 
                                stokes_i: np.ndarray, 
                                stokes_v: np.ndarray, 
                                center_pixel_idx: int,
                                smooth_window: int = SMOOTHING_WINDOW_SIZE, 
                                smooth_poly: int = SMOOTHING_POLY_ORDER,
                                search_half_width: int = SEARCH_WINDOW_HALF_PIX, 
                                noise_region_width: int = SIGMA_TIGHT_WINDOW_PIX,
                                min_rel_velocity: float = MIN_RELATIVE_AMPLITUDE, 
                                noise_factor: float = NOISE_THRESHOLD_FACTOR) -> Dict[str, Any]:
    """
    Analyzes a single Stokes V spectrum to find sigma-component peaks and estimate B.
    
    Returns a dictionary containing detection status, estimated field (B_G), 
    wavelengths of peaks, noise stats, and quality flags.
    """
    n_pixels = len(wavelengths)
    result = {
        "found": False, "B_G": None, "wa": None, "wb": None, 
        "a_sub": None, "b_sub": None, "delta_lambda": None, 
        "V_rel": None, "noise": None, "SNR": None,
        "suspect": False, "suspect_reason": None
    }

    if n_pixels < 5:
        result["suspect"] = True
        result["suspect_reason"] = "insufficient_pixels"
        return result

    # 1. Smoothing
    window_len = ensure_odd(min(smooth_window, max(3, n_pixels - 1)))
    if n_pixels >= window_len:
        try:
            v_smoothed = savgol_filter(stokes_v, window_len, smooth_poly)
        except Exception:
            v_smoothed = stokes_v.copy()
    else:
        v_smoothed = stokes_v.copy()

    # 2. Amplitude and Noise Estimation
    # Continuum estimate using edges
    continuum_level = 0.5 * (np.median(stokes_i[:max(1, n_pixels // 12)]) + 
                             np.median(stokes_i[-max(1, n_pixels // 12):]))
    
    v_amplitude = np.nanmax(np.abs(v_smoothed))
    relative_v = float(v_amplitude / max(abs(continuum_level), 1.0))
    result["V_rel"] = relative_v

    # Define regions
    search_start = max(0, center_pixel_idx - search_half_width)
    search_end = min(n_pixels, center_pixel_idx + search_half_width)
    
    inner_start = max(0, center_pixel_idx - noise_region_width)
    inner_end = min(n_pixels, center_pixel_idx + noise_region_width)

    # Estimate noise from outer regions (excluding the line core)
    mask_noise = np.ones_like(v_smoothed, dtype=bool)
    mask_noise[inner_start:inner_end] = False
    
    noise_samples = v_smoothed[mask_noise] if np.any(mask_noise) else v_smoothed
    noise_level = calculate_mad_std(noise_samples) if noise_samples.size > 4 else calculate_mad_std(v_smoothed)
    
    result["noise"] = float(noise_level)
    snr = v_amplitude / (noise_level if noise_level > 0 else 1e-12)
    result["SNR"] = float(snr)

    # 3. Validation Checks
    if relative_v < min_rel_velocity or snr < 1.0:
        result["suspect"] = True
        result["suspect_reason"] = "V_too_weak_or_low_SNR"
        return result

    # 4. Peak Finding
    prominence_threshold = max(noise_factor * noise_level, 0.01 * v_amplitude)
    
    # Find positive and negative peaks in the search window
    search_region_v = v_smoothed[search_start:search_end]
    
    pos_peaks_local, _ = find_peaks(search_region_v, prominence=prominence_threshold)
    neg_peaks_local, _ = find_peaks(-search_region_v, prominence=prominence_threshold)
    
    pos_peaks_global = (pos_peaks_local + search_start).tolist()
    neg_peaks_global = (neg_peaks_local + search_start).tolist()

    # 5. Pairing Peaks
    valid_pairs = []
    for p_idx in pos_peaks_global:
        for n_idx in neg_peaks_global:
            # Ensure peaks are somewhat separated
            if abs(p_idx - n_idx) < 2:
                continue
            # Ensure peaks are on opposite sides of the line center
            if (p_idx - center_pixel_idx) * (n_idx - center_pixel_idx) < 0:
                valid_pairs.append((p_idx, n_idx))

    if not valid_pairs:
        result["suspect"] = True
        result["suspect_reason"] = "no_opposite_V_peaks"
        return result

    # Select the "strongest" pair (highest sum of absolute amplitudes)
    best_pair = max(valid_pairs, key=lambda pair: abs(v_smoothed[pair[0]]) + abs(v_smoothed[pair[1]]))
    idx_a, idx_b = sorted(best_pair)

    # 6. Sub-pixel Refinement
    # Use positive V for one, negative V (inverted) for the other to fit parabola
    pos_a_sub = calculate_parabolic_centroid(v_smoothed if v_smoothed[idx_a] >= 0 else -v_smoothed, idx_a)
    pos_b_sub = calculate_parabolic_centroid(v_smoothed if v_smoothed[idx_b] >= 0 else -v_smoothed, idx_b)

    # Interpolate wavelength at sub-pixel positions
    wavelength_a = float(np.interp(pos_a_sub, np.arange(n_pixels), wavelengths))
    wavelength_b = float(np.interp(pos_b_sub, np.arange(n_pixels), wavelengths))

    # 7. Magnetic Field Calculation
    delta_lambda = abs(wavelength_b - wavelength_a) / 2.0
    denominator = (ZEEMAN_CONSTANT_K * LANDÉ_FACTOR_EFF * (REFERENCE_WAVELENGTH_0**2))
    
    if denominator == 0:
        result["suspect"] = True
        result["suspect_reason"] = "zero_denominator_error"
        return result

    magnetic_field_gauss = delta_lambda / denominator

    # 8. Final Checks
    is_edge_issue = False
    if (idx_a <= EDGE_SAFETY_MARGIN_PIX) or (idx_b <= EDGE_SAFETY_MARGIN_PIX) or \
       (idx_a >= n_pixels - 1 - EDGE_SAFETY_MARGIN_PIX) or (idx_b >= n_pixels - 1 - EDGE_SAFETY_MARGIN_PIX):
        is_edge_issue = True

    if abs(pos_b_sub - pos_a_sub) < 1.5:
        result["suspect"] = True
        result["suspect_reason"] = "peaks_too_close"
        return result

    result.update({
        "found": True,
        "B_G": float(magnetic_field_gauss),
        "wa": wavelength_a,
        "wb": wavelength_b,
        "a_sub": float(pos_a_sub),
        "b_sub": float(pos_b_sub),
        "delta_lambda": float(delta_lambda)
    })

    if is_edge_issue:
        result["suspect"] = True
        result["suspect_reason"] = "peak_on_edge"
    
    if abs(magnetic_field_gauss) > MAX_REALISTIC_B_GAUSS:
        result["suspect"] = True
        result["suspect_reason"] = "B_out_of_range"
        result["B_G"] = None
        result["found"] = False

    return result


# =============================================================================
# MONTE CARLO ERROR ESTIMATION
# =============================================================================

def estimate_b_error_mc(wavelengths: np.ndarray, 
                        stokes_i: np.ndarray, 
                        stokes_v: np.ndarray, 
                        center_pixel_idx: int, 
                        analysis_function: callable, 
                        n_iterations: int = 200, 
                        noise_estimate: float = None, 
                        show_progress: bool = False) -> Dict[str, Any]:
    """
    Estimates the uncertainty of the B-field measurement using Monte Carlo simulation.
    Injects Gaussian noise into the Stokes V profile and re-runs the analysis.
    """
    n_pixels = len(wavelengths)
    
    # Determine noise level if not provided
    if noise_estimate is None:
        inner_start = max(0, center_pixel_idx - SIGMA_TIGHT_WINDOW_PIX)
        inner_end = min(n_pixels, center_pixel_idx + SIGMA_TIGHT_WINDOW_PIX)
        
        mask_noise = np.ones_like(stokes_v, dtype=bool)
        mask_noise[inner_start:inner_end] = False
        
        noise_samples = stokes_v[mask_noise] if np.any(mask_noise) else stokes_v
        sigma_noise = calculate_mad_std(noise_samples)
    else:
        sigma_noise = float(noise_estimate)

    empty_result = {
        "B_median": None, "B_mean": None, "B_std": None, 
        "B_p16": None, "B_p84": None, "N_success": 0
    }

    if sigma_noise <= 0:
        return empty_result

    b_values = []
    rng = np.random.default_rng()

    for i in range(n_iterations):
        # Inject noise
        v_perturbed = stokes_v + rng.normal(0.0, sigma_noise, size=stokes_v.shape)
        
        # Re-analyze
        result = analysis_function(wavelengths, stokes_i, v_perturbed, center_pixel_idx)
        
        if result.get("found") and result.get("B_G") is not None:
            b_values.append(result["B_G"])
        
        if show_progress and (i % max(1, n_iterations // 10) == 0):
            print(f"[MC] Iteration {i}/{n_iterations}, successes: {len(b_values)}")

    if not b_values:
        return empty_result

    b_values_arr = np.array(b_values)
    
    return {
        "B_median": float(np.median(b_values_arr)),
        "B_mean": float(np.mean(b_values_arr)),
        "B_std": float(np.std(b_values_arr, ddof=1)) if b_values_arr.size >= 2 else None,
        "B_p16": float(np.percentile(b_values_arr, 16)) if b_values_arr.size >= 1 else None,
        "B_p84": float(np.percentile(b_values_arr, 84)) if b_values_arr.size >= 1 else None,
        "N_success": int(b_values_arr.size)
    }


# =============================================================================
# CALIBRATION AND DATA PREPARATION
# =============================================================================

def cross_calibrate_wavelength(header: fits.Header, 
                               intensity_data_2d: np.ndarray, 
                               wavelengths: np.ndarray, 
                               lab_lines: Tuple[float, float] = (LINE_LAB_1, LINE_LAB_2), 
                               use_atlas: bool = False, 
                               atlas_wav: np.ndarray = None, 
                               atlas_intensity: np.ndarray = None) -> Tuple[np.ndarray, float, float, Dict]:
    """
    Performs wavelength calibration using reference atlas matching and/or known spectral lines.
    """
    # 1. Atlas Shift (Optional)
    current_wavelengths = wavelengths.copy()
    info = {"method": "none", "shift_pix": 0, "a": 1.0, "b": 0.0, 
            "used_atlas": False, "used_lines": False}

    if use_atlas and (atlas_wav is not None) and (atlas_intensity is not None):
        try:
            # Interpolate atlas to current wavelength grid
            ref_i_interp = np.interp(wavelengths, atlas_wav, atlas_intensity, left=np.nan, right=np.nan)
            
            # Create a "quiet sun" profile from the data (top 1/6th of rows assumed quiet)
            quiet_sun_profile = np.median(intensity_data_2d[:max(1, intensity_data_2d.shape[0] // 6), :], axis=0)
            
            shift_pix = compute_cross_correlation_shift(quiet_sun_profile, ref_i_interp)
            delta_lambda_shift = shift_pix * (wavelengths[1] - wavelengths[0])
            
            current_wavelengths = current_wavelengths + delta_lambda_shift
            info.update({"method": "atlas_shift", "shift_pix": int(shift_pix), "used_atlas": True})
        except Exception as e:
            print(f"[CALIB] Atlas shift failed: {e}")

    # 2. Linear Scaling using Spectral Lines
    observed_positions = []
    lab_positions = []
    
    # Use median profile of the entire slit
    median_intensity = np.median(intensity_data_2d, axis=0)
    
    for line_lambda in lab_lines:
        idx = robust_find_line_center(current_wavelengths, median_intensity, line_lambda, window_angstrom=0.6)
        if idx is not None:
            observed_positions.append(current_wavelengths[idx])
            lab_positions.append(line_lambda)

    if len(observed_positions) >= 2:
        scale_a, offset_b = solve_linear_wavelength_scale(observed_positions, lab_positions)
        calibrated_wavelengths = scale_a * current_wavelengths + offset_b
        
        info.update({
            "method": "lines_scale", 
            "used_lines": True, 
            "a": float(scale_a), 
            "b": float(offset_b), 
            "obs_positions": observed_positions, 
            "lab_positions": lab_positions
        })
        return calibrated_wavelengths, float(scale_a), float(offset_b), info

    return current_wavelengths, 1.0, 0.0, info


def prepare_me_input_file(wavelengths: np.ndarray, 
                          i: np.ndarray, q: np.ndarray, u: np.ndarray, v: np.ndarray, 
                          output_dir: str, slit_index: int) -> str:
    """Writes Stokes profiles to a text file for external ME code."""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"slit_{slit_index:04d}_stokes.txt")
    data_matrix = np.vstack([wavelengths, i, q, u, v]).T
    header_str = "wav I Q U V"
    np.savetxt(filename, data_matrix, header=header_str)
    return filename


def slit_index_to_arcsec(header: fits.Header, slit_idx: int) -> Tuple[Optional[float], Optional[float]]:
    """Converts slit index to arcseconds using WCS header info."""
    xcen = header.get('XCEN', None)
    crpix2 = header.get('CRPIX2', header.get('CRPIX', None))
    xscale = header.get('XSCALE', header.get('CDELT2', header.get('CDELT', None)))
    
    if xcen is None or crpix2 is None or xscale is None:
        return None, None
        
    x_arcsec = float(xcen) + (float(slit_idx) - float(crpix2)) * float(xscale)
    ycen = header.get('YCEN', None)
    y_arcsec = float(ycen) if (ycen is not None) else None
    
    return x_arcsec, y_arcsec


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def analyze_fits_file(fits_path: str, 
                      output_prefix: str = DEFAULT_OUTPUT_PREFIX, 
                      n_mc_iterations: int = DEFAULT_MC_ITERATIONS, 
                      run_me_prep: bool = False, 
                      me_cmd_template: Optional[str] = None, 
                      use_atlas: bool = USE_ATLAS_REF, 
                      atlas_wav: Optional[np.ndarray] = None, 
                      atlas_intensity: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Main driver function: loads FITS, calibrates, iterates over slits, and computes B.
    """
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"FITS file not found: {fits_path}")

    # --- Load Data ---
    with fits.open(fits_path, memmap=True) as hdul:
        header = hdul[0].header
        raw_data = hdul[0].data
        
        # Handle cases where data might be in a different HDU
        if raw_data is None:
            for hdu in hdul:
                if hasattr(hdu, 'data') and hdu.data is not None:
                    raw_data = hdu.data
                    header = hdu.header
                    break
        
        if raw_data is None:
            raise RuntimeError("No array data found in FITS HDUs.")
        
        data_cube = raw_data.astype(float)

    # --- Validate Dimensions ---
    if data_cube.ndim != 3:
        raise RuntimeError(f"FITS data must be 3D (4, Nslit, Nlambda). Got: {data_cube.shape}")

    # Ensure shape is (4, Nslit, Nlambda)
    if data_cube.shape[0] != 4:
        if data_cube.shape[2] == 4:
            data_cube = np.transpose(data_cube, (2, 0, 1))
        elif data_cube.shape[1] == 4:
            data_cube = np.transpose(data_cube, (1, 0, 2))
        else:
            raise RuntimeError(f"Unexpected FITS shape. Expected first axis=4 (Stokes). Got: {data_cube.shape}")

    _, n_slits, n_lambda = data_cube.shape
    
    # --- Wavelength Construction & Calibration ---
    raw_wavelengths, cdelt, _, _ = build_wavelength_axis(header, n_lambda)
    print(f"[INFO] File {fits_path} shape: {data_cube.shape}")
    print(f"[INFO] Wav[0]={raw_wavelengths[0]:.6f}, Wav[-1]={raw_wavelengths[-1]:.6f}, dL={cdelt:.6f}")

    intensity_data = data_cube[0, :, :]
    calibrated_wavelengths, _, _, calib_info = cross_calibrate_wavelength(
        header, intensity_data, raw_wavelengths, 
        lab_lines=(LINE_LAB_1, LINE_LAB_2), 
        use_atlas=use_atlas, atlas_wav=atlas_wav, atlas_intensity=atlas_intensity
    )
    print(f"[CALIB] Info: {calib_info}")

    # --- Prepare Outputs ---
    results_list = []
    examples_dir = f"{output_prefix}_examples"
    os.makedirs(examples_dir, exist_ok=True)
    
    b_profile_array = np.full(n_slits, np.nan)
    sigma_b_profile_array = np.full(n_slits, np.nan)

    # --- Main Processing Loop ---
    for s_idx in range(n_slits):
        # Spatial Averaging
        s_lo = max(0, s_idx - (SPATIAL_AVERAGING_WINDOW // 2))
        s_hi = min(n_slits, s_lo + SPATIAL_AVERAGING_WINDOW)
        
        if SPATIAL_AVERAGING_WINDOW > 1:
            stokes_i = data_cube[0, s_lo:s_hi, :].mean(axis=0)
            stokes_q = data_cube[1, s_lo:s_hi, :].mean(axis=0)
            stokes_u = data_cube[2, s_lo:s_hi, :].mean(axis=0)
            stokes_v = data_cube[3, s_lo:s_hi, :].mean(axis=0)
        else:
            stokes_i = data_cube[0, s_idx, :]
            stokes_q = data_cube[1, s_idx, :]
            stokes_u = data_cube[2, s_idx, :]
            stokes_v = data_cube[3, s_idx, :]

        # Determine center pixel for search
        idx_center = int(np.argmin(np.abs(calibrated_wavelengths - REFERENCE_WAVELENGTH_0)))
        if len(calibrated_wavelengths) == 0 or idx_center < 0 or idx_center >= n_lambda:
            idx_center = n_lambda // 2

        # --- Method 1: Sigma-V Analysis ---
        sigma_res = analyze_sigma_v_on_spectrum(calibrated_wavelengths, stokes_i, stokes_v, idx_center)
        
        used_method = "none"
        b_final_val = None
        wfa_correlation = None

        if sigma_res.get("found") and sigma_res.get("B_G") is not None:
            used_method = "sigma"
            b_final_val = sigma_res["B_G"]
        else:
            # --- Method 2: Fallback to Weak Field Approximation (WFA) ---
            try:
                # Calculate derivative of I
                smooth_win = ensure_odd(min(SMOOTHING_WINDOW_SIZE, max(3, n_lambda - 1)))
                i_smooth = savgol_filter(stokes_i, smooth_win, SMOOTHING_POLY_ORDER) if n_lambda >= smooth_win else stokes_i
                
                # Gradient: dI/dlambda
                di_dlambda = np.gradient(i_smooth, calibrated_wavelengths)
                
                # WFA Equation: V = -C * B * (dI/dlambda)  =>  V = X * B, where X = -C * (dI/dlambda)
                const_c = ZEEMAN_CONSTANT_K * (REFERENCE_WAVELENGTH_0**2) * LANDÉ_FACTOR_EFF
                regressor_x = -const_c * di_dlambda
                
                # Segment for fitting
                seg_lo = max(0, idx_center - SIGMA_TIGHT_WINDOW_PIX)
                seg_hi = min(n_lambda, idx_center + SIGMA_TIGHT_WINDOW_PIX)
                
                x_seg = regressor_x[seg_lo:seg_hi]
                v_seg = stokes_v[seg_lo:seg_hi]
                
                valid_mask = np.isfinite(x_seg) & np.isfinite(v_seg)
                
                if valid_mask.sum() >= 5:
                    # Linear regression
                    design_mat = np.vstack([x_seg[valid_mask], np.ones(valid_mask.sum())]).T
                    solution, _, _, _ = np.linalg.lstsq(design_mat, v_seg[valid_mask], rcond=None)
                    
                    slope_b = solution[0]
                    model_v = design_mat.dot(solution)
                    
                    # Quality checks
                    residuals = v_seg[valid_mask] - model_v
                    resid_noise = calculate_mad_std(residuals)
                    seg_snr = np.nanmax(np.abs(v_seg[valid_mask])) / (resid_noise if resid_noise > 0 else 1e-12)
                    
                    corr_coeff = np.corrcoef(v_seg[valid_mask], model_v)[0, 1] if valid_mask.sum() >= 3 else 0.0
                    
                    if np.isfinite(slope_b) and abs(corr_coeff) >= 0.4 and seg_snr >= 5.0:
                        used_method = "wfa"
                        b_final_val = float(slope_b)
                        wfa_correlation = float(corr_coeff)
                    else:
                        used_method = "none"
                else:
                    used_method = "none"
            except Exception:
                used_method = "none"

        # --- Monte Carlo Error Analysis ---
        mc_stats = {
            "B_median": None, "B_mean": None, "B_std": None, 
            "B_p16": None, "B_p84": None, "N_success": 0
        }
        
        if b_final_val is not None:
            # Estimate noise for MC injection
            noise_est_window_lo = max(0, idx_center - SIGMA_TIGHT_WINDOW_PIX)
            noise_est_window_hi = min(n_lambda, idx_center + SIGMA_TIGHT_WINDOW_PIX)
            
            mask_mc = np.ones_like(stokes_v, dtype=bool)
            mask_mc[noise_est_window_lo:noise_est_window_hi] = False
            
            mc_noise_samples = stokes_v[mask_mc] if np.any(mask_mc) else stokes_v
            mc_noise_est = calculate_mad_std(mc_noise_samples)
            
            n_mc_actual = max(50, min(n_mc_iterations, 500))
            
            # Run MC using the sigma-V function as the estimator
            mc_stats = estimate_b_error_mc(
                calibrated_wavelengths, stokes_i, stokes_v, idx_center, 
                analyze_sigma_v_on_spectrum, 
                n_iterations=n_mc_actual, 
                noise_estimate=mc_noise_est
            )

        # --- Record Results ---
        x_arc, y_arc = slit_index_to_arcsec(header, s_idx)
        
        row_data = {
            "slit": int(s_idx),
            "x_pix": int(s_idx),
            "x_arcsec": float(x_arc) if x_arc is not None else None,
            "y_arcsec": float(y_arc) if y_arc is not None else None,
            "idx_center": int(idx_center),
            "used": used_method,
            
            "B_sigma_G": float(sigma_res["B_G"]) if (sigma_res.get("found") and sigma_res.get("B_G") is not None) else None,
            "B_wfa_G": float(b_final_val) if (used_method in ("wfa", "sigma") and b_final_val is not None) else None,
            "wfa_r": float(wfa_correlation) if wfa_correlation is not None else None,
            
            "V_rel": float(sigma_res.get("V_rel")) if sigma_res.get("V_rel") is not None else None,
            "noise": float(sigma_res.get("noise")) if sigma_res.get("noise") is not None else None,
            "SNR": float(sigma_res.get("SNR")) if sigma_res.get("SNR") is not None else None,
            "sigma_found": bool(sigma_res.get("found", False)),
            
            "wa_A": float(sigma_res.get("wa")) if sigma_res.get("wa") is not None else None,
            "wb_A": float(sigma_res.get("wb")) if sigma_res.get("wb") is not None else None,
            "delta_lambda_A": float(sigma_res.get("delta_lambda")) if sigma_res.get("delta_lambda") is not None else None,
            
            "suspect": bool(sigma_res.get("suspect", False)),
            "suspect_reason": sigma_res.get("suspect_reason"),
            
            "B_MC_median": mc_stats["B_median"],
            "B_MC_mean": mc_stats["B_mean"],
            "B_MC_std": mc_stats["B_std"],
            "B_p16": mc_stats["B_p16"],
            "B_p84": mc_stats["B_p84"],
            "B_MC_n_success": mc_stats["N_success"]
        }

        # Resolve final B choice
        b_choice = row_data["B_sigma_G"] if row_data["B_sigma_G"] is not None else row_data["B_wfa_G"]
        row_data["B_G"] = float(b_choice) if b_choice is not None else None
        
        results_list.append(row_data)
        
        b_profile_array[s_idx] = row_data["B_G"] if row_data["B_G"] is not None else np.nan
        sigma_b_profile_array[s_idx] = float(mc_stats["B_std"]) if mc_stats["B_std"] is not None else np.nan

        # --- Diagnostic Plot for Suspect/Found Cases ---
        if row_data["sigma_found"] or row_data["suspect"]:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
            
            # Plot Intensity
            ax1.plot(calibrated_wavelengths, stokes_i, label="I")
            ax1.set_ylabel("Intensity")
            ax1.grid(True)
            
            # Plot Stokes V (Raw vs Smoothed)
            win_len = ensure_odd(min(SMOOTHING_WINDOW_SIZE, max(3, n_lambda - 1)))
            v_smooth_plot = savgol_filter(stokes_v, win_len, SMOOTHING_POLY_ORDER) if n_lambda >= win_len else stokes_v
            
            ax2.plot(calibrated_wavelengths, stokes_v, label="V_raw", alpha=0.6)
            ax2.plot(calibrated_wavelengths, v_smooth_plot, label="V_smooth", color='red', lw=1)
            
            # Mark sigma components
            if row_data["wa_A"] is not None:
                ax2.axvline(row_data["wa_A"], color='magenta', linestyle='--', label='sigma -')
            if row_data["wb_A"] is not None:
                ax2.axvline(row_data["wb_A"], color='cyan', linestyle='--', label='sigma +')
                
            ax2.set_ylabel("Stokes V")
            ax2.set_xlabel("Wavelength (Å)")
            ax2.grid(True)
            ax2.legend()
            
            plot_title = (f"Slit={s_idx} X={row_data['x_arcsec']} "
                          f"B={row_data['B_G']} Method={row_data['used']} "
                          f"Suspect={row_data['suspect']}")
            fig.suptitle(plot_title)
            
            output_png = os.path.join(examples_dir, f"slit_{s_idx:04d}.png")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            try:
                plt.savefig(output_png, dpi=150)
            except Exception:
                pass
            plt.close(fig)

        if (s_idx % 50) == 0:
            print(f"[PROGRESS] Processed slit {s_idx}/{n_slits}")

    # --- Save Results ---
    df_results = pd.DataFrame(results_list)
    csv_filename = f"{output_prefix}_results.csv"
    df_results.to_csv(csv_filename, index=False, float_format="%.6f")
    
    np.save(f"{output_prefix}_B_profile.npy", b_profile_array)

    # --- Generate Summary Plots ---
    # 1. B Profile
    plt.figure(figsize=(10, 4))
    x_axis_indices = np.arange(len(b_profile_array))
    plt.plot(x_axis_indices, b_profile_array, marker='o', lw=0.8)
    plt.grid(True)
    plt.xlabel("Slit Index")
    plt.ylabel("B (Gauss)")
    plt.title("Magnetic Field Profile")
    plt.savefig(f"{output_prefix}_B_profile.png", dpi=200)
    plt.close()

    # 2. Uncertainty Profile
    plt.figure(figsize=(10, 4))
    plt.plot(x_axis_indices, sigma_b_profile_array, marker='o', lw=0.8)
    plt.grid(True)
    plt.xlabel("Slit Index")
    plt.ylabel("Sigma B (Gauss)")
    plt.title("Magnetic Field Uncertainty (Monte Carlo)")
    plt.savefig(f"{output_prefix}_B_sigma_profile.png", dpi=200)
    plt.close()

    # 3. Scatter Map (Approximate)
    try:
        coords = df_results['x_arcsec'].values
        if np.all(~pd.isna(coords)):
            plt.figure(figsize=(8, 4))
            sc = plt.scatter(coords, np.zeros_like(coords), c=df_results['B_G'].values, cmap='inferno', s=15)
            plt.colorbar(sc, label='B (Gauss)')
            plt.xlabel('X (arcsec)')
            plt.title('B along scan (approximate)')
            plt.savefig(f"{output_prefix}_B_map_scan.png", dpi=200)
            plt.close()
    except Exception:
        pass
    
    print(f"[INFO] Saved results CSV: {csv_filename}")

    # --- 2D Map Construction ---
    try:
        print("[PLOT] Building 2D B-map...")
        # Note: Original code logic assumes 'data' is (4, Nslit, Nlambda) and Npix=1
        # It builds a map of shape (Nslit, 1). This logic is preserved.
        b_map = np.full((n_slits, 1), np.nan)

        for s_idx in range(n_slits):
            i_prof = data_cube[0, s_idx, :]
            v_prof = data_cube[3, s_idx, :]
            idx_cen = int(np.argmin(np.abs(calibrated_wavelengths - REFERENCE_WAVELENGTH_0)))
            
            res_2d = analyze_sigma_v_on_spectrum(calibrated_wavelengths, i_prof, v_prof, idx_cen)
            if res_2d.get("found") and res_2d.get("B_G") is not None:
                b_map[s_idx, 0] = res_2d["B_G"]

        plt.figure(figsize=(8, 6))
        im = plt.imshow(b_map.T, aspect='auto', cmap='RdBu_r',
                        extent=[0, n_slits, 0, 1], origin='lower')
        plt.colorbar(im, label="B (Gauss)")
        plt.xlabel("Slit Index")
        plt.ylabel("Pixel along slit (Aggregated)")
        plt.title("2D Magnetic Field Map (Sigma-V)")
        plt.savefig(f"{output_prefix}_B_map_2D.png", dpi=200)
        plt.close()
        print(f"[PLOT] Saved 2D map -> {output_prefix}_B_map_2D.png")
    except Exception as e:
        print(f"[WARN] Failed to build 2D B-map: {e}")

    print(f"[INFO] Saved diagnostic PNGs in: {examples_dir}")

    # --- ME Solver Preparation (Optional) ---
    if run_me_prep and (me_cmd_template is not None):
        me_input_dir = f"{output_prefix}_ME_input"
        os.makedirs(me_input_dir, exist_ok=True)
        
        # Prepare inputs for top 20 strongest fields
        top_slits = df_results[df_results['B_G'].notna()].sort_values('B_G', ascending=False)['slit'].tolist()[:20]
        
        for s in top_slits:
            s_idx = int(s)
            s_lo = max(0, s_idx - (SPATIAL_AVERAGING_WINDOW // 2))
            s_hi = min(n_slits, s_lo + SPATIAL_AVERAGING_WINDOW)
            
            if SPATIAL_AVERAGING_WINDOW > 1:
                i_s = data_cube[0, s_lo:s_hi, :].mean(axis=0)
                q_s = data_cube[1, s_lo:s_hi, :].mean(axis=0)
                u_s = data_cube[2, s_lo:s_hi, :].mean(axis=0)
                v_s = data_cube[3, s_lo:s_hi, :].mean(axis=0)
            else:
                i_s = data_cube[0, s_idx, :]
                q_s = data_cube[1, s_idx, :]
                u_s = data_cube[2, s_idx, :]
                v_s = data_cube[3, s_idx, :]
                
            infile = prepare_me_input_file(calibrated_wavelengths, i_s, q_s, u_s, v_s, me_input_dir, s_idx)
            print(f"[ME] Prepared input: {infile}")
        
        print("[ME] To run external ME solver set ME_SOLVER_CMD_TEMPLATE.")

    return df_results


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main_cli():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(description="Hinode SP Sigma-V Analyzer v3 (Refactored)")
    parser.add_argument("--fits", help="Path to input FITS file", default=DEFAULT_FITS_FILE)
    parser.add_argument("--out", help="Output filename prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--mc", type=int, help="Number of Monte Carlo iterations", default=DEFAULT_MC_ITERATIONS)
    parser.add_argument("--atlas", action='store_true', help="Enable Atlas-based calibration")
    parser.add_argument("--run_me", action='store_true', help="Prepare Milne-Eddington inputs")
    
    args = parser.parse_args()

    output_prefix = args.out
    n_mc_iters = int(args.mc)
    use_atlas_flag = bool(args.atlas)
    run_me_flag = bool(args.run_me)

    print("=== Hinode SP Sigma-V Analyzer ===")
    print(f"Input FITS: {args.fits}")
    print(f"Output Prefix: {output_prefix}")
    
    start_time = time.time()
    
    atlas_wavelengths = None
    atlas_intensities = None
    
    if use_atlas_flag and (REF_ATLAS_WAV_PATH is not None):
        try:
            atlas_data = np.loadtxt(REF_ATLAS_WAV_PATH)
            atlas_wavelengths = atlas_data[:, 0]
            atlas_intensities = atlas_data[:, 1]
            print(f"[ATLAS] Loaded atlas from {REF_ATLAS_WAV_PATH}")
        except Exception as e:
            print(f"[ATLAS] Failed to load atlas: {e}")
            atlas_wavelengths = None
            atlas_intensities = None

    try:
        df_result = analyze_fits_file(
            args.fits, 
            output_prefix=output_prefix, 
            n_mc_iterations=n_mc_iters, 
            run_me_prep=run_me_flag, 
            me_cmd_template=ME_SOLVER_CMD_TEMPLATE, 
            use_atlas=use_atlas_flag, 
            atlas_wav=atlas_wavelengths, 
            atlas_intensity=atlas_intensities
        )
        
        elapsed = time.time() - start_time
        print(f"Analysis complete in {elapsed:.1f} seconds.")
        
        if 'used' in df_result.columns:
            print("\nMethod Usage Statistics:")
            print(df_result['used'].value_counts(dropna=False))
            
        if 'suspect' in df_result.columns:
            print(f"Suspect measurements: {int(df_result['suspect'].sum())}")
            
    except Exception as error:
        print(f"[ERROR] Analysis failed: {error}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_cli()
