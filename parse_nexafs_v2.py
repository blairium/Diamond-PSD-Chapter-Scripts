"""NEXAFS scan parsing and double-normalisation.

Implements the double-normalisation method described in:
Watts, Thomsen & Dastoor (2006), "Methods in carbon K-edge NEXAFS:
Experiment and analysis", J. Electron Spectrosc. Relat. Phenom. 151, 105-120.

Compared to ``parse_nexafs.py``, this version replaces hardcoded indices
and column-count-dependent branching with explicit, overridable
configuration objects: the energy-calibration reference feature and search
window, the pre-edge background-fit window, and the post-edge
step-normalisation window are all specified in eV rather than as raw row
indices, so they can be adjusted per sample without editing the code.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from scipy.signal import savgol_filter
from uncertainties import ufloat, unumpy

DEFAULT_COLUMNS: tuple[str, ...] = (
    "index",
    "Energy Setpoint [eV]",
    "unchanging value, maybe energy steps",
    "Monochromator Energy",
    "Drain Current VF [eV]",
    "I zero",
    "Reference Foil VF",
    "MCP?",
    "Channeltron",
    "Direct_PHD_VF",  # photodiode
    "TFY_PHD_VF",
    "Drain Current (Keithley1)",
    "I zero (Keithley 3)",
    "Reference Foil (Keithley4)",
    "Keithly6",
    "Beam Current [mA]",
    "BL_PHD_VF",
    "BL_PHD_Keithley2",
    "Gap Request, mm",
    "20  [1-D Detector  17]  SR14ID01:GAP_MONITOR, , mm",
    "  21  [1-D Detector  18]  SR14ID01IOC68:IP330_1.VAL, , ",
)

DEFAULT_HEADER_ROWS = 150
# Detectors take a few points to settle at the start of a scan (seen as a
# near-zero transient on the Channeltron channel); drop them before use. 5 was
# not always enough - e.g. sxr132454's row 5 (Channeltron 1587) still sits well
# below the ~2400-2600 plateau every neighbouring row settles to by row 6, and
# since the default pre-edge background-fit window starts at each scan's own
# first surviving point, that one still-settling point got outsized leverage
# on the fitted slope (see double_normalise's BackgroundFit).
DEFAULT_LEADING_ROWS_TO_DROP = 6

ENERGY_COLUMN = "Monochromator Energy"  # "Energy Setpoint [eV]"
CORRECTED_ENERGY_COLUMN = "Corrected Energy"


def parse_nexafs(
    file: str | Path,
    *,
    columns: Sequence[str] = DEFAULT_COLUMNS,
    header_rows: int = DEFAULT_HEADER_ROWS,
    leading_rows_to_drop: int = DEFAULT_LEADING_ROWS_TO_DROP,
) -> pl.DataFrame:
    """Parse a raw NEXAFS ``.asc`` scan into a labelled DataFrame.

    Args:
        file: Path to the ``.asc`` file.
        columns: Column names to assign, in the order they appear in the file.
        header_rows: Number of metadata header lines preceding the data table.
        leading_rows_to_drop: Number of data rows to discard from the start
            of the scan.

    Returns:
        Parsed scan data, with the leading transient rows removed.
    """
    return pl.read_csv(file, separator="\t", new_columns=list(columns), skip_rows=header_rows)[
        leading_rows_to_drop:
    ]


@dataclass(frozen=True)
class EnergyCalibration:
    """Where to find a scan's calibration feature, and where it should sit.

    Attributes:
        reference_column: Column whose peak marks the calibration feature.
        search_window_eV: ``(low, high)`` energy range to search for the peak.
        calibrated_energy_eV: True energy the peak should be shifted to
            (defaults to the graphite 1s -> pi* peak at 285 eV).
    """

    reference_column: str = "Reference Foil VF"
    search_window_eV: tuple[float, float] = (283.0, 287.0)
    calibrated_energy_eV: float = 285.0


def _poisson_uarray(counts: np.ndarray) -> np.ndarray:
    """Raw counts as an ``uncertainties`` array with Poisson (sqrt(N)) uncertainty.

    Every raw channel here (Channeltron, I0, the photodiode's own counts and
    I0) is a photon/electron count subject to shot noise, so its 1-sigma
    uncertainty is ``sqrt(N)``.
    """
    counts = np.asarray(counts, dtype=float)
    return unumpy.uarray(counts, np.sqrt(np.abs(counts)))


def _interp_uarray(x: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """Linear interpolation of an ``uncertainties`` array onto new points.

    ``np.interp`` only accepts plain floats. The interpolation weight between
    two bracketing points is an ordinary (non-uncertain) number - it depends
    only on the energy grids, not on the uncertain data - so a manual
    weighted sum lets each ``ufloat``'s own arithmetic propagate the two
    bracketing points' uncertainty correctly.

    Args:
        x: Points to interpolate onto.
        xp: The x-coordinates *yp* is sampled at (must be increasing).
        yp: ``uncertainties`` array of values at *xp*.
    """
    idx = np.clip(np.searchsorted(xp, x) - 1, 0, len(xp) - 2)
    x0, x1 = xp[idx], xp[idx + 1]
    y0, y1 = yp[idx], yp[idx + 1]
    weight = (x - x0) / (x1 - x0)
    return y0 + weight * (y1 - y0)


def _select_energy_window(
    df: pl.DataFrame, energy_column: str, window_eV: tuple[float, float]
) -> pl.DataFrame:
    low, high = window_eV
    selected = df.filter(pl.col(energy_column).is_between(low, high))
    if selected.is_empty():
        raise ValueError(f"No data points fall within {window_eV} eV of column {energy_column!r}.")
    return selected


# Savitzky-Golay smoothing applied before locating the calibration peak (see
# _quadratic_peak_energy) - a quadratic (polyorder 2) local fit over a 5-point
# window, matching the quadratic vertex fit that follows it. window_length is
# clamped down (to the largest smaller odd value) for search windows with
# fewer points than this.
CALIBRATION_SMOOTHING_WINDOW = 5
CALIBRATION_SMOOTHING_POLYORDER = 2


def _quadratic_peak_energy(energy: np.ndarray, values: np.ndarray) -> float:
    """Sub-grid energy of *values*' peak within *energy*, via a 3-point quadratic fit.

    A scan's raw calibration-feature signal is discrete and noisy, so the
    single loudest sample isn't necessarily the true peak location - on a
    ~0.1 eV grid, picking whichever grid point happens to read highest can
    jitter the recovered peak by that much between otherwise-identical scans
    (see :func:`energy_correct`). A Savitzky-Golay filter first reduces
    sensitivity to any one noisy sample (a local polynomial fit over a
    sliding window, rather than a plain moving average, so it doesn't flatten
    the peak itself the way averaging would), then a quadratic fit through
    the (smoothed) peak and its two neighbours gives a sub-grid vertex
    estimate instead of snapping to the nearest raw sample.

    Falls back to the (smoothed) argmax itself if the peak sits at either
    edge of *values* (no two-sided neighbourhood to fit), the fit is
    degenerate (e.g. three collinear points - no parabola, so no vertex), or
    the fit isn't a maximum, or its vertex falls outside the 3-point
    neighbourhood it was fit to (an ill-conditioned fit can otherwise place
    the vertex arbitrarily far away).
    """
    window_length = min(CALIBRATION_SMOOTHING_WINDOW, len(values))
    if window_length % 2 == 0:
        window_length -= 1
    if window_length > CALIBRATION_SMOOTHING_POLYORDER:
        smoothed = savgol_filter(
            values.astype(float),
            window_length=window_length,
            polyorder=CALIBRATION_SMOOTHING_POLYORDER,
        )
    else:
        smoothed = values.astype(float).copy()

    idx = int(np.argmax(smoothed))
    if idx == 0 or idx == len(smoothed) - 1:
        return float(energy[idx])

    x0, x1, x2 = energy[idx - 1], energy[idx], energy[idx + 1]
    y0, y1, y2 = smoothed[idx - 1], smoothed[idx], smoothed[idx + 1]
    design = np.array([[x0**2, x0, 1.0], [x1**2, x1, 1.0], [x2**2, x2, 1.0]])
    try:
        a, b, _c = np.linalg.solve(design, [y0, y1, y2])
    except np.linalg.LinAlgError:
        return float(x1)
    if a >= 0:
        return float(x1)
    vertex = -b / (2 * a)
    if not (x0 <= vertex <= x2):
        return float(x1)
    return float(vertex)


def energy_correct(
    df: pl.DataFrame, calibration: EnergyCalibration = EnergyCalibration()
) -> pl.DataFrame:
    """Shift a scan's energy axis so its calibration peak sits at the target energy.

    Locates the peak of ``calibration.reference_column`` within
    ``calibration.search_window_eV`` (see :func:`_quadratic_peak_energy` for
    how - a smoothed, sub-grid estimate rather than the single loudest raw
    sample, since otherwise-identical scans were found to calibrate up to
    ~0.1 eV apart - a full grid step - purely from which sample happened to
    read highest) and shifts ``Energy Setpoint [eV]`` by a constant offset so
    that peak lands at ``calibration.calibrated_energy_eV``, storing the
    result in a new ``"Corrected Energy"`` column.

    Args:
        df: DataFrame from :func:`parse_nexafs`.
        calibration: Calibration feature, search window and target energy.

    Returns:
        ``df`` with an added ``"Corrected Energy"`` column.
    """
    window = _select_energy_window(df, ENERGY_COLUMN, calibration.search_window_eV)
    peak_energy = _quadratic_peak_energy(
        window[ENERGY_COLUMN].to_numpy(), window[calibration.reference_column].to_numpy()
    )
    shift = calibration.calibrated_energy_eV - peak_energy
    return df.with_columns((pl.col(ENERGY_COLUMN) + shift).alias(CORRECTED_ENERGY_COLUMN))


@dataclass(frozen=True)
class BackgroundFit:
    """Configuration for the linear pre-edge background subtraction.

    Attributes:
        fit_window_eV: ``(low, high)`` pre-edge energy range, on the
            calibrated energy axis, used to fit a linear background. Defaults
            to ``(275.5, 279.5)`` - a 4 eV window starting at the historical
            275.5 eV pre-edge energy, wide enough that the fitted slope isn't
            dominated by point-to-point noise (a narrower ~1.5 eV window was
            found to undershoot/overshoot the true pre-edge continuum by the
            time it's extrapolated out to the nominated peaks - see the
            comparison against QANT-normalised reference spectra in the
            :func:`double_normalise` diagnosis this default came from).
        apply: If ``False``, skip subtracting the fitted background (it is
            still computed and stored).
    """

    fit_window_eV: tuple[float, float] = (275.6, 280)
    apply: bool = True


@dataclass(frozen=True)
class StepNormalisation:
    """Configuration for scaling a spectrum to unit edge-jump height.

    Attributes:
        window_eV: ``(low, high)`` post-edge energy range, on the calibrated
            energy axis, expected to be flat and used to measure the step height.
        reducer: Function reducing the background-subtracted values in
            ``window_eV`` to a single step-height scalar (e.g. ``np.mean``,
            ``np.max``).
    """

    window_eV: tuple[float, float] = (313.0, 321.0)
    reducer: Callable[[np.ndarray], float] = np.mean


def double_normalise(
    sample_file: str | Path,
    photodiode_file: str | Path,
    *,
    sample_calibration: EnergyCalibration = EnergyCalibration(),
    photodiode_calibration: EnergyCalibration = EnergyCalibration(),
    background: BackgroundFit = BackgroundFit(),
    step_normalisation: StepNormalisation = StepNormalisation(),
) -> pl.DataFrame:
    """Double-normalise a sample NEXAFS scan against a photodiode reference scan.

    The sample's Channeltron signal is normalised to its own I0 (mesh
    current), then divided by the equivalently I0-normalised photodiode
    reference scan to remove the beamline transmission function. A linear
    background fitted over ``background.fit_window_eV`` is subtracted, and
    the result is scaled so the mean/max (per ``step_normalisation.reducer``)
    over ``step_normalisation.window_eV`` is unity.

    The photodiode reference is interpolated onto the sample's own
    (possibly non-uniformly stepped) calibrated energy axis, so the two
    scans do not need matching lengths or step sizes.

    Uncertainty is propagated end-to-end from Poisson (shot-noise) counting
    statistics on every raw channel (Channeltron, I0, and the photodiode's
    own counts and I0 - see :func:`_poisson_uarray`), through each
    normalisation step, via the ``uncertainties`` package:

    * The I0-normalised ratios (``Sample/I0``, and the photodiode's own
      ratio before it's interpolated onto the sample's energy axis - see
      :func:`_interp_uarray`) combine relative counting error in quadrature.
    * The background fit's own uncertainty comes from its covariance matrix
      (``np.polyfit(..., cov=True)``), which - since the fit is unweighted -
      is scaled by the pre-edge points' residual scatter; this folds in
      noise sources beyond pure counting statistics (e.g. any real point-to-
      point drift) without requiring a separate model for them.
    * The step height's uncertainty treats ``step_normalisation.window_eV``'s
      points as independent measurements of one underlying value - exactly
      the assumption ``reducer`` (default ``np.mean``) already makes by
      combining them into a single number - via standard error-of-the-mean:
      ``sqrt(sum(sigma_i ** 2)) / N``. This is exact when ``reducer`` is
      ``np.mean``; for any other reducer it is an approximation, since only
      the mean has a closed-form uncertainty here.

    Args:
        sample_file: Path to the sample's ``.asc`` scan.
        photodiode_file: Path to the photodiode reference ``.asc`` scan.
        sample_calibration: Energy calibration to apply to the sample scan.
        photodiode_calibration: Energy calibration to apply to the photodiode scan.
        background: Pre-edge linear background fit configuration.
        step_normalisation: Post-edge step-height normalisation configuration.

    Returns:
        The sample DataFrame with added columns: ``"Corrected Energy"``,
        ``"Sample/I0"``, ``"Photodiode/I0"``, ``"Corrected"``,
        ``"Background Fit"``, ``"Background subtracted"``, ``"Step Scaled"``
        and ``"Step Scaled Error"`` (the propagated 1-sigma uncertainty on
        ``"Step Scaled"``).
    """
    data = energy_correct(parse_nexafs(sample_file), sample_calibration)
    photodiode = energy_correct(parse_nexafs(photodiode_file), photodiode_calibration)

    data = data.with_columns((pl.col("Channeltron") / pl.col("I zero")).alias("Sample/I0"))
    sample_ratio_u = _poisson_uarray(data["Channeltron"].to_numpy()) / _poisson_uarray(
        data["I zero"].to_numpy()
    )

    photodiode_i0 = np.interp(
        data[CORRECTED_ENERGY_COLUMN].to_numpy(),
        photodiode[CORRECTED_ENERGY_COLUMN].to_numpy(),
        (photodiode["Direct_PHD_VF"] / photodiode["I zero"]).to_numpy(),
    )
    data = data.with_columns(pl.Series("Photodiode/I0", photodiode_i0))
    photodiode_ratio_u = _poisson_uarray(photodiode["Direct_PHD_VF"].to_numpy()) / _poisson_uarray(
        photodiode["I zero"].to_numpy()
    )
    photodiode_ratio_interp_u = _interp_uarray(
        data[CORRECTED_ENERGY_COLUMN].to_numpy(),
        photodiode[CORRECTED_ENERGY_COLUMN].to_numpy(),
        photodiode_ratio_u,
    )

    data = data.with_columns((pl.col("Sample/I0") / pl.col("Photodiode/I0")).alias("Corrected"))
    corrected_u = sample_ratio_u / photodiode_ratio_interp_u

    energy = data[CORRECTED_ENERGY_COLUMN].to_numpy()

    fit_points = _select_energy_window(data, CORRECTED_ENERGY_COLUMN, background.fit_window_eV)
    fit_coeffs, fit_cov = np.polyfit(
        fit_points[CORRECTED_ENERGY_COLUMN].to_numpy(),
        fit_points["Corrected"].to_numpy(),
        1,
        cov=True,
    )
    background_values = np.poly1d(fit_coeffs)(energy)

    data = data.with_columns(pl.Series("Background Fit", background_values))

    # Uncertainty of the fitted line's value at each energy - the standard
    # OLS prediction-of-mean-response variance, x' Cov(coeffs) x, for
    # coefficient order [slope, intercept] matching np.polyfit's degree-1 fit.
    design = np.stack([energy, np.ones_like(energy)], axis=1)
    background_variance = np.clip(np.einsum("ij,jk,ik->i", design, fit_cov, design), 0, None)
    background_u = unumpy.uarray(background_values, np.sqrt(background_variance))

    data = data.with_columns(
        (
            pl.col("Corrected") - pl.col("Background Fit")
            if background.apply
            else pl.col("Corrected")
        ).alias("Background subtracted")
    )
    background_subtracted_u = corrected_u - background_u if background.apply else corrected_u

    step_points = _select_energy_window(data, CORRECTED_ENERGY_COLUMN, step_normalisation.window_eV)
    step_height = step_normalisation.reducer(step_points["Background subtracted"].to_numpy())
    if step_height == 0:
        raise ValueError(
            f"Step normalisation window {step_normalisation.window_eV} eV has "
            "zero height; choose a different window."
        )

    step_window_lo, step_window_hi = step_normalisation.window_eV
    step_mask = (energy >= step_window_lo) & (energy <= step_window_hi)
    step_sigmas = unumpy.std_devs(background_subtracted_u[step_mask])
    step_height_u = ufloat(step_height, np.sqrt(np.sum(step_sigmas**2)) / len(step_sigmas))

    data = data.with_columns((pl.col("Background subtracted") / step_height).alias("Step Scaled"))
    step_scaled_u = background_subtracted_u / step_height_u
    data = data.with_columns(pl.Series("Step Scaled Error", unumpy.std_devs(step_scaled_u)))
    return data
