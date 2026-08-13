# parse_nexafs_v2

NEXAFS scan parsing, energy calibration, and double-normalisation.

Implements the double-normalisation method described in Watts, Thomsen &
Dastoor (2006), "Methods in carbon K-edge NEXAFS: Experiment and analysis",
*J. Electron Spectrosc. Relat. Phenom.* 151, 105-120.

Compared to an earlier `parse_nexafs.py`, this version replaces hardcoded
indices and column-count-dependent branching with explicit, overridable
configuration objects: the energy-calibration reference feature and search
window, the pre-edge background-fit window, and the post-edge
step-normalisation window are all specified in eV rather than as raw row
indices, so they can be adjusted per sample without editing the code.


## Public API

- `parse_nexafs(file, *, columns=DEFAULT_COLUMNS, header_rows=DEFAULT_HEADER_ROWS, leading_rows_to_drop=DEFAULT_LEADING_ROWS_TO_DROP)`
  Parse a raw NEXAFS `.asc` scan into a labelled `polars.DataFrame`, dropping
  the leading rows where detectors are still settling.

- `energy_correct(df, calibration=EnergyCalibration())`
  Shift a scan's energy axis so its calibration peak sits at the target
  energy. Locates the peak via a Savitzky-Golay-smoothed, sub-grid quadratic
  vertex fit rather than the single loudest raw sample, and stores the
  result in a new `"Corrected Energy"` column.

- `double_normalise(sample_file, photodiode_file, *, sample_calibration=EnergyCalibration(), photodiode_calibration=EnergyCalibration(), background=BackgroundFit(), step_normalisation=StepNormalisation())`
  Double-normalise a sample scan against a photodiode reference scan: the
  sample's Channeltron signal is normalised to its own I0, divided by the
  equivalently I0-normalised photodiode reference to remove the beamline
  transmission function, a linear pre-edge background is subtracted, and the
  result is scaled to unit edge-jump height. Uncertainty is propagated
  end-to-end from Poisson counting statistics on every raw channel through
  each normalisation step via the `uncertainties` package.

### Configuration dataclasses

- `EnergyCalibration(reference_column, search_window_eV, calibrated_energy_eV)`
  Where to find a scan's calibration feature and where it should sit
  (defaults to the graphite 1s -> pi* peak at 285 eV).

- `BackgroundFit(fit_window_eV, apply)`
  Pre-edge energy range used to fit a linear background, and whether to
  subtract it.

- `StepNormalisation(window_eV, reducer)`
  Post-edge energy range expected to be flat, and the function (default
  `np.mean`) reducing it to a single step-height scalar.

## Usage

```python
from diamond_desorption_experiment.parse_nexafs_v2 import double_normalise

result = double_normalise("sample.asc", "photodiode.asc")
# result columns include "Corrected Energy", "Step Scaled",
# and "Step Scaled Error" (the propagated 1-sigma uncertainty)
```

Pass overridden configuration objects to adjust calibration or normalisation
windows per sample, e.g.:

```python
from diamond_desorption_experiment.parse_nexafs_v2 import (
    BackgroundFit,
    StepNormalisation,
    double_normalise,
)

result = double_normalise(
    "sample.asc",
    "photodiode.asc",
    background=BackgroundFit(fit_window_eV=(276.0, 280.0)),
    step_normalisation=StepNormalisation(window_eV=(310.0, 320.0)),
)
```
