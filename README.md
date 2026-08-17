# parse_nexafs

NEXAFS scan parsing, energy calibration, and double-normalisation.

Intended for NEXAFS `.asc` scans from the SXR beamline of the Australian
Synchrotron — the default column layout (`DEFAULT_COLUMNS`), header size
(`DEFAULT_HEADER_ROWS`), and calibration/background/step-normalisation
energy windows below are all tuned to that beamline's output format and are
not guaranteed to suit scans from other instruments.

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

## Overriding the default energy windows

Every energy window used by calibration, background-fitting, and
step-normalisation is a keyword argument on its config dataclass, with the
value below as its default — pass only the fields you want to change:

| Dataclass           | Field                | Default            | Meaning                                                        |
| -------------------- | --------------------- | ------------------- | ---------------------------------------------------------------- |
| `EnergyCalibration` | `reference_column`   | `"Reference Foil VF"` | Column whose peak marks the calibration feature                |
| `EnergyCalibration` | `search_window_eV`   | `(283.0, 287.0)`    | Range searched for that peak                                   |
| `EnergyCalibration` | `calibrated_energy_eV` | `285.0`            | True energy the peak is shifted to (graphite 1s -> pi*)         |
| `BackgroundFit`     | `fit_window_eV`      | `(275.6, 280)`      | Pre-edge range used to fit the linear background                |
| `BackgroundFit`     | `apply`              | `True`              | Whether the fitted background is subtracted                     |
| `StepNormalisation` | `window_eV`          | `(313.0, 321.0)`    | Post-edge range used to measure the edge-jump step height       |
| `StepNormalisation` | `reducer`            | `np.mean`           | Function reducing that window to a single step-height scalar    |

For example, to fit the background over a different pre-edge range and
measure the step over a different post-edge range:

```python
from diamond_desorption_experiment.parse_nexafs_v2 import (
    BackgroundFit,
    StepNormalisation,
    double_normalise,
)

result = double_normalise(
    "sample.asc",
    "photodiode.asc",
    background=BackgroundFit(fit_window_eV=(275.6, 278.9)),
    step_normalisation=StepNormalisation(window_eV=(332.4, 338.5)),
)
```

`sample_calibration` and `photodiode_calibration` on `double_normalise` take
independent `EnergyCalibration` instances, since the sample and photodiode
scans can need different search windows:

```python
from diamond_desorption_experiment.parse_nexafs_v2 import (
    EnergyCalibration,
    double_normalise,
)

result = double_normalise(
    "sample.asc",
    "photodiode.asc",
    sample_calibration=EnergyCalibration(search_window_eV=(284.8, 288.8)),
)
```
