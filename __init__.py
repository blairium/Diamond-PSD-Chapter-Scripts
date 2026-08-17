"""Re-export the public API of :mod:`parse_nexafs`.

This package is checked out as a git submodule at
``src/diamond_desorption_experiment/parse_nexafs`` in the private
Diamond-Desorption-Experiment repository, so
``diamond_desorption_experiment.parse_nexafs`` keeps working as an
ordinary module import everywhere in that codebase.
"""

from .parse_nexafs_v2 import (
    BackgroundFit,
    EnergyCalibration,
    StepNormalisation,
    double_normalise,
    energy_correct,
    parse_nexafs,
)

__all__ = [
    "BackgroundFit",
    "EnergyCalibration",
    "StepNormalisation",
    "double_normalise",
    "energy_correct",
    "parse_nexafs",
]
