"""verl integration shim for the dependency-free PolySeg-Sci core.

The scientific schemas, estimators, allocation, continuation records, and
verifiers live in the top-level :mod:`polyseg_sci.core` package and do not import verl or any
trainer registry.  This module preserves the historical import path and
contains only the estimator-registry adapter used by verl.
"""

from __future__ import annotations

from polyseg_sci.core import *  # noqa: F401,F403
from polyseg_sci.core import (
    RESTORE_EVIDENCE_KINDS,
    _science_config_value,
    _science_field,
    _science_objective_mode,
    compute_polyseg_science_advantage_registered,
    prefix_hash,
    sampling_config_hash,
)
from verl.trainer.ppo.core_algos import AdvantageEstimator, register_adv_est


@register_adv_est(AdvantageEstimator.POLYSEG_SCI)
def _registered_polyseg_science_advantage(*args, **kwargs):
    return compute_polyseg_science_advantage_registered(*args, **kwargs)
