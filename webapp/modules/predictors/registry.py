"""Registry of all known MHC-I prediction backends."""
from __future__ import annotations

from .base import BaseMHCIPredictor
from .mhcflurry_predictor import MHCflurryPredictor
from .netmhcpan_predictor import NetMHCpanPredictor
from .netmhcstabpan_predictor import NetMHCstabpanPredictor

# Ordered by priority / maturity
ALL_PREDICTORS: list[type[BaseMHCIPredictor]] = [
    MHCflurryPredictor,
    NetMHCpanPredictor,
    NetMHCstabpanPredictor,
]


def get_available_predictors() -> list[type[BaseMHCIPredictor]]:
    """Return predictor classes whose dependencies are installed."""
    return [cls for cls in ALL_PREDICTORS if cls.is_available()]


def get_predictors_by_type(predictor_type: str) -> list[type[BaseMHCIPredictor]]:
    """Return available predictor classes matching *predictor_type* ('binding' or 'stability')."""
    return [cls for cls in ALL_PREDICTORS if cls.predictor_type == predictor_type and cls.is_available()]


def get_predictor_by_name(name: str) -> type[BaseMHCIPredictor] | None:
    for cls in ALL_PREDICTORS:
        if cls.name == name:
            return cls
    return None
