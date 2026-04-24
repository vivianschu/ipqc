"""SUPERSEDED — do not import.

MHC-I binding predictions now use locally-installed open-source tools via
modules/prediction.py and modules/predictors/.  This file is kept only as a
tombstone so that any stale import immediately surfaces a clear error.
"""
raise ImportError(
    "modules.iedb has been removed. "
    "Use modules.prediction and modules.predictors instead."
)
