"""
constants.py
============
Constants and default variable lists for the transformer jet tagging project.
"""

JET_VARS_DEFAULT = ["pt", "eta"]

TRACK_VARS_DEFAULT = [
    # Perigee parametrization
    "qOverP", "deta", "dphi", "d0", "z0SinTheta",
    # Diagonal of covariance matrix
    "qOverPUncertainty", "thetaUncertainty", "phiUncertainty",
    # Lifetime-signed impact parameter significances
    "lifetimeSignedD0Significance", "lifetimeSignedZ0SinThetaSignificance",
    # Hit-level variables
    "numberOfPixelHits", "numberOfSCTHits",
    "numberOfInnermostPixelLayerHits", "numberOfNextToInnermostPixelLayerHits",
    "numberOfInnermostPixelLayerSharedHits", "numberOfInnermostPixelLayerSplitHits",
    "numberOfPixelSharedHits", "numberOfPixelSplitHits", "numberOfSCTSharedHits",
]

JET_FLAVOUR_LABEL = "HadronConeExclTruthLabelID"

# Maps HadronConeExclTruthLabelID
JET_FLAVOUR_MAP = {0: 0, 4: 1, 5: 2, 15: 3}
FLAVOUR_LABELS = {0: "b-jet", 1: "c-jet", 2: "light-jet", 3: "tau-jet"}
FLAVOUR_LABELS_INV: dict[str, int] = {v: k for k, v in FLAVOUR_LABELS.items()}
FLAVOUR_COLORS = {0: "#55A868", 1: "#DD8452", 2: "#4C72B0", 3: "#C44E52"}

# Supported activations
SUPPORTED_ACTIVATIONS = ("relu", "leakyrelu", "sigmoid", "tanh", "softplus")

# Supported optimizers
SUPPORTED_OPTIMIZERS = ("adamw", "adam", "sgd")

# Supported devices
SUPPORTED_DEVICES = ("auto", "cpu", "cuda")
