"""MOKAN model entry point.

MOKAN means Multi-scale Orthogonal-memory KAN. It routes low-dimensional
inputs through the upstream TimeKAN implementation and uses the project
extension for high-dimensional inputs.
"""

from types import SimpleNamespace

from mokan.wavelet_multiscale_model import AdaptiveModel


class Model(AdaptiveModel):
    """Dimension-adaptive MOKAN model exposed to the standard runner."""

    def __init__(self, configs):
        values = vars(configs).copy()
        values.setdefault("begin_order", 1)
        values.setdefault("feature_switch_threshold", 100)
        values.setdefault("num_blocks", 2)
        values.setdefault("level", -6)
        values.setdefault("wavelet_name", "db1")
        values.setdefault("wavelet_mode", "symmetric")
        super().__init__(SimpleNamespace(**values))

