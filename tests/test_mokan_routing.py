from pathlib import Path
import sys
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.MOKAN import Model  # noqa: E402
from models.TimeKAN import Model as UpstreamTimeKAN  # noqa: E402


def _config(feature_dim=7):
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=96,
        label_len=0,
        pred_len=24,
        down_sampling_window=2,
        down_sampling_layers=3,
        channel_independence=1,
        e_layers=2,
        d_model=16,
        begin_order=1,
        moving_avg=25,
        enc_in=feature_dim,
        c_out=feature_dim,
        use_future_temporal_feature=0,
        embed="timeF",
        freq="h",
        dropout=0.0,
        use_norm=1,
        feature_switch_threshold=100,
        num_blocks=2,
        level=-6,
        wavelet_name="db1",
        wavelet_mode="symmetric",
    )


def test_low_dimensional_route_uses_upstream_timekan():
    model = Model(_config()).eval()
    assert model._impl_name == "official_timekan"
    assert isinstance(model._impl, UpstreamTimeKAN)

    with torch.no_grad():
        output = model(torch.randn(2, 96, 7), None, None, None)
    assert output.shape == (2, 24, 7)

