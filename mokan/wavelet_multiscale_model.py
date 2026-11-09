import math
import importlib
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

try:
    import ptwt
except ModuleNotFoundError:  # Optional: only required by the high-dimensional wavelet path.
    ptwt = None

# Prefer project-native modules when available.
try:
    from layers.Autoformer_EncDec import series_decomp  # type: ignore
    from layers.Embed import DataEmbedding_wo_pos  # type: ignore
    from layers.StandardNorm import Normalize  # type: ignore
    from layers.ChebyKANLayer import ChebyKANLinear  # type: ignore
except Exception:
    from CheKan import ChebyKANLinear

    class _MovingAvg(nn.Module):
        def __init__(self, kernel_size: int):
            super().__init__()
            self.kernel_size = kernel_size
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

        def forward(self, x):
            # x: [B, T, C]
            front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
            x = torch.cat([front, x, end], dim=1)
            x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
            return x

    class series_decomp(nn.Module):
        def __init__(self, kernel_size: int):
            super().__init__()
            self.moving_avg = _MovingAvg(kernel_size)

        def forward(self, x):
            moving_mean = self.moving_avg(x)
            return x - moving_mean, moving_mean

    class DataEmbedding_wo_pos(nn.Module):
        def __init__(self, c_in, d_model, embed=None, freq=None, dropout=0.1):
            super().__init__()
            self.value_embedding = nn.Linear(c_in, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, x_mark=None):
            return self.dropout(self.value_embedding(x))

    class Normalize(nn.Module):
        def __init__(self, num_features, affine=True, non_norm=False, eps=1e-5):
            super().__init__()
            self.num_features = num_features
            self.affine = affine
            self.non_norm = non_norm
            self.eps = eps
            if self.affine:
                self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
                self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))

        def forward(self, x, mode='norm'):
            if self.non_norm:
                return x
            if mode == 'norm':
                self.mean = x.mean(dim=1, keepdim=True).detach()
                self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
                x = (x - self.mean) / self.stdev
                if self.affine:
                    x = x * self.affine_weight + self.affine_bias
                return x
            if mode == 'denorm':
                if self.affine:
                    x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
                x = x * self.stdev + self.mean
                return x
            raise ValueError(f'Unsupported mode: {mode}')


class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features, order):
        super().__init__()
        self.fc1 = ChebyKANLinear(in_features, out_features, order)

    def forward(self, x):
        bsz, t_len, c_dim = x.shape
        x = self.fc1(x.reshape(bsz * t_len, c_dim))
        x = x.reshape(bsz, t_len, -1).contiguous()
        return x


class FrequencyDecomp(nn.Module):
    def __init__(self, configs):
        super(FrequencyDecomp, self).__init__()
        self.configs = configs

    def forward(self, level_list):
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high_res = self.frequency_interpolation(
                out_low.transpose(1, 2),
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers - i)),
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers - i - 1)),
            ).transpose(1, 2)
            out_high_left = out_high - out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]
            out_level_list.append(out_high_left)
        out_level_list.reverse()
        return out_level_list

    def frequency_interpolation(self, x, seq_len, target_len):
        len_ratio = seq_len / target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros([x_fft.size(0), x_fft.size(1), target_len // 2 + 1], dtype=x_fft.dtype, device=x_fft.device)
        out_fft[:, :, : seq_len // 2 + 1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2)
        out = out * len_ratio
        return out


class FrequencyMixing(nn.Module):
    def __init__(self, configs):
        super(FrequencyMixing, self).__init__()
        self.configs = configs
        self.front_block = M_KAN(
            configs.d_model,
            self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers)),
            order=configs.begin_order,
        )

        self.front_blocks = torch.nn.ModuleList(
            [
                M_KAN(
                    configs.d_model,
                    self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers - i - 1)),
                    order=i + configs.begin_order + 1,
                )
                for i in range(configs.down_sampling_layers)
            ]
        )

    def forward(self, level_list):
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_low = self.front_block(out_low)
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high = self.front_blocks[i](out_high)
            out_high_res = self.frequency_interpolation(
                out_low.transpose(1, 2),
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers - i)),
                self.configs.seq_len // (self.configs.down_sampling_window ** (self.configs.down_sampling_layers - i - 1)),
            ).transpose(1, 2)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]
            out_level_list.append(out_low)
        out_level_list.reverse()
        return out_level_list

    def frequency_interpolation(self, x, seq_len, target_len):
        len_ratio = seq_len / target_len
        x_fft = torch.fft.rfft(x, dim=2)
        out_fft = torch.zeros([x_fft.size(0), x_fft.size(1), target_len // 2 + 1], dtype=x_fft.dtype, device=x_fft.device)
        out_fft[:, :, : seq_len // 2 + 1] = x_fft
        out = torch.fft.irfft(out_fft, dim=2)
        out = out * len_ratio
        return out


class M_KAN(nn.Module):
    def __init__(self, d_model, seq_len, order):
        super().__init__()
        self.channel_mixer = nn.Sequential(ChebyKANLayer(d_model, d_model, order))
        self.conv = BasicConv(d_model, d_model, kernel_size=3, degree=order, groups=d_model)

    def forward(self, x):
        x1 = self.channel_mixer(x)
        x2 = self.conv(x)
        out = x1 + x2
        return out


class BasicConv(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size,
        degree,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        act=False,
        bn=False,
        bias=False,
        dropout=0.0,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = c_out
        self.conv = nn.Conv1d(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class WaveletMultiscaleModel(nn.Module):
    def __init__(self, configs):
        super(WaveletMultiscaleModel, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.res_blocks = nn.ModuleList([FrequencyDecomp(configs) for _ in range(configs.e_layers)])
        self.add_blocks = nn.ModuleList([FrequencyMixing(configs) for _ in range(configs.e_layers)])

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature

        self.wavelet_name = getattr(configs, 'wavelet_name', 'db1')
        self.wavelet_mode = getattr(configs, 'wavelet_mode', 'symmetric')

        self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for _ in range(configs.down_sampling_layers + 1)
            ]
        )
        self.projection_layer = nn.Linear(configs.d_model, 1, bias=True)
        self.predict_layer = nn.Linear(configs.seq_len, configs.pred_len)

    def forecast(self, x_enc):
        x_enc = self.__multi_level_process_inputs(x_enc)
        x_list = []
        for i, x in zip(range(len(x_enc)), x_enc):
            bsz, t_len, n_vars = x.size()
            x = self.normalize_layers[i](x, 'norm')
            x = x.permute(0, 2, 1).contiguous().reshape(bsz * n_vars, t_len, 1)
            x_list.append(x)

        enc_out_list = []
        for i, x in zip(range(len(x_list)), x_list):
            enc_out = self.enc_embedding(x, None)
            enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.res_blocks[i](enc_out_list)
            enc_out_list = self.add_blocks[i](enc_out_list)

        dec_out = enc_out_list[0]
        dec_out = self.predict_layer(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = self.projection_layer(dec_out).reshape(bsz, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def __resize_time_length(self, x, target_len):
        if x.size(1) == target_len:
            return x
        x = F.interpolate(x.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
        return x

    def __wavelet_downsample(self, x, target_len):
        # x: [B, T, C] -> flatten channels to do batched DWT in one call.
        if ptwt is None:
            raise ImportError(
                "The high-dimensional wavelet path requires ptwt. "
                "Install the project requirements before using it."
            )
        bsz, t_len, c_dim = x.shape
        x_flat = x.transpose(1, 2).reshape(bsz * c_dim, t_len)
        coeffs = ptwt.wavedec(x_flat, wavelet=self.wavelet_name, level=1, mode=self.wavelet_mode)
        approx = coeffs[0]
        approx = approx.reshape(bsz, c_dim, -1).transpose(1, 2).contiguous()
        approx = self.__resize_time_length(approx, target_len)
        return approx

    def __multi_level_process_inputs(self, x_enc):
        # Replace AvgPool multiscale with wavelet approximation pyramid.
        x_levels = [x_enc]
        x_cur = x_enc
        for i in range(self.configs.down_sampling_layers):
            target_len = self.configs.seq_len // (self.configs.down_sampling_window ** (i + 1))
            x_cur = self.__wavelet_downsample(x_cur, target_len)
            x_levels.append(x_cur)
        return x_levels

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out
        raise ValueError('Other tasks implemented yet')


class AdaptiveModel(nn.Module):
    """Route models by input feature count.

    Low-dimensional datasets use the unmodified upstream TimeKAN model. The
    custom wavelet/HiPPO stack is reserved for high-dimensional datasets where
    the additional feature mixing is useful.
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.pred_len = int(configs.pred_len)
        self.num_blocks = int(getattr(configs, "num_blocks", 2))
        self.level = int(getattr(configs, "level", -6))
        self.feature_switch_threshold = int(getattr(configs, "feature_switch_threshold", 100))

        self._impl = None
        self._impl_name = None
        self._built_feature_dim = None

        init_feature_dim = int(getattr(configs, "enc_in", 0))
        if init_feature_dim > 0:
            self._build_impl(init_feature_dim, torch.device("cpu"))

    @staticmethod
    def _load_official_timekan():
        """Load the upstream TimeKAN model without depending on cwd.

        The public repository keeps the upstream checkout next to this project
        by default. ``TIMEKAN_OFFICIAL_ROOT`` can point to another checkout,
        which is useful when the extension is installed separately.
        """
        source_root = Path(__file__).resolve()
        candidates = []
        configured_root = os.environ.get("TIMEKAN_OFFICIAL_ROOT")
        if configured_root:
            candidates.append(Path(configured_root))
        candidates.extend(
            [
                source_root.parents[2] / "MOKAN",
                source_root.parents[3] / "MOKAN",
            ]
        )
        for candidate in candidates:
            if (candidate / "models" / "TimeKAN.py").is_file():
                candidate_str = str(candidate)
                if candidate_str not in sys.path:
                    sys.path.insert(0, candidate_str)
                try:
                    return importlib.import_module("models.TimeKAN").Model
                except ModuleNotFoundError:
                    continue
        try:
            return importlib.import_module("models.TimeKAN").Model
        except ModuleNotFoundError as first_error:
            raise ImportError(
                "Unable to load the upstream TimeKAN model. Set "
                "TIMEKAN_OFFICIAL_ROOT to a checkout containing "
                "models/TimeKAN.py."
            ) from first_error

    @staticmethod
    def _load_high_dim_backend():
        """Load the project-specific high-dimensional implementation."""
        if ptwt is None:
            raise ImportError(
                "The high-dimensional adaptive path requires ptwt. "
                "Install requirements.txt before selecting enc_in above "
                "the feature switch threshold."
            )
        try:
            from original_architecture import StackedFrequencyEnsembleModel
            return StackedFrequencyEnsembleModel
        except ModuleNotFoundError:
            source_dir = str(Path(__file__).resolve().parent)
            if source_dir not in sys.path:
                sys.path.insert(0, source_dir)
            from original_architecture import StackedFrequencyEnsembleModel
            return StackedFrequencyEnsembleModel

    def _build_impl(self, feature_dim: int, device: torch.device):
        use_high_dim = feature_dim > self.feature_switch_threshold
        if not use_high_dim:
            BackendModel = self._load_official_timekan()
            cfg = SimpleNamespace(**self.configs.__dict__)
            cfg.enc_in = feature_dim
            cfg.c_out = feature_dim
            self._impl_name = "official_timekan"
            self._impl = BackendModel(cfg).to(device)
        else:
            BackendModel = self._load_high_dim_backend()

            hippo_target_dim = int(getattr(self.configs, "hippo_target_dim", 0) or 0)
            hippo_N = int(getattr(self.configs, "hippo_N", 32))
            use_hippo_time_branch = bool(getattr(self.configs, "use_hippo_time_branch", False))

            self._impl_name = "original_architecture"
            self._impl = BackendModel(
                features=feature_dim,
                out_length=self.pred_len,
                in_length=int(getattr(self.configs, "seq_len", 96)),
                num_blocks=self.num_blocks,
                level=self.level,
                wavelet=getattr(self.configs, "wavelet_name", "db1"),
                hippo_target_dim=hippo_target_dim if hippo_target_dim > 0 else None,
                hippo_N=hippo_N,
                use_hippo_time_branch=use_hippo_time_branch,
            ).to(device)
        self._built_feature_dim = feature_dim

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        bsz, seq_len, feature_dim = x_enc.shape
        _ = (bsz, seq_len, x_mark_enc, x_mark_dec, mask)

        if self._impl is None or self._built_feature_dim != feature_dim:
            self._build_impl(feature_dim, x_enc.device)

        x_original = x_enc
        if isinstance(x_dec, torch.Tensor) and x_dec.shape == x_enc.shape:
            x_original = x_dec
        if self._impl_name == "official_timekan":
            return self._impl(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        return self._impl(x_enc, x_original)


Model = AdaptiveModel
