import torch
import torch.nn as nn
import torch.nn.functional as F
import ptwt
from torch.fft import rfft, irfft
from CheKan import ChebyKANLinear
from hippo import HiPPOFeatureTransform, HiPPOTimeBranch

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def _init_params(self):
        # 初始化可学习的仿射变换参数
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _get_statistics(self, x):
        # x shape: [Batch, Time, Features]
        dim2reduce = 1  # 沿着 Time 维度计算均值和方差
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

def wavelet_dec(data, level=1, wavelet='db1'):
    
    batch_size, T, F_dim = data.shape

    # 1. 合并 Batch 和 Feature 维度: [Batch, T, F] -> [Batch, F, T] -> [Batch*F, T]
    x = data.transpose(1, 2).reshape(-1, T)

    # 2. 并行进行小波分解
    # ptwt.wavedec 返回一个列表 [approx, detail_level, detail_level-1, ...]
    coeffs = ptwt.wavedec(x, wavelet=wavelet, level=level, mode='symmetric')

    # 3. 提取需要的近似分量和细节分量 (这里逻辑保持和你原代码一致，只取第一层)
    # 注意：ptwt 返回的列表顺序是 [cA_n, cD_n, cD_n-1, ..., cD_1]
    # 你原代码似乎假设 level=1 或者递归处理了，这里为通用性，我们取最深层的近似和细节
    approx = coeffs[0]  # [Batch*F, T_new]
    detail = coeffs[1]  # [Batch*F, T_new]

    # 4. 拼接
    wave_data = torch.cat([approx, detail], dim=-1)  # [Batch*F, 2 * T_new]

    # 5. 恢复维度: [Batch*F, T_total] -> [Batch, F, T_total] -> [Batch, T_total, F]
    T_total = wave_data.shape[-1]
    wave_data = wave_data.view(batch_size, F_dim, T_total).transpose(1, 2)

    return wave_data


def pooling_dec(data, level=1, axes=[1]):
    """
    使用 PyTorch 的平均池化模拟小波分解，返回第 level-1 级的低频分量。
    np_data: [batch_size, T, F], numpy 数组。
    level: 分解层数。
    axes: 沿哪个轴做分解(仅支持单轴,axes=[1])。
    返回: torch.Tensor,形状匹配小波分解的低频分量。
    """

    # 调整轴序为 [batch_size, F, T] 以适配 AvgPool1d
    data = data.permute(0, 2, 1)

    # 定义平均池化层：窗口大小=2，步幅=2
    pool = nn.AvgPool1d(kernel_size=2, stride=2)

    # 进行 level-1 次池化
    current_data = data
    for _ in range(level - 1):
        current_data = pool(current_data)  # 沿最后一个维度（T）池化

    # 恢复原始轴序 [batch_size, T, F]
    current_data = current_data.permute(0, 2, 1)

    return current_data

# Fourier 变换函数
def fourier_transform(data, level=1):
    data = data.permute(0, 2, 1)  # [batch_size, F, T]
    fft_data = rfft(data, dim=-1)  # [batch_size, F, T//2+1]
    freq_len = fft_data.shape[-1]
    freq_mask = torch.zeros_like(fft_data, dtype=torch.bool)
    if level == 1:
        freq_mask[..., :freq_len // 3] = True
    elif level == 2:
        freq_mask[..., freq_len // 3:2 * freq_len // 3] = True
    elif level == 3:
        freq_mask[..., 2 * freq_len // 3:] = True
    fft_data = fft_data * freq_mask
    time_data = irfft(fft_data, n=data.shape[-1], dim=-1)  # [batch_size, F, T]
    return time_data.permute(0, 2, 1)  # [batch_size, T, F]


class PreProcessBlock(nn.Module):
    def __init__(self, in_length, out_length):
        super(PreProcessBlock, self).__init__()
        self.in_length = in_length
        self.out_length = out_length
        self.mlp = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 48)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(self.in_length + 48, self.out_length),
        )


    def forward(self, input_series, x_original=None):

        B,T,F = input_series.shape  # [batch_size, T, F]

        x = input_series.permute(0, 2, 1).contiguous() # [B,F,T]
        x = x.view(B*F,T,1) #[B*F,T,1]
        x = x.squeeze(-1) #[B*F,T]
        # print(x.shape)
        # print(T)
        if x_original is not None:
            B0, T0, F0 = x_original.shape
            x_raw = x_original.permute(0, 2, 1).contiguous()  # [B,F,T]
            x_raw = x_raw.view(B0 * F0, T0, 1)  # [B*F,T,1]
            x_raw = self.mlp(x_raw.squeeze(-1))  # [B*F,48]
            # print(x_raw.shape)
            # x_raw = x_raw.unsqueeze(-1) #[B*F,48,1]
            x = torch.cat([x, x_raw], dim=1)  # [B*F,T0+T1]

        x = self.fc1(x)  # [B*F,T+T1] -> [B*F,T]

        return x


# FourierDecompositionBlock
class DecompositionBlock(nn.Module):
    def __init__(self, level=None, wavelet='db1'):
        super(DecompositionBlock, self).__init__()
        self.level = level
        self.wavelet = wavelet

    def forward(self, input_series):
        if self.level is None:
            return input_series  # 不应用变换
        elif self.level > 0:
            if self.level <= 3:
                return wavelet_dec(input_series, self.level, wavelet=self.wavelet)
                # fourier_transform(input_series, self.level)
            elif 3 < self.level < 7:
                return fourier_transform(input_series, self.level-3)
            else: return pooling_dec(input_series, self.level - 6)
        elif self.level < 0:
            # Keep legacy behavior: always use 6 branches for original architecture.
            # 3 wavelet branches (lv=1,2,3) + 3 Fourier branches (lv=1,2,3).
            components = [
                wavelet_dec(input_series, level=1, wavelet=self.wavelet),
                wavelet_dec(input_series, level=2, wavelet=self.wavelet),
                wavelet_dec(input_series, level=3, wavelet=self.wavelet),
                fourier_transform(input_series, level=1),
                fourier_transform(input_series, level=2),
                fourier_transform(input_series, level=3),
            ]
            # 打印每个元素的形状
            # print("DecompositionBlock (level < 0) output shapes:")
            # for i, component in enumerate(components):
            #     print(f"Component {i} (level={i + 1 if i < 3 else i - 2}): {component.shape}")
            return components


def get_decomposed_length(T, level):
    """
    根据分解等级和原始长度 T，计算分解后的序列长度。
    """
    if level is None:
        return T
    elif 0 < level <= 3:
        # 注意：这里需要处理奇数偶数除法的细节，建议向下取整以保持安全，或者用 ceil
        # 假设 T 是偶数，或者在 wavelet_dec 中做了 padding
        # Level 1: T/2 + T/2 = T
        # Level 2: T/4 + T/4 = T/2
        # Level 3: T/8 + T/8 = T/4
        denom = 2 ** (level - 1)
        return T // denom

    # --- 傅里叶变换 (Level 4-6) ---
    elif 3 < level < 7:
        return T
    return None


# HiPPOKANBlock
class HiPPOKANBlock(nn.Module):
    def __init__(self, in_length, num_features=1, head=4):
        super(HiPPOKANBlock, self).__init__()

        self.kan = ChebyKANLinear(in_length, in_length, 3)

        self.temporal_mixer = nn.Sequential(
            nn.Linear(in_length, in_length),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.gate = nn.Linear(in_length, in_length)

        self.norm = nn.LayerNorm(in_length)
        
        self.num_features = num_features
        
        # if num_features > 1 and num_features <= 50:
        #     self.feature_cross_attn = nn.MultiheadAttention(
        #         embed_dim=in_length,
        #         num_heads=min(2, num_features),
        #         batch_first=True
        #     )
        #     self.norm_feature = nn.LayerNorm(in_length)
        #     self.use_cross_attn = True
        # else:
        #     self.use_cross_attn = False


    def forward(self, input_series):
        x = input_series # [B*F,T]
        x = self.kan(x)  # [B*F,T]

        # if self.use_cross_attn:
        #     B_total, T = x.shape[0], x.shape[1]
        #     B = B_total // self.num_features
            
        #     x_reshaped = x.view(B, self.num_features, T)  # [B, F, T]
        #     x_cross, _ = self.feature_cross_attn(x_reshaped, x_reshaped, x_reshaped)
        #     x_cross = x_cross.contiguous().view(-1, T)  # [B*F, T]

        res = x
        x_mixed = self.temporal_mixer(x)  # [B*F, T]

        gate = torch.sigmoid(self.gate(x))
        x = x_mixed * gate
        # if self.use_cross_attn:
        #     x = x_mixed * gate + x_cross * (1 - gate)
        # else:
        #     x = x_mixed * gate

        x = self.norm(x + res)

        return x




# WaveletHiPPOKAN（单个频带）
class WaveletHiPPOKAN(nn.Module):
    def __init__(self, in_length, out_length, head=4,
                 decomp_level=None):
        super(WaveletHiPPOKAN, self).__init__()
        self.out_length = out_length
        self.in_length = in_length
        self.decomp = DecompositionBlock(level=decomp_level)
        self.preprocess = PreProcessBlock(self.in_length, self.out_length)
        self.hippo_kan = HiPPOKANBlock(self.out_length, head)
        self.fc = nn.Linear(self.out_length, self.out_length )

    def forward(self, input_series, x_original=None):
        x = self.decomp(input_series) # [B,T_decomp,F] # 先应用分解
        # print(x.shape)
        B, T, F = input_series.shape
        x = self.preprocess(x, x_original) # [B*F,T_out]
        x = self.hippo_kan(x) # [B*F, T_out]
        x = self.fc(x)  # [B*F, T_out]
        x = x.view(B, F, -1) # [B,F,T_out]
        x = x.permute(0, 2, 1).contiguous() # [B,T_out,F]

        return x


class GatedFusion(nn.Module):
    """
    门控融合单元，用于动态加权 x_wave 和 x_fourier。
    输入 x_fused 形状: [B, T, F, 2]，其中 x_fused[..., 0] 是 x_wave，x_fused[..., 1] 是 x_fourier
    """

    def __init__(self, feature_dim):
        super(GatedFusion, self).__init__()

        # 门控线性层：从输入的两个特征维度映射回一个门控权重
        # feature_dim = 2 (因为我们堆叠了 x_wave 和 x_fourier)
        self.gate_linear = nn.Linear(feature_dim, 1)

    def forward(self, x_fused):
        # x_fused 形状: [B, T, F, 2]
        x_wave = x_fused[..., 0]  # [B, T, F]
        x_fourier = x_fused[..., 1]  # [B, T, F]

        # 1. 计算门控权重 g
        # 使用 sigmoid 确保 g 的值在 [0, 1] 之间
        # 门控线性层对维度 2（即 [x_wave, x_fourier]）进行操作
        # g 的形状: [B, T, F, 1]
        g = torch.sigmoid(self.gate_linear(x_fused))

        # 2. 应用门控机制进行加权融合
        # 结果形状: [B, T, F]
        # g 决定了 x_wave 的贡献
        # (1 - g) 决定了 x_fourier 的贡献
        x_out = g.squeeze(-1) * x_wave + (1 - g).squeeze(-1) * x_fourier

        return x_out
# EnsembleKAN
class EnsembleKAN(nn.Module):
    def __init__(self, branches, out_length, in_length=96,
                 use_hippo_time_branch=False, hippo_N=32):
        super(EnsembleKAN, self).__init__()
        # self.in_shape = in_shape
        self.out_length = out_length
        self.in_length = in_length
        self.branches = nn.ModuleList(branches)
        if use_hippo_time_branch:
            self.time_branch = HiPPOTimeBranch(in_length, out_length, N=hippo_N)
        else:
            self.time_branch = nn.Linear(in_length, out_length)
        # self.Feature = out_shape[-1]
        self.attention = nn.MultiheadAttention(
            embed_dim=self.out_length,
            num_heads=4,
            batch_first=True
        )
        # self.fusion_layer = nn.Sequential(
        #     nn.Linear(2, 1)
        #     # nn.ReLU()
        # )
        self.fusion_layer = GatedFusion(feature_dim=3)  # 维度是 2，因为 x_wave 和 x_fourier 被堆叠
    def forward(self, inputs, x_original):
        x = [b(inp, x_original) for b, inp in zip(self.branches, inputs)]

        w_low = x[0]
        w_mid = x[1]
        w_high = x[2]
        # 上采样到 out_length
        w_mid = F.interpolate(w_mid.permute(0, 2, 1), size=self.out_length,
                              mode='linear', align_corners=False).permute(0, 2, 1).contiguous()
        w_high = F.interpolate(w_high.permute(0, 2, 1), size=self.out_length,
                               mode='linear', align_corners=False).permute(0, 2, 1).contiguous()

        x_wave = w_low + w_mid + w_high

        f_low = x[3]
        f_mid = x[4]
        f_high = x[5]
        x_fourier = f_low + f_mid + f_high

        # Fallback to branch input when external x_original length does not match.
        x_time_src = x_original
        if x_time_src is None or x_time_src.shape[1] != self.in_length:
            x_time_src = inputs[0]
        if isinstance(self.time_branch, HiPPOTimeBranch):
            x_time = self.time_branch(x_time_src)  # [B, T_out, F]
        else:
            x_time = self.time_branch(x_time_src.permute(0, 2, 1)).permute(0, 2, 1)

        x = torch.stack([x_wave, x_fourier, x_time],dim=-1) # [B,T,F,2]
        x_fusion = self.fusion_layer(x) # [B,T,F,1]
        # x = x.squeeze(-1) #[B,T,F]

        x_att = x_fusion.permute(0, 2, 1).contiguous()  # [B,F,T]
        x_att_out, _ = self.attention(x_att, x_att, x_att)
        x_att_out = x_att_out.permute(0, 2, 1).contiguous()  # [B,T,F]

        x = x_fusion + x_att_out
        return x




# FrequencyEnsembleModel（单层模型）
class FrequencyEnsembleModel(nn.Module):
    def __init__(self, out_length,
                 in_length=96, level=None, pretrained_branches=None, wavelet='db1',
                 use_hippo_time_branch=False, hippo_N=32):
        super(FrequencyEnsembleModel, self).__init__()
        self.decomp = DecompositionBlock(level, wavelet=wavelet)
        self.in_length = in_length

        if pretrained_branches is None:
            branch_count = 6 if (level is not None and level < 0) else (-level)
            self.branches = nn.ModuleList([
                WaveletHiPPOKAN(get_decomposed_length(self.in_length, lv + 1), out_length, decomp_level=None)
                for lv in range(branch_count)
            ])
        else:
            self.branches = nn.ModuleList(pretrained_branches)
        self.ensemble = EnsembleKAN(self.branches, out_length, in_length=self.in_length,
                                    use_hippo_time_branch=use_hippo_time_branch, hippo_N=hippo_N)

    def forward(self, input_series, x_orginal=None):
        components = self.decomp(input_series)
        # for component in components:
        #     print(component.shape)
        return self.ensemble(components, x_orginal)


# StackedFrequencyEnsembleModel（堆叠多层）
class StackedFrequencyEnsembleModel(nn.Module):
    def __init__(self, features, out_length,
                 in_length=96, num_blocks=2, level=None, pretrained_branches=None, wavelet='db1',
                 hippo_target_dim=None, hippo_N=32, use_hippo_time_branch=False):
        super(StackedFrequencyEnsembleModel, self).__init__()
        self.num_blocks = num_blocks  # 控制 Block 数量（L）
        self.in_length = in_length
        self.features = features
        self.hippo_target_dim = hippo_target_dim

        # Determine if HiPPO feature transform should be applied
        self.use_hippo_transform = (
            hippo_target_dim is not None and hippo_target_dim != features
        )
        internal_features = hippo_target_dim if self.use_hippo_transform else features

        # 1. 初始化 RevIN (always uses original features for input norm)
        self.revin = RevIN(features, affine=True)

        # 2. HiPPO feature transform (optional)
        if self.use_hippo_transform:
            self.hippo_transform = HiPPOFeatureTransform(features, hippo_target_dim, N=hippo_N)
            self.output_proj = nn.Linear(hippo_target_dim, features)
        else:
            self.hippo_transform = None
            self.output_proj = None

        # 3. Internal layers
        self.layers = nn.ModuleList()
        self.res_projections = nn.ModuleList()  # 用于调整残差维度的层

        self.layers.append(FrequencyEnsembleModel(
            out_length=out_length,
            in_length=self.in_length,
            level=level,
            pretrained_branches=pretrained_branches,
            wavelet=wavelet,
            use_hippo_time_branch=use_hippo_time_branch,
            hippo_N=hippo_N,
        ))
        # 第一层：输入长度可能不等于输出长度
        if self.in_length != out_length:
            self.res_projections.append(nn.Linear(self.in_length, out_length))
        else:
            self.res_projections.append(nn.Identity())

        for _ in range(num_blocks - 1):
            self.layers.append(FrequencyEnsembleModel(
                out_length=out_length,
                in_length=out_length,
                level=level,
                pretrained_branches=None,
                wavelet=wavelet,
                use_hippo_time_branch=use_hippo_time_branch,
                hippo_N=hippo_N,
            ))
            self.res_projections.append(nn.Identity())

        # 4. BatchNorm layers use internal_features
        self.norm_layers = nn.ModuleList([
            nn.BatchNorm1d(internal_features) for _ in range(num_blocks)
        ])

    def forward(self, input_series, x_original=None):
        # RevIN norm on original features
        x = self.revin(input_series, 'norm')  # [B, T_in, F_orig]

        # Optional: HiPPO feature transform (also transform x_original for consistency)
        if self.use_hippo_transform:
            x = self.hippo_transform(x)  # [B, T_in, F_target]
            if x_original is not None:
                x_original = self.hippo_transform(x_original)  # [B, T_in, F_target]

        for i, layer in enumerate(self.layers):
            identity = x

            out = layer(x, x_original)  # 通过当前层计算输出

            if not isinstance(self.res_projections[i], nn.Identity):
                # 转置用于 Linear 变换时间轴: [B, T, F] -> [B, F, T] -> Linear -> [B, F, T_out]
                identity = identity.permute(0, 2, 1)
                identity = self.res_projections[i](identity)  # Linear(in_len, out_len)
                identity = identity.permute(0, 2, 1)

            # 残差连接
            x = out + identity
            # 归一化 (BatchNorm)
            x = x.permute(0, 2, 1).contiguous()
            x = self.norm_layers[i](x)
            x = x.permute(0, 2, 1).contiguous()

        # Optional: project back to original feature dim
        if self.use_hippo_transform:
            # [B, T_out, F_target] -> Linear(F_target, F_orig) -> [B, T_out, F_orig]
            x = self.output_proj(x)

        x = self.revin(x, 'denorm')
        return x