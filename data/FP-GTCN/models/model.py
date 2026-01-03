"""
模型定义模块
包含GCN-TCN模型及相关组件
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


def _causal_trim(x: torch.Tensor, padding: int) -> torch.Tensor:
    if padding == 0:
        return x
    return x[..., :-padding]


class CausalConv1d(nn.Module):
    """因果卷积（即裁剪保持长度不变）"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1,
                 groups: int = 1, bias: bool = True):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.padding = padding
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=padding, dilation=dilation,
                              groups=groups, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = _causal_trim(y, self.padding)
        return y


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise(逐通道) + Pointwise(1x1) 的因果卷积"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.dw = CausalConv1d(in_ch, in_ch, kernel_size=kernel_size, dilation=dilation, groups=in_ch, bias=bias)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class InceptionBranch(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, depthwise: bool):
        super().__init__()
        if depthwise:
            self.op = DepthwiseSeparableConv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)
        else:
            self.op = CausalConv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.op(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InceptionTCNBlock(nn.Module):
    """
    并行多分支时序卷积块（Inception 风格）：
      - 多个不同 kernel_size 的因果卷积分支并行
      - 可选 avg-pool 分支
      - 1x1 融合 + 残差
    """
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel_set=(3, 5, 7),
                 dilation: int = 1,
                 dropout: float = 0.0,
                 depthwise: bool = False,
                 use_pool_branch: bool = False):
        super().__init__()
        n_br = len(kernel_set) + (1 if use_pool_branch else 0)
        branch_out = max(1, out_ch // n_br)

        self.branches = nn.ModuleList([
            InceptionBranch(in_ch, branch_out, k, dilation=dilation, depthwise=depthwise)
            for k in kernel_set
        ])

        self.use_pool_branch = use_pool_branch
        if use_pool_branch:
            self.pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
            self.pool_pw = nn.Conv1d(in_ch, branch_out, kernel_size=1)
            self.pool_bn = nn.BatchNorm1d(branch_out)
            self.pool_act = nn.ReLU(inplace=True)

        fused_ch = branch_out * n_br
        self.fuse = nn.Conv1d(fused_ch, out_ch, kernel_size=1)
        self.fuse_bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        if self.use_pool_branch:
            p = self.pool(x)
            p = self.pool_pw(p)
            p = self.pool_bn(p)
            p = self.pool_act(p)
            feats.append(p)
        h = torch.cat(feats, dim=1)
        h = self.fuse(h)
        h = self.fuse_bn(h)
        h = self.dropout(h)
        y = h + self.residual(x)
        y = self.out_act(y)
        return y


class InceptionTCN(nn.Module):
    """
    Inception-TCN 堆叠，指数膨胀扩张率（dilation_rates）。
    输入/输出接口与常规 TCN 一致：(B, C_in, T) -> (B, C_out, T)
    """
    def __init__(self,
                 in_ch: int,
                 hidden_ch: int,
                 out_ch: int,
                 num_stages: int,
                 kernel_set=(3, 5, 7),
                 dilation_rates=(1, 2, 4),
                 dropout: float = 0.0,
                 depthwise: bool = False,
                 use_pool_branch: bool = False):
        super().__init__()
        assert num_stages == len(dilation_rates), "num_stages 必须等于 len(dilation_rates)"
        blocks = []
        ch_in = in_ch
        for i in range(num_stages):
            ch_out = hidden_ch if i < num_stages - 1 else out_ch
            blocks.append(InceptionTCNBlock(
                in_ch=ch_in,
                out_ch=ch_out,
                kernel_set=kernel_set,
                dilation=dilation_rates[i],
                dropout=dropout,
                depthwise=depthwise,
                use_pool_branch=use_pool_branch,
            ))
            ch_in = ch_out
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _as_odd(k: int) -> int:
    return k if (k % 2 == 1) else (k + 1)


class TCN(nn.Module):
    """
    新签名（备用）：TCN(in_channels, hidden_dim, out_channels, kernel_size, dilation_rates, dropout)
    内部是 InceptionTCN；仅当你未来需要直接用新签名时才会用到。
    """
    def __init__(self, in_channels, hidden_dim, out_channels, kernel_size, dilation_rates, dropout):
        super().__init__()
        if isinstance(kernel_size, int):
            k1 = 3
            k2 = _as_odd(max(3, int(kernel_size)))
            k3 = _as_odd(k2 + 2)
            kernel_set = (k1, k2, k3)
        else:
            kernel_set = tuple(kernel_size)

        self.impl = InceptionTCN(
            in_ch=in_channels,
            hidden_ch=hidden_dim,
            out_ch=out_channels,
            num_stages=len(dilation_rates),
            kernel_set=kernel_set,
            dilation_rates=dilation_rates,
            dropout=dropout,
            depthwise=False,
            use_pool_branch=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 这里 TCN 约定输入为 (B, C, T)，输出为 (B, C_out, T)
        return self.impl(x)


class TemporalConvNet(nn.Module):
    """
    旧签名（与你现有调用完全一致）：
      TemporalConvNet(in_channels, num_channels_list, kernel_size=KERNEL_SIZE, dropout=dropout, dilation_rates=None)

    - 输入 x 形状：(B, T, C_in)，内部会自动转成 (B, C_in, T) 做卷积；
    - 输出形状保持为 (B, T, C_out)，这样你后面的 `x_seq[:, -1, :]` 不用改。
    """
    def __init__(self, in_channels, num_channels_list, kernel_size=5, dropout=0.0, dilation_rates=None):
        super().__init__()
        assert isinstance(num_channels_list, (list, tuple)) and len(num_channels_list) >= 1, \
            "num_channels_list 必须是长度>=1的列表或元组"

        num_stages = len(num_channels_list)
        # 自动生成指数扩张率（与旧版风格一致）
        if dilation_rates is None:
            dilation_rates = [2 ** i for i in range(num_stages)]
        else:
            assert len(dilation_rates) == num_stages, "dilation_rates 长度需与 num_channels_list 相同"

        # Inception 分支 kernel 集
        if isinstance(kernel_size, int):
            k1 = 3
            k2 = _as_odd(max(3, int(kernel_size)))
            k3 = _as_odd(k2 + 2)
            kernel_set = (k1, k2, k3)
        else:
            kernel_set = tuple(kernel_size)

        hidden_ch = num_channels_list[0]
        out_ch = num_channels_list[-1]

        self.impl = InceptionTCN(
            in_ch=in_channels,
            hidden_ch=hidden_ch,
            out_ch=out_ch,
            num_stages=num_stages,
            kernel_set=kernel_set,
            dilation_rates=dilation_rates,
            dropout=dropout,
            depthwise=False,
            use_pool_branch=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 兼容你当前数据流：输入 (B, T, C) -> 转置为 (B, C, T) 做时序卷积 -> 再转回 (B, T, C)
        # 这样下游的 self.fc(x_seq[:, -1, :]) 保持不变
        x = x.transpose(1, 2)           # (B, C, T)
        y = self.impl(x)                # (B, C_out, T)
        y = y.transpose(1, 2)           # (B, T, C_out)
        return y


class GCNTCN(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, window_size, kernel_size, dropout=0.3):
        super().__init__()
        self.gcn = GCNConv(feature_dim, hidden_dim)
        self.temporal_model = TemporalConvNet(hidden_dim, [hidden_dim, hidden_dim],
                                              kernel_size=kernel_size, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_dim, 1)
        self.window_size = window_size

    def forward(self, sequences, edge_index):
        graph_embeddings = []
        for t in range(self.window_size):
            x_list = [seq[t].x for seq in sequences]
            gcn_out_list = [self.gcn(x, edge_index) for x in x_list]
            pooled = [torch.mean(h, dim=0) for h in gcn_out_list]
            graph_embeddings.append(torch.stack(pooled))
        x_seq = torch.stack(graph_embeddings, dim=1)
        x_seq = self.temporal_model(x_seq)
        return self.fc(x_seq[:, -1, :]).squeeze(-1)