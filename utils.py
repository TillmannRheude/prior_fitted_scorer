import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


import torch
import torch.nn as nn
import math


class NoResidualTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", batch_first=True, norm_first=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm_first = norm_first

    def _sa_block(self, x, attn_mask=None, key_padding_mask=None, is_causal=False):
        x, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if self.norm_first:
            x = self._sa_block(self.norm1(src), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = self._ff_block(self.norm2(x))
        else:
            x = self.norm1(self._sa_block(src, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(self._ff_block(x))
        return x


class ScaledResidualTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """
    TransformerEncoderLayer with configurable residual strength alpha:

        x <- x + alpha * block(x)

    alpha=1.0 recovers the standard residual transformer. Smaller values weaken
    the residual branch contribution while keeping the same attention/FFN
    sublayers and normalization behavior.
    """
    def __init__(self, *args, residual_alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_alpha = float(residual_alpha)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            x = self.residual_alpha * x + (self._sa_block(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                is_causal=is_causal,
            ))
            x = self.residual_alpha * x + (self._ff_block(self.norm2(x)))
        else:
            x = self.norm1(
                self.residual_alpha * x + (self._sa_block(
                    x,
                    src_mask,
                    src_key_padding_mask,
                    is_causal=is_causal,
                ))
            )
            x = self.norm2(self.residual_alpha * x + (self._ff_block(x)))
        return x

class _DummySelfAttn:
    def __init__(self, batch_first=True):
        self.batch_first = batch_first
        self._qkv_same_embed_dim = True
        self.in_proj_bias = None
        self.in_proj_weight = None
        self.out_proj = None

class TokenwiseMLPEncoderLayer(nn.Module):
    """
    Drop-in replacement for TransformerEncoderLayer that removes attention and
    applies the same MLP independently at every token position.

    This keeps the forward signature expected by nn.TransformerEncoder so it
    can be swapped in the same way as NoResidualTransformerEncoderLayer.
    """
    def __init__(
        self,
        d_model,
        nhead=None,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        batch_first=True,
        norm_first=False,
    ):
        super().__init__()
        self.self_attn = _DummySelfAttn(batch_first=batch_first)

        self.d_model = int(d_model)
        self.batch_first = bool(batch_first)
        self.norm_first = bool(norm_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.out_dropout = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def _mlp_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.out_dropout(x)
        return x

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        # Kept for compatibility with nn.TransformerEncoder; this layer is
        # intentionally token-wise and ignores sequence masks.
        del src_mask, src_key_padding_mask, is_causal

        if self.norm_first:
            x = self._mlp_block(self.norm(src))
        else:
            x = self.norm(self._mlp_block(src))
        return x


class UniformAttentionEncoderLayer(nn.Module):
    """
    Drop-in encoder layer that replaces learned self-attention with fixed
    uniform averaging over tokens:

        h_i = (1 / T) sum_j x_j

    followed by a position-wise feedforward block. This keeps the same forward
    signature expected by nn.TransformerEncoder.
    """
    def __init__(
        self,
        d_model,
        nhead=None,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        batch_first=True,
        norm_first=False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.batch_first = bool(batch_first)
        self.norm_first = bool(norm_first)

        # Dummy attribute for nn.TransformerEncoder internal checks.
        self.self_attn = _DummySelfAttn(batch_first=batch_first)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def _uniform_attention_block(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (B, T, D) or (T, B, D), got {tuple(x.shape)}")

        if self.batch_first:
            # x: (B, T, D)
            if src_key_padding_mask is None:
                mean_tok = x.mean(dim=1, keepdim=True)
            else:
                keep = (~src_key_padding_mask).to(dtype=x.dtype, device=x.device)  # (B, T)
                denom = keep.sum(dim=1, keepdim=True).clamp_min(1.0)
                mean_tok = (x * keep.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom.unsqueeze(-1)
            out = mean_tok.expand(-1, x.shape[1], -1)
        else:
            # x: (T, B, D)
            if src_key_padding_mask is None:
                mean_tok = x.mean(dim=0, keepdim=True)
            else:
                keep = (~src_key_padding_mask).to(dtype=x.dtype, device=x.device).transpose(0, 1)  # (T, B)
                denom = keep.sum(dim=0, keepdim=True).clamp_min(1.0)
                mean_tok = (x * keep.unsqueeze(-1)).sum(dim=0, keepdim=True) / denom.unsqueeze(-1)
            out = mean_tok.expand(x.shape[0], -1, -1)

        return self.dropout1(out)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        del src_mask, is_causal

        if self.norm_first:
            x = self._uniform_attention_block(self.norm1(src), src_key_padding_mask=src_key_padding_mask)
            x = self._ff_block(self.norm2(x))
        else:
            x = self.norm1(self._uniform_attention_block(src, src_key_padding_mask=src_key_padding_mask))
            x = self.norm2(self._ff_block(x))
        return x


def sinusoidal_positional_encoding(
    seq_len: int,
    d_model: int,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Standard sinusoidal positional encoding tensor of shape (1, seq_len, d_model).
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if d_model < 1:
        raise ValueError(f"d_model must be >= 1, got {d_model}")

    positions = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)  # (S, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d_model)
    )  # (ceil(D/2),)

    pe = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(positions * div_term)
    if d_model > 1:
        pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])

    return pe.unsqueeze(0)  # (1, S, D)

@torch.no_grad()
def burkholz_relu_init_linear(layer: nn.Linear, mode: str = "orthogonal"):
    """
    Burkholz-Dubatovka block initialization for a ReLU Linear layer:

        W = [[ W0, -W0],
             [-W0,  W0]]

    Requires both in_features and out_features to be even.
    """
    if not isinstance(layer, nn.Linear):
        return

    out_dim, in_dim = layer.weight.shape

    if in_dim % 2 != 0 or out_dim % 2 != 0:
        raise ValueError(
            f"Burkholz init requires even dims, got in={in_dim}, out={out_dim}"
        )

    half_out = out_dim // 2
    half_in = in_dim // 2

    if mode == "gsm":
        # Paper's GSM choice: sigma_w^2 = 2 / N_l.
        # For PyTorch orientation, N_l corresponds to output layer width.
        std = math.sqrt(2.0 / out_dim)
        W0 = torch.randn(
            half_out,
            half_in,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        ) * std

    elif mode == "orthogonal":
        W0 = torch.empty(
            half_out,
            half_in,
            device=layer.weight.device,
            dtype=layer.weight.dtype,
        )
        nn.init.orthogonal_(W0)

        # Use gain 1, not sqrt(2), for the perfect-isometry variant.
        # This matches the paper's sigma_w^2 = 1 orthogonal W0 condition.
    else:
        raise ValueError("mode must be 'gsm' or 'orthogonal'")

    W = torch.empty_like(layer.weight)
    W[:half_out, :half_in] = W0
    W[:half_out, half_in:] = -W0
    W[half_out:, :half_in] = -W0
    W[half_out:, half_in:] = W0

    layer.weight.copy_(W)

    if layer.bias is not None:
        layer.bias.zero_()


class AdditiveCouplingBlock(nn.Module):
    """
    Simple invertible additive coupling block for tabular vectors.

    Split x = [x1, x2] and apply:
        y1 = x1
        y2 = x2 + alpha * t(x1)

    Optionally swap halves before/after the transform to alternate which part
    gets updated across stacked blocks. This is invertible by construction and
    starts near the identity when alpha is small.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        dropout: float = 0.0,
        alpha_init: float = 1e-2,
        swap: bool = False,
    ):
        super().__init__()
        if dim < 2:
            raise ValueError("AdditiveCouplingBlock requires dim >= 2.")

        self.dim = int(dim)
        self.swap = bool(swap)
        self.split1 = self.dim // 2
        self.split2 = self.dim - self.split1
        hidden_dim = self.dim if hidden_dim is None else int(hidden_dim)

        self.net = nn.Sequential(
            nn.LayerNorm(self.split1),
            nn.Linear(self.split1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.split2),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self._init_near_zero_()

    def _init_near_zero_(self) -> None:
        with torch.no_grad():
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {x.shape[-1]}")

        if self.swap:
            x = torch.cat([x[..., self.split1:], x[..., :self.split1]], dim=-1)

        x1 = x[..., :self.split1]
        x2 = x[..., self.split1:]
        delta = self.alpha * self.net(x1)
        y = torch.cat([x1, x2 + delta], dim=-1)

        if self.swap:
            y = torch.cat([y[..., self.split2:], y[..., :self.split2]], dim=-1)
        return y

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        if y.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {y.shape[-1]}")

        if self.swap:
            y = torch.cat([y[..., self.split1:], y[..., :self.split1]], dim=-1)

        y1 = y[..., :self.split1]
        y2 = y[..., self.split1:]
        delta = self.alpha * self.net(y1)
        x = torch.cat([y1, y2 - delta], dim=-1)

        if self.swap:
            x = torch.cat([x[..., self.split2:], x[..., :self.split2]], dim=-1)
        return x

class InvertibleTabularAdapter(nn.Module):
    """
    Stack of additive coupling blocks for tabular encoders.

    Intended use:
      1. project input to emb_dim with a standard Linear
      2. pass through this adapter
      3. feed the result to the multimodal scorer

    Because each block is invertible and initialized near identity, this gives
    a more conditioning-friendly alternative to a free MLP.
    """
    def __init__(
        self,
        dim: int,
        num_blocks: int = 2,
        hidden_dim: int = None,
        dropout: float = 0.0,
        alpha_init: float = 1e-2,
    ):
        super().__init__()
        self.dim = int(dim)
        self.blocks = nn.ModuleList([
            AdditiveCouplingBlock(
                dim=self.dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                alpha_init=alpha_init,
                swap=bool(i % 2),
            )
            for i in range(int(num_blocks))
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        for block in reversed(self.blocks):
            y = block.inverse(y)
        return y


class NearIsometricLinear(nn.Module):
    """
    Linear layer with an orthogonal / semi-orthogonal weight matrix and a
    learnable scalar gain initialized near 1.

    This is a lightweight way to preserve geometry better than a free Linear
    layer. For square layers it is orthogonal; for rectangular layers it is
    semi-orthogonal in the sense supported by PyTorch's orthogonal
    parametrization.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gain_init: float = 1.0,
        learnable_gain: bool = True,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)

        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        nn.utils.parametrizations.orthogonal(self.linear, "weight")

        with torch.no_grad():
            if self.linear.bias is not None:
                self.linear.bias.zero_()

        if learnable_gain:
            self.log_gain = nn.Parameter(torch.tensor(float(gain_init)).log())
        else:
            self.log_gain = None
            self.register_buffer("_gain_const", torch.tensor(float(gain_init)))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.linear.bias is not None}, "
            f"learnable_gain={self.log_gain is not None}"
        )

    def _gain(self) -> torch.Tensor:
        if self.log_gain is not None:
            return self.log_gain.exp()
        return self._gain_const

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._gain() * self.linear.weight
        return F.linear(x, weight, self.linear.bias)


class BoundedOrthogonalLinear(nn.Module):
    """
    Orthogonal / semi-orthogonal mixing layer with bounded per-dimension gains.

    The effective map is:
        y = W (diag(g) x) + b

    where:
      - W is orthogonal (or semi-orthogonal for rectangular shapes),
      - each gain g_i is constrained to [gain_min, gain_max].

    This is intended as a more explicitly bi-Lipschitz alternative to a free
    Linear layer: it limits both expansion and collapse coordinate-wise before
    the orthogonal mixing.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gain_min: float = 0.5,
        gain_max: float = 2.0,
        gain_init: float = 1.0,
    ):
        super().__init__()
        if gain_min <= 0.0:
            raise ValueError("gain_min must be positive.")
        if gain_max <= gain_min:
            raise ValueError("gain_max must be larger than gain_min.")
        if not (gain_min <= gain_init <= gain_max):
            raise ValueError("gain_init must lie in [gain_min, gain_max].")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.gain_min = float(gain_min)
        self.gain_max = float(gain_max)

        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        nn.utils.parametrizations.orthogonal(self.linear, "weight")

        with torch.no_grad():
            if self.linear.bias is not None:
                self.linear.bias.zero_()

        # Reparameterize gains to stay in [gain_min, gain_max].
        frac = (float(gain_init) - self.gain_min) / (self.gain_max - self.gain_min)
        frac = min(max(frac, 1e-6), 1.0 - 1e-6)
        init_logit = math.log(frac / (1.0 - frac))
        self.gain_logits = nn.Parameter(torch.full((self.in_features,), init_logit))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.linear.bias is not None}, "
            f"gain_min={self.gain_min}, gain_max={self.gain_max}"
        )

    def gains(self) -> torch.Tensor:
        sigma = torch.sigmoid(self.gain_logits)
        return self.gain_min + (self.gain_max - self.gain_min) * sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gains().to(dtype=x.dtype, device=x.device)
        x_scaled = x * g
        return self.linear(x_scaled)


class BoundedSVDLinear(nn.Module):
    """
    Linear layer parameterized as

        W = U diag(s) V^T

    with orthogonal U/V factors and singular values constrained to a range
    [sv_min, sv_max].

    This lets us test the conditioning hypothesis directly:
      - sv_max controls expansion
      - sv_min controls collapse

    The layer can be used as a drop-in replacement for nn.Linear inside an MLP.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        sv_min: float = 0.01,  # 1.0,
        sv_max: float = 10.0,  # 2.0,
        learnable_singular_values: bool = True,
    ):
        super().__init__()
        if sv_min <= 0.0:
            raise ValueError("sv_min must be positive.")
        if sv_max <= sv_min:
            raise ValueError("sv_max must be larger than sv_min.")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = min(self.in_features, self.out_features)
        self.sv_min = float(sv_min)
        self.sv_max = float(sv_max)
        self.learnable_singular_values = bool(learnable_singular_values)

        # Full orthogonal matrices; we use only the first `rank` columns.
        self.U = nn.Linear(self.out_features, self.out_features, bias=False)
        self.V = nn.Linear(self.in_features, self.in_features, bias=False)
        nn.utils.parametrizations.orthogonal(self.U, "weight")
        nn.utils.parametrizations.orthogonal(self.V, "weight")

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)

        # tensor sv_init 
        sv_init = (self.sv_min + self.sv_max) / 2
        # vector sv_init, e.g., emb_dim/2 values 0.005 and emb_dim/2 values 3.0
        #sv_init = [0.5] * (self.rank // 2) + [1.2] * (self.rank // 2)
        init_s = self._normalize_sv_init(sv_init)
        if self.learnable_singular_values:
            init_logits = self._sv_to_logits(init_s)
            self.sv_logits = nn.Parameter(init_logits)
        else:
            self.register_buffer("sv_fixed", init_s)
            self.register_parameter("sv_logits", None)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.bias is not None}, "
            f"sv_min={self.sv_min}, sv_max={self.sv_max}, "
            f"learnable_singular_values={self.learnable_singular_values}"
        )

    def _normalize_sv_init(self, sv_init: float) -> torch.Tensor:
        # Tensor input
        if torch.is_tensor(sv_init):
            init_s = sv_init.detach().clone().float().reshape(-1)
            if init_s.numel() == 1:
                init_s = init_s.expand(self.rank)
            elif init_s.numel() != self.rank:
                raise ValueError(
                    f"sv_init tensor must have 1 or {self.rank} elements, got {init_s.numel()}."
                )
        # Python list/tuple input (vector feature)
        elif isinstance(sv_init, (list, tuple)):
            init_s = torch.tensor(sv_init, dtype=torch.float32).reshape(-1)
        else:
            init_s = torch.full((self.rank,), float(sv_init), dtype=torch.float32)

        init_s = init_s.clamp(min=self.sv_min, max=self.sv_max)
        return init_s

    def _sv_to_logits(self, s: torch.Tensor) -> torch.Tensor:
        frac = (s - self.sv_min) / (self.sv_max - self.sv_min)
        frac = frac.clamp(min=1e-6, max=1.0 - 1e-6)
        return torch.log(frac / (1.0 - frac))

    def singular_values(self) -> torch.Tensor:
        if self.learnable_singular_values:
            sigma = torch.sigmoid(self.sv_logits)
            return self.sv_min + (self.sv_max - self.sv_min) * sigma
        return self.sv_fixed

    def weight_matrix(self) -> torch.Tensor:
        # Both .weight.T tensors are orthogonal square matrices.
        Umat = self.U.weight.T[:, : self.rank]  # (out, rank)
        Vmat = self.V.weight.T[:, : self.rank]  # (in, rank)
        s = self.singular_values().to(dtype=Umat.dtype, device=Umat.device)  # (rank,)
        return (Umat * s.unsqueeze(0)) @ Vmat.T  # (out, in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight_matrix(), self.bias)


class ProductPreservingActivation(nn.Module):
    """
    A near-identity activation designed to perturb coordinates gently instead of
    replacing them aggressively.

    Modes:
      - "additive":      f(x) = x + alpha * tanh(beta * x)
      - "multiplicative": f(x) = x * (1 + alpha * tanh(beta * x))

    The multiplicative mode is the more default for
    learned multimodal scoring.
    """
    def __init__(
        self,
        alpha_init: float = 0.05,
        beta_init: float = 1.0,
        learnable_alpha: bool = True,
        learnable_beta: bool = True,
        mode: str = "multiplicative",
    ):
        super().__init__()
        if mode not in {"additive", "multiplicative"}:
            raise ValueError(f"Unknown mode: {mode}. Expected 'additive' or 'multiplicative'.")

        self.mode = str(mode)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init))) if learnable_alpha else None
        self.beta = nn.Parameter(torch.tensor(float(beta_init))) if learnable_beta else None

        if not learnable_alpha:
            self.register_buffer("_alpha_const", torch.tensor(float(alpha_init)))
        if not learnable_beta:
            self.register_buffer("_beta_const", torch.tensor(float(beta_init)))

    def extra_repr(self) -> str:
        return f"mode={self.mode!r}, learnable_alpha={self.alpha is not None}, learnable_beta={self.beta is not None}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha if self.alpha is not None else self._alpha_const
        beta = self.beta if self.beta is not None else self._beta_const

        perturb = torch.tanh(beta * x)
        if self.mode == "additive":
            return x + alpha * perturb
        return x * (1.0 + alpha * perturb)


class ExponentialModulationActivation(nn.Module):
    """
    A smooth near-identity multiplicative activation:

        f(x) = x * exp(alpha * tanh(beta * x))

    This preserves sign, keeps every coordinate alive, and modulates each
    feature by a bounded smooth gain. For alpha=0 it is exactly the identity.
    """
    def __init__(
        self,
        alpha_init: float = 0.05,
        beta_init: float = 1.0,
        learnable_alpha: bool = True,
        learnable_beta: bool = True,
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init))) if learnable_alpha else None
        self.beta = nn.Parameter(torch.tensor(float(beta_init))) if learnable_beta else None

        if not learnable_alpha:
            self.register_buffer("_alpha_const", torch.tensor(float(alpha_init)))
        if not learnable_beta:
            self.register_buffer("_beta_const", torch.tensor(float(beta_init)))

    def extra_repr(self) -> str:
        return f"learnable_alpha={self.alpha is not None}, learnable_beta={self.beta is not None}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha if self.alpha is not None else self._alpha_const
        beta = self.beta if self.beta is not None else self._beta_const
        gain = torch.exp(alpha * torch.tanh(beta * x))
        return x * gain


class SigmoidSelfAttention(nn.Module):
    """
    Multi-head self-attention with sigmoid attention gates instead of softmax.

    This intentionally mirrors the tensor layout and projection structure of
    standard multi-head attention as closely as possible; the key difference is
    that attention scores are passed through sigmoid rather than softmax.
    """
    def __init__(self, d_model, nhead, dropout=0.0, batch_first=True, bias=True):
        super().__init__()

        self.batch_first = bool(batch_first)

        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if not self.batch_first:
            x = x.transpose(0, 1)

        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = self._apply_masks(
            scores,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            seq_len=T,
        )
        gates = torch.sigmoid(scores)
        gates = self.dropout(gates)

        out = torch.matmul(gates, v)  # [B, H, T, Hd]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.out_proj(out)
        if not self.batch_first:
            out = out.transpose(0, 1)
        return out

    def _apply_masks(
        self,
        scores: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        is_causal: bool = False,
        seq_len: int = None,
    ) -> torch.Tensor:
        if is_causal:
            causal_mask = torch.ones(
                seq_len,
                seq_len,
                device=scores.device,
                dtype=torch.bool,
            ).triu(1)
            scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        if attn_mask is not None:
            attn_mask = attn_mask.to(device=scores.device)
            if attn_mask.dtype == torch.bool:
                if attn_mask.ndim == 2:
                    scores = scores.masked_fill(attn_mask[None, None, :, :], float("-inf"))
                elif attn_mask.ndim == 3:
                    scores = scores.masked_fill(
                        attn_mask.view(-1, self.nhead, seq_len, seq_len),
                        float("-inf"),
                    )
                else:
                    raise ValueError(f"Unsupported attn_mask shape: {tuple(attn_mask.shape)}")
            else:
                if attn_mask.ndim == 2:
                    scores = scores + attn_mask[None, None, :, :]
                elif attn_mask.ndim == 3:
                    scores = scores + attn_mask.view(-1, self.nhead, seq_len, seq_len)
                else:
                    raise ValueError(f"Unsupported attn_mask shape: {tuple(attn_mask.shape)}")

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        return scores

class SigmoidTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        batch_first=True,
        norm_first=False,
        bias=True,
    ):
        super().__init__()
        self.norm_first = bool(norm_first)
        self.batch_first = bool(batch_first)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = SigmoidSelfAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def _sa_block(self, x, attn_mask=None, key_padding_mask=None, is_causal=False):
        x = self.self_attn(
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )
        return self.dropout1(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                is_causal=is_causal,
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(
                    x,
                    src_mask,
                    src_key_padding_mask,
                    is_causal=is_causal,
                )
            )
            x = self.norm2(x + self._ff_block(x))
        return x

class PatchEncoder_CXR(nn.Module):
    def __init__(self, image_size=320, patch_size=64, in_channels=3, emb_dim=256, num_tokens=None):
        super().__init__()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.in_channels = int(in_channels)
        self.emb_dim = int(emb_dim)
        self.num_tokens = None if num_tokens is None else int(num_tokens)

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by patch_size ({self.patch_size})"
            )

        self.patch_dim = self.in_channels * self.patch_size * self.patch_size
        self.proj = nn.Linear(self.patch_dim, self.emb_dim)

    def forward(self, x):
        # x: [B, 3, 320, 320]
        B, C, H, W = x.shape
        if C != self.in_channels or H != self.image_size or W != self.image_size:
            raise ValueError(
                f"Expected input of shape [B,{self.in_channels},{self.image_size},{self.image_size}], got {tuple(x.shape)}"
            )

        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)          # [B, C, H/p, W/p, p, p]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()   # [B, H/p, W/p, C, p, p]
        x = x.view(B, -1, self.patch_dim)              # [B, T, C*p*p]
        x = self.proj(x)                               # [B, T, D]
        if self.num_tokens is not None:
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.num_tokens).transpose(1, 2)
        return x

class PatchEncoder_ECG(nn.Module):
    def __init__(
        self,
        input_size=(5000, 12),
        patch_size=(250, 12),
        in_channels=1,
        emb_dim=256,
        num_tokens=None,
    ):
        super().__init__()
        self.input_size = tuple(input_size)
        self.patch_size = tuple(patch_size)
        self.in_channels = int(in_channels)
        self.emb_dim = int(emb_dim)
        self.num_tokens = None if num_tokens is None else int(num_tokens)

        if len(self.input_size) != 2 or len(self.patch_size) != 2:
            raise ValueError("input_size and patch_size must be 2D tuples")
        if self.input_size[0] % self.patch_size[0] != 0 or self.input_size[1] % self.patch_size[1] != 0:
            raise ValueError(
                f"input_size {self.input_size} must be divisible by patch_size {self.patch_size}"
            )

        self.patch_dim = self.in_channels * self.patch_size[0] * self.patch_size[1]
        self.proj = nn.Linear(self.patch_dim, self.emb_dim)

    def forward(self, x):
        # x: [B, 1, 5000, 12]
        B, C, H, W = x.shape
        if C != self.in_channels or (H, W) != self.input_size:
            raise ValueError(
                f"Expected input of shape [B,{self.in_channels},{self.input_size[0]},{self.input_size[1]}], got {tuple(x.shape)}"
            )

        ph, pw = self.patch_size
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)      # [B, C, H/ph, W/pw, ph, pw]
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()   # [B, H/ph, W/pw, C, ph, pw]
        x = x.view(B, -1, self.patch_dim)              # [B, T, C*ph*pw]
        x = self.proj(x)                               # [B, T, D]
        if self.num_tokens is not None:
            x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.num_tokens).transpose(1, 2)
        return x


def mimetic_init_svd_(
    module: nn.Module,
    alpha1: float = 0.7,
    beta1: float = 0.7,
    alpha2: float = 0.4,
    beta2: float = 0.4
) -> None:
    """
    Applies mimetic initialization using SVD factorization.

    [1] A. Trockman and J. Z. Kolter, “Mimetic initialization of self-attention layers” 
    """
    if isinstance(module, (nn.MultiheadAttention)):
        print("applying mimetic_init_svd_ to multi head attention")
        embed_dim = module.embed_dim
        device = module.in_proj_weight.device
        dtype = module.in_proj_weight.dtype
        num_heads = module.num_heads
        head_dim = embed_dim // num_heads

        with torch.no_grad():
            eye = torch.eye(embed_dim, device=device, dtype=dtype)
            Z1 = torch.randn(embed_dim, embed_dim, device=device, dtype=dtype) * (1/embed_dim)

            # W_Q and W_K source calculation (embed_dim, embed_dim)
            W_Q_W_KT = (alpha1 * Z1) - (beta1 * eye)
            U_1, S_1, V_1T = torch.linalg.svd(W_Q_W_KT, full_matrices=True) # U_1, V_1T are embed_dim x embed_dim
            S_1 = torch.diag(torch.sqrt(S_1))

            # Construct W_V and W_proj from SVD of W_Q_W_KT
            W_V = U_1 @ S_1
            W_proj = V_1T @ (S_1**0.5)

            # Process each head separately for Q and K with a new Z2
            W_Q = torch.zeros(embed_dim, embed_dim, device=device, dtype=dtype)
            W_K = torch.zeros(embed_dim, embed_dim, device=device, dtype=dtype)
            for h in range(num_heads):
                Z2 = torch.randn(embed_dim, embed_dim, device=device, dtype=dtype) * (1/embed_dim)
                W_V_W_Tproj = (alpha2 * Z2) + (beta2 * eye)
                U_2, S_2, V_2T = torch.linalg.svd(W_V_W_Tproj, full_matrices=False)
                S_2 = torch.diag(torch.sqrt(S_2))

                head_W_Q = U_2[:, :head_dim] @ (S_2[:head_dim, :head_dim]**0.5) # (d, k) @ (k, k) -> (d, k)
                head_W_K = V_2T.T[:, :head_dim] @ (S_2[:head_dim, :head_dim]**0.5) # (d, k) @ (k, k) -> (d, k)

                # Assign to the appropriate slice of the final weight matrices
                start_idx = h * head_dim
                end_idx = (h + 1) * head_dim
                W_Q[:, start_idx:end_idx] = head_W_Q
                W_K[:, start_idx:end_idx] = head_W_K

            # Assign concatenated/calculated module weights
            module.in_proj_weight.data[:embed_dim] = W_Q
            module.in_proj_weight.data[embed_dim:2*embed_dim] = W_K
            module.in_proj_weight.data[2*embed_dim:] = W_V
            if isinstance(module.out_proj, nn.Linear):
                module.out_proj.weight.data = W_proj
            else: 
                module.out_proj.data = W_proj

            # Zero Bias Initialization
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)

            print("mimetic_init_svd_ applied to multi head attention")
