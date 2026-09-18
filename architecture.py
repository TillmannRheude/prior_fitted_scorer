import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager

from utils import sinusoidal_positional_encoding


def mimetic_init_svd_(
    module: nn.Module,
    alpha1: float = 0.7,
    beta1: float = 0.7,
    alpha2: float = 0.4,
    beta2: float = 0.4,
) -> None:
    """Apply SVD-based mimetic initialization to multi-head attention.

    Based on Trockman and Kolter, "Mimetic Initialization of
    Self-Attention Layers".
    """
    if not isinstance(module, nn.MultiheadAttention):
        return

    embed_dim = module.embed_dim
    device = module.in_proj_weight.device
    dtype = module.in_proj_weight.dtype
    num_heads = module.num_heads
    head_dim = embed_dim // num_heads

    with torch.no_grad():
        eye = torch.eye(embed_dim, device=device, dtype=dtype)
        z1 = (
            torch.randn(embed_dim, embed_dim, device=device, dtype=dtype)
            / embed_dim
        )

        w_q_w_kt = (alpha1 * z1) - (beta1 * eye)
        u_1, s_1, v_1_t = torch.linalg.svd(
            w_q_w_kt,
            full_matrices=True,
        )
        s_1 = torch.diag(torch.sqrt(s_1))

        w_v = u_1 @ s_1
        w_proj = v_1_t @ (s_1**0.5)

        w_q = torch.zeros(
            embed_dim,
            embed_dim,
            device=device,
            dtype=dtype,
        )
        w_k = torch.zeros_like(w_q)
        for head in range(num_heads):
            z2 = (
                torch.randn(
                    embed_dim,
                    embed_dim,
                    device=device,
                    dtype=dtype,
                )
                / embed_dim
            )
            w_v_w_tproj = (alpha2 * z2) + (beta2 * eye)
            u_2, s_2, v_2_t = torch.linalg.svd(
                w_v_w_tproj,
                full_matrices=False,
            )
            s_2 = torch.diag(torch.sqrt(s_2))

            head_w_q = (
                u_2[:, :head_dim]
                @ (s_2[:head_dim, :head_dim] ** 0.5)
            )
            head_w_k = (
                v_2_t.T[:, :head_dim]
                @ (s_2[:head_dim, :head_dim] ** 0.5)
            )

            start = head * head_dim
            end = (head + 1) * head_dim
            w_q[:, start:end] = head_w_q
            w_k[:, start:end] = head_w_k

        module.in_proj_weight[:embed_dim].copy_(w_q)
        module.in_proj_weight[embed_dim : 2 * embed_dim].copy_(w_k)
        module.in_proj_weight[2 * embed_dim :].copy_(w_v)
        module.out_proj.weight.copy_(w_proj)

        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
        if module.out_proj.bias is not None:
            nn.init.zeros_(module.out_proj.bias)


class FixedPatternMultiheadAttention(nn.MultiheadAttention):
    """Multi-head value projection with a fixed, content-independent pattern."""

    VALID_PATTERNS = {"identity", "uniform", "fixed_cross_modal"}

    def __init__(
        self,
        *args,
        attention_pattern: str,
        token_groups: list[int],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        pattern = str(attention_pattern).lower()
        if pattern not in self.VALID_PATTERNS:
            raise ValueError(
                f"Unsupported fixed attention pattern {pattern!r}; "
                f"expected one of {sorted(self.VALID_PATTERNS)}."
            )
        self.attention_pattern = pattern
        self.token_groups = tuple(int(group) for group in token_groups)

    def _base_allowed_mask(
        self,
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if query_length != len(self.token_groups):
            raise ValueError(
                f"Fixed attention expected {len(self.token_groups)} tokens, "
                f"got query length {query_length}."
            )
        if key_length != len(self.token_groups):
            raise ValueError(
                f"Fixed attention expected {len(self.token_groups)} tokens, "
                f"got key length {key_length}."
            )

        if self.attention_pattern == "identity":
            return torch.eye(
                query_length,
                key_length,
                dtype=torch.bool,
                device=device,
            )
        if self.attention_pattern == "uniform":
            return torch.ones(
                query_length,
                key_length,
                dtype=torch.bool,
                device=device,
            )

        groups = torch.tensor(self.token_groups, device=device)
        query_groups = groups[:, None]
        key_groups = groups[None, :]
        cls_edges = (query_groups == -1) | (key_groups == -1)
        register_edges = (query_groups == -2) | (key_groups == -2)
        cross_modality_edges = (
            (query_groups >= 0)
            & (key_groups >= 0)
            & (query_groups != key_groups)
        )
        return cls_edges | register_edges | cross_modality_edges

    @staticmethod
    def _blocked_from_mask(
        mask: torch.Tensor,
        *,
        batch_size: int,
        num_heads: int,
        query_length: int,
        key_length: int,
    ) -> torch.Tensor:
        if mask.dtype == torch.bool:
            blocked = mask
        else:
            blocked = torch.isneginf(mask) | (mask < -1e4)

        if blocked.ndim == 2:
            blocked = blocked[None, None, :, :]
        elif blocked.ndim == 3:
            if blocked.shape[0] == batch_size * num_heads:
                blocked = blocked.reshape(
                    batch_size,
                    num_heads,
                    query_length,
                    key_length,
                )
            elif blocked.shape[0] == batch_size:
                blocked = blocked[:, None, :, :]
            else:
                raise ValueError(
                    f"Unsupported fixed-attention mask shape {tuple(mask.shape)}."
                )
        elif blocked.ndim != 4:
            raise ValueError(
                f"Unsupported fixed-attention mask shape {tuple(mask.shape)}."
            )
        return blocked

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        if self.batch_first:
            query_bf, key_bf, value_bf = query, key, value
        else:
            query_bf = query.transpose(0, 1)
            key_bf = key.transpose(0, 1)
            value_bf = value.transpose(0, 1)

        batch_size, query_length, _ = query_bf.shape
        key_length = key_bf.shape[1]
        if key_bf.shape != value_bf.shape:
            raise ValueError("Fixed attention requires matching key/value shapes.")

        value_weight = self.in_proj_weight[2 * self.embed_dim :]
        value_bias = (
            None
            if self.in_proj_bias is None
            else self.in_proj_bias[2 * self.embed_dim :]
        )
        projected_value = F.linear(value_bf, value_weight, value_bias)
        projected_value = (
            projected_value.reshape(
                batch_size,
                key_length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        allowed = self._base_allowed_mask(
            query_length,
            key_length,
            query_bf.device,
        )
        allowed = allowed[None, None, :, :].expand(
            batch_size,
            self.num_heads,
            -1,
            -1,
        )

        if attn_mask is not None:
            blocked = self._blocked_from_mask(
                attn_mask,
                batch_size=batch_size,
                num_heads=self.num_heads,
                query_length=query_length,
                key_length=key_length,
            )
            allowed = allowed & ~blocked

        if is_causal:
            causal_blocked = torch.ones(
                query_length,
                key_length,
                dtype=torch.bool,
                device=query_bf.device,
            ).triu(diagonal=1)
            allowed = allowed & ~causal_blocked[None, None, :, :]

        if key_padding_mask is not None:
            if key_padding_mask.dtype == torch.bool:
                key_blocked = key_padding_mask
            else:
                key_blocked = torch.isneginf(key_padding_mask) | (
                    key_padding_mask < -1e4
                )
            key_blocked = key_blocked.to(device=query_bf.device)
            allowed = allowed & ~key_blocked[:, None, None, :]

        attention_weights = allowed.to(dtype=projected_value.dtype)
        denominator = attention_weights.sum(dim=-1, keepdim=True)
        attention_weights = attention_weights / denominator.clamp_min(1.0)
        attention_weights = F.dropout(
            attention_weights,
            p=self.dropout,
            training=self.training,
        )

        attention_output = attention_weights @ projected_value
        attention_output = (
            attention_output.transpose(1, 2)
            .reshape(batch_size, query_length, self.embed_dim)
        )
        attention_output = self.out_proj(attention_output)
        if not self.batch_first:
            attention_output = attention_output.transpose(0, 1)

        if not need_weights:
            return attention_output, None
        if average_attn_weights:
            returned_weights = attention_weights.mean(dim=1)
        else:
            returned_weights = attention_weights
        return attention_output, returned_weights


def _fixed_attention_fastpath_guard(module, inputs):
    """A no-op hook that keeps TransformerEncoderLayer on its Python path."""
    return None


@contextmanager
def _maybe_disable_flash_mem_efficient_sdp(disable: bool):
    if not disable or not hasattr(torch.backends, "cuda"):
        yield
        return

    cuda_backend = torch.backends.cuda
    required_attrs = (
        "flash_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "enable_flash_sdp",
        "enable_mem_efficient_sdp",
    )
    if not all(hasattr(cuda_backend, attr) for attr in required_attrs):
        yield
        return

    flash_was_enabled = cuda_backend.flash_sdp_enabled()
    mem_efficient_was_enabled = cuda_backend.mem_efficient_sdp_enabled()
    cuda_backend.enable_flash_sdp(False)
    cuda_backend.enable_mem_efficient_sdp(False)
    try:
        yield
    finally:
        cuda_backend.enable_flash_sdp(flash_was_enabled)
        cuda_backend.enable_mem_efficient_sdp(mem_efficient_was_enabled)

class Contrastive_Model(nn.Module):
    def __init__(
        self,
        encoders: nn.ModuleList = nn.ModuleList([]),
    ):
        super().__init__()
        self.encoders = encoders

    def forward(
        self, 
        x: list = [torch.Tensor, torch.Tensor, torch.Tensor],
        ):
        embeddings = [self.encoders[i](x[i]) for i in range(len(x))]
        return {
            "embeddings": embeddings
        }


class TransformerScorer(nn.Module):
    def __init__(
        self,
        transformer_params: dict = {
            "d_model": 256,
            "nhead": 2,
            "num_layers": 2,
            "dropout": 0.0,
        },
        proj_output_dim: int = 1,
        max_modalities: int = 3,
        seq_dims: list[int] = [1, 1, 1],
        num_register_tokens: int = 0,
        use_mask_tokens: bool = False,
        candidate_dependent: bool = True,
        input_dim: int = None,
    ):
        super().__init__()
        d_model = transformer_params["d_model"]
        use_sigmoid_attention = bool(transformer_params.get("use_sigmoid_attention", False))
        use_sinusoidal_pe = bool(transformer_params.get("use_sinusoidal_pe", False))
        self.initialization = str(
            transformer_params.get("initialization", "xavier")
        ).lower()
        valid_initializations = {"xavier", "kaiming", "mimetic"}
        if self.initialization not in valid_initializations:
            raise ValueError(
                f"Unsupported transformer initialization {self.initialization!r}; "
                f"expected one of {sorted(valid_initializations)}."
            )
        self.attention_replacement = str(
            transformer_params.get("attention_replacement", "standard")
        ).lower()
        valid_attention_replacements = {
            "standard",
            *FixedPatternMultiheadAttention.VALID_PATTERNS,
        }
        if self.attention_replacement not in valid_attention_replacements:
            raise ValueError(
                f"Unsupported attention replacement "
                f"{self.attention_replacement!r}; expected one of "
                f"{sorted(valid_attention_replacements)}."
            )
        if use_sigmoid_attention and self.attention_replacement != "standard":
            raise ValueError(
                "Fixed attention replacements cannot be combined with "
                "use_sigmoid_attention=True."
            )
        if use_sigmoid_attention and self.initialization == "mimetic":
            raise ValueError(
                "Mimetic initialization currently requires standard "
                "nn.MultiheadAttention."
            )
        self.freeze_attention = bool(
            transformer_params.get("freeze_attention", False)
        )
        self.freeze_ffn = bool(transformer_params.get("freeze_ffn", False))
        self.disable_flash_mem_efficient_sdp = bool(
            transformer_params.get("disable_flash_mem_efficient_sdp", False)
        )
        num_register_tokens = int(transformer_params.get("num_register_tokens", num_register_tokens))
        input_dim = d_model if input_dim is None else int(input_dim)
        self.max_modalities = int(max_modalities)
        self.seq_dims = list(seq_dims)
        self.num_register_tokens = int(num_register_tokens)
        self.use_mask_tokens = bool(use_mask_tokens)
        self.candidate_dependent = bool(candidate_dependent)
        self.input_dim = input_dim
        self.d_model = int(d_model)
        self.use_sinusoidal_pe = use_sinusoidal_pe

        self.input_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(self.max_modalities)
        ])

        if self.use_mask_tokens:
            self.mask_tokens = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, d_model)) for _ in range(self.max_modalities)
            ])
        else:
            self.mask_tokens = None
        
        # Learnable CLS token (prepended); score is a linear readout of CLS.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        if self.num_register_tokens > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, self.num_register_tokens, d_model))
        else:
            self.register_tokens = None

        if self.use_sinusoidal_pe:
            # [CLS] + modality tokens + optional registers
            total_tokens = 1 + int(sum(self.seq_dims)) + int(self.num_register_tokens)
            pe = sinusoidal_positional_encoding(seq_len=total_tokens, d_model=d_model)
            self.register_buffer("pos_encoding", pe, persistent=False)
        else:
            self.pos_encoding = None

        if not use_sigmoid_attention:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=transformer_params["nhead"],
                dim_feedforward=d_model,
                dropout=transformer_params["dropout"],
                batch_first=True,
                norm_first=True,
                activation="gelu"
            )
        else:
            from utils import SigmoidTransformerEncoderLayer
            encoder_layer = SigmoidTransformerEncoderLayer(
                d_model=d_model,
                nhead=transformer_params["nhead"],
                dim_feedforward=d_model,
                dropout=transformer_params["dropout"],
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
        if self.attention_replacement != "standard":
            token_groups = [-1]
            for modality, seq_dim in enumerate(self.seq_dims):
                token_groups.extend([modality] * int(seq_dim))
            token_groups.extend([-2] * self.num_register_tokens)
            encoder_layer.self_attn = FixedPatternMultiheadAttention(
                embed_dim=d_model,
                num_heads=transformer_params["nhead"],
                dropout=transformer_params["dropout"],
                batch_first=True,
                attention_pattern=self.attention_replacement,
                token_groups=token_groups,
            )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_params["num_layers"],
        )
        if self.attention_replacement != "standard":
            for layer in self.transformer.layers:
                layer.self_attn.register_forward_pre_hook(
                    _fixed_attention_fastpath_guard
                )

        self.proj_head = nn.Linear(d_model, proj_output_dim)

        self.apply(self._init_weights)
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                self._init_attention_projection(module)
        if self.initialization == "mimetic":
            mimetic_kwargs = {
                "alpha1": float(transformer_params.get("mimetic_alpha1", 0.7)),
                "beta1": float(transformer_params.get("mimetic_beta1", 0.7)),
                "alpha2": float(transformer_params.get("mimetic_alpha2", 0.4)),
                "beta2": float(transformer_params.get("mimetic_beta2", 0.4)),
            }
            for module in self.modules():
                mimetic_init_svd_(module, **mimetic_kwargs)

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        if self.num_register_tokens > 0:
            nn.init.normal_(self.register_tokens, mean=0.0, std=0.02)
        if self.mask_tokens is not None:
            for param in self.mask_tokens:
                nn.init.normal_(param, mean=0.0, std=0.02)
        self._apply_component_freezing()

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            if self.initialization in {"xavier", "mimetic"}:
                nn.init.xavier_uniform_(m.weight)
            else:
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_in",
                    nonlinearity="linear",
                )
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _init_attention_projection(
        self,
        module: nn.MultiheadAttention,
    ) -> None:
        if self.initialization in {"xavier", "mimetic"}:
            nn.init.xavier_uniform_(module.in_proj_weight)
        else:
            nn.init.kaiming_normal_(
                module.in_proj_weight,
                mode="fan_in",
                nonlinearity="linear",
            )
        if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)

    def _apply_component_freezing(self) -> None:
        for layer in self.transformer.layers:
            if self.freeze_attention:
                for parameter in layer.self_attn.parameters():
                    parameter.requires_grad_(False)
            if self.freeze_ffn:
                for module in (layer.linear1, layer.linear2):
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        source_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
        return_features: bool = False,
    ):
        embeddings = [
            embedding[:, None, :] if embedding.ndim == 2 else embedding for embedding in embeddings
        ]  # [B, 1, D]

        batch_size = embeddings[0].shape[0]
        if src_key_padding_mask is not None:
            expected_shape = (batch_size, self.max_modalities)
            if src_key_padding_mask.shape != expected_shape:
                raise ValueError(
                    "Expected modality-level src_key_padding_mask shape "
                    f"{expected_shape}, got {tuple(src_key_padding_mask.shape)}"
                )
            src_key_padding_mask = src_key_padding_mask.to(
                device=embeddings[0].device,
                dtype=torch.bool,
            )

        if source_mask is not None:
            if not self.use_mask_tokens or self.mask_tokens is None:
                raise ValueError("source_mask was provided, but use_mask_tokens is disabled for TransformerScorer.")
            if source_mask.shape != (embeddings[0].shape[0], self.max_modalities):
                raise ValueError(
                    f"Expected source_mask shape {(embeddings[0].shape[0], self.max_modalities)}, got {tuple(source_mask.shape)}"
                )
            source_mask = source_mask.to(device=embeddings[0].device, dtype=torch.bool)

        modality_tokens = []
        for m, embedding in enumerate(embeddings):
            if embedding.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected modality {m} embedding dim {self.input_dim}, got {embedding.shape[-1]}."
                )
            tok = self.input_projs[m](embedding)
            if source_mask is not None:
                missing = source_mask[:, m]
                if missing.any():
                    mask_tok = self.mask_tokens[m].expand(tok.shape[0], tok.shape[1], -1)
                    missing_view = missing[:, None, None]
                    tok = torch.where(missing_view, mask_tok, tok)
            modality_tokens.append(tok)

        modality = torch.cat(modality_tokens, dim=1)  # [B, sum(T_m), D]
        n_modality_tokens = modality.shape[1]
        cls = self.cls_token.expand(modality.shape[0], -1, -1)
        tokens = torch.cat([cls, modality], dim=1)  # [B, 1 + sum(T_m), D]

        token_padding_mask = None
        if src_key_padding_mask is not None:
            modality_padding = torch.cat(
                [
                    src_key_padding_mask[:, m : m + 1].expand(-1, embedding.shape[1])
                    for m, embedding in enumerate(embeddings)
                ],
                dim=1,
            )
            cls_padding = torch.zeros(
                (batch_size, 1),
                device=tokens.device,
                dtype=torch.bool,
            )
            token_padding_mask = torch.cat([cls_padding, modality_padding], dim=1)

        # Optional registers at the end (not used for scoring).
        if self.num_register_tokens > 0:
            reg = self.register_tokens.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([tokens, reg], dim=1)
            if token_padding_mask is not None:
                register_padding = torch.zeros(
                    (batch_size, self.num_register_tokens),
                    device=tokens.device,
                    dtype=torch.bool,
                )
                token_padding_mask = torch.cat(
                    [token_padding_mask, register_padding],
                    dim=1,
                )

        if self.pos_encoding is not None:
            if self.pos_encoding.shape[1] != tokens.shape[1]:
                raise ValueError(
                    f"Positional encoding length ({self.pos_encoding.shape[1]}) does not match token length ({tokens.shape[1]})."
                )
            tokens = tokens + self.pos_encoding.to(device=tokens.device, dtype=tokens.dtype)

        with _maybe_disable_flash_mem_efficient_sdp(self.disable_flash_mem_efficient_sdp):
            transformer_output = self.transformer(
                tokens,
                src_key_padding_mask=token_padding_mask,
            )  # [B, 1 + M (+ R), D]

        cls_out = transformer_output[:, 0, :]  # [B, D]
        modality_out = transformer_output[:, 1 : 1 + n_modality_tokens, :]  # [B, M, D]
        flat_features = torch.flatten(modality_out, start_dim=1)  # [B, M*D]
        z = self.proj_head(cls_out)
        if return_features:
            return {
                "z": z,
                "cls_token": cls_out,
                "transformer_flat": flat_features,
                "transformer_tokens": modality_out,
            }
        return z

class TransformerScorer_Model(nn.Module):
    def __init__(
        self,
        contrastive_model: Contrastive_Model = Contrastive_Model(),
        transformer_params: dict = {
            "d_model": 256,
            "nhead": 2,
            "num_layers": 2,
        },
        proj_output_dim: int = 1,
        seq_dims: list[int] = [1, 1, 1],
        candidate_dependent: bool = True,
    ):
        super().__init__()
        self.candidate_dependent = bool(candidate_dependent)
        encoders = getattr(contrastive_model, "encoders", None)
        encoder_output_dim = transformer_params["d_model"]
        if encoders is not None and len(encoders) > 0:
            encoder_output_dim = getattr(encoders[0], "emb_dim", encoder_output_dim)
            for encoder in encoders:
                current_dim = getattr(encoder, "emb_dim", encoder_output_dim)
                if int(current_dim) != int(encoder_output_dim):
                    raise ValueError(
                        "TransformerScorer expects all encoder output dimensions to match; "
                        f"got {encoder_output_dim} and {current_dim}."
                    )
        self.transformer = TransformerScorer(
            transformer_params=transformer_params,
            proj_output_dim=proj_output_dim,
            seq_dims=seq_dims,
            candidate_dependent=candidate_dependent,
            input_dim=encoder_output_dim,
        )
        self.contrastive_model = contrastive_model

    def forward(
        self,
        x: list = [torch.Tensor, torch.Tensor, torch.Tensor],
        source_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
    ):
        embeddings = self.contrastive_model(x)["embeddings"]
        transformer_output = self.transformer(
            embeddings,
            source_mask=source_mask,
            src_key_padding_mask=src_key_padding_mask,
            return_features=True,
        )
        return {
            "embeddings": embeddings,
            "z": transformer_output["z"],
            "cls_token": transformer_output["cls_token"],
            "transformer_flat": transformer_output["transformer_flat"],
            "transformer_tokens": transformer_output["transformer_tokens"],
        }


class MLPScorer(nn.Module):
    """Joint scalar scorer over concatenated modality embeddings."""

    def __init__(
        self,
        input_dim: int,
        mlp_params: dict,
        proj_output_dim: int = 1,
        max_modalities: int = 3,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.max_modalities = int(max_modalities)
        self.include_missingness = bool(
            mlp_params.get("include_missingness", True)
        )
        self.use_residual = bool(mlp_params.get("use_residual", False))

        configured_hidden_dims = mlp_params.get("hidden_dims")
        if configured_hidden_dims is None:
            hidden_dim = int(mlp_params.get("hidden_dim", 512))
            num_layers = int(mlp_params.get("num_layers", 2))
            hidden_dims = [hidden_dim for _ in range(num_layers)]
        else:
            hidden_dims = [
                int(hidden_dim) for hidden_dim in configured_hidden_dims
            ]
        if not hidden_dims:
            raise ValueError("MLPScorer requires at least one hidden dimension.")
        if any(hidden_dim < 1 for hidden_dim in hidden_dims):
            raise ValueError(
                f"MLPScorer hidden dimensions must be positive, got {hidden_dims}."
            )
        if self.use_residual and len(set(hidden_dims)) != 1:
            raise ValueError(
                "MLPScorer residual connections require identical hidden "
                f"dimensions in every layer, got {hidden_dims}."
            )
        if self.use_residual and len(hidden_dims) < 2:
            raise ValueError(
                "MLPScorer residual connections require at least two "
                "hidden layers."
            )

        dropout = float(mlp_params.get("dropout", 0.0))
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"MLPScorer dropout must be in [0, 1), got {dropout}.")

        activation_name = str(mlp_params.get("activation", "gelu")).lower()
        activations = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
        }
        if activation_name not in activations:
            raise ValueError(
                f"Unsupported MLPScorer activation {activation_name!r}; "
                f"expected one of {sorted(activations)}."
            )

        scorer_input_dim = self.max_modalities * self.input_dim
        if self.include_missingness:
            scorer_input_dim += self.max_modalities

        layers = []
        self._hidden_block_ranges = []
        current_dim = scorer_input_dim
        for hidden_dim in hidden_dims:
            block_start = len(layers)
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    activations[activation_name](),
                ]
            )
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            self._hidden_block_ranges.append((block_start, len(layers)))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, int(proj_output_dim)))

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        embeddings: list[torch.Tensor],
        source_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
        return_features: bool = False,
    ):
        if len(embeddings) != self.max_modalities:
            raise ValueError(
                f"MLPScorer expected {self.max_modalities} modalities, "
                f"got {len(embeddings)}."
            )
        if source_mask is not None:
            if src_key_padding_mask is not None:
                raise ValueError(
                    "Provide only one of source_mask or src_key_padding_mask "
                    "to MLPScorer."
                )
            src_key_padding_mask = source_mask

        pooled_embeddings = []
        batch_size = embeddings[0].shape[0]
        for modality, embedding in enumerate(embeddings):
            if embedding.ndim == 2:
                pooled = embedding
            elif embedding.ndim == 3:
                pooled = embedding.mean(dim=1)
            else:
                raise ValueError(
                    f"Expected modality {modality} embedding shape (B,D) or "
                    f"(B,T,D), got {tuple(embedding.shape)}."
                )
            if pooled.shape[0] != batch_size:
                raise ValueError("All MLPScorer modalities must share a batch size.")
            if pooled.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Expected modality {modality} embedding dim {self.input_dim}, "
                    f"got {pooled.shape[-1]}."
                )
            pooled_embeddings.append(pooled)

        if src_key_padding_mask is None:
            missing = torch.zeros(
                (batch_size, self.max_modalities),
                dtype=torch.bool,
                device=pooled_embeddings[0].device,
            )
        else:
            expected_shape = (batch_size, self.max_modalities)
            if tuple(src_key_padding_mask.shape) != expected_shape:
                raise ValueError(
                    f"Expected modality mask shape {expected_shape}, "
                    f"got {tuple(src_key_padding_mask.shape)}."
                )
            missing = src_key_padding_mask.to(
                device=pooled_embeddings[0].device,
                dtype=torch.bool,
            )

        pooled_embeddings = [
            embedding.masked_fill(missing[:, modality : modality + 1], 0.0)
            for modality, embedding in enumerate(pooled_embeddings)
        ]
        scorer_features = torch.cat(pooled_embeddings, dim=-1)
        if self.include_missingness:
            present = (~missing).to(dtype=scorer_features.dtype)
            scorer_features = torch.cat([scorer_features, present], dim=-1)

        if not self.use_residual:
            z = self.mlp(scorer_features)
        else:
            hidden = scorer_features
            for block_index, (start, end) in enumerate(
                self._hidden_block_ranges
            ):
                residual = hidden
                for module in self.mlp[start:end]:
                    hidden = module(hidden)
                if block_index > 0:
                    hidden = hidden + residual
            z = self.mlp[-1](hidden)
        if return_features:
            return {"z": z, "mlp_features": scorer_features}
        return z


class MLPScorer_Model(nn.Module):
    def __init__(
        self,
        contrastive_model: Contrastive_Model,
        mlp_params: dict,
        proj_output_dim: int = 1,
        max_modalities: int = 3,
    ):
        super().__init__()
        encoders = getattr(contrastive_model, "encoders", None)
        if encoders is None or len(encoders) == 0:
            raise ValueError("MLPScorer_Model requires a non-empty encoder stack.")

        encoder_output_dim = int(getattr(encoders[0], "emb_dim"))
        for encoder in encoders[1:]:
            current_dim = int(getattr(encoder, "emb_dim"))
            if current_dim != encoder_output_dim:
                raise ValueError(
                    "MLPScorer expects all encoder output dimensions to match; "
                    f"got {encoder_output_dim} and {current_dim}."
                )

        self.scorer = MLPScorer(
            input_dim=encoder_output_dim,
            mlp_params=mlp_params,
            proj_output_dim=proj_output_dim,
            max_modalities=max_modalities,
        )
        self.contrastive_model = contrastive_model

    @property
    def transformer(self) -> MLPScorer:
        """Compatibility alias for the shared learned-scorer objective."""
        return self.scorer

    def forward(
        self,
        x: list[torch.Tensor],
        source_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
    ):
        embeddings = self.contrastive_model(x)["embeddings"]
        scorer_output = self.scorer(
            embeddings,
            source_mask=source_mask,
            src_key_padding_mask=src_key_padding_mask,
            return_features=True,
        )
        return {
            "embeddings": embeddings,
            "z": scorer_output["z"],
            "mlp_features": scorer_output["mlp_features"],
        }
