import torch
import torch.nn as nn

from torchvision import models
from transformers import AutoModel

from helpers import (
    _disable_resnet_skip_connections_,
    _init_linear_near_identity_,
)


def _init_encoder_weights(m) -> None:
    if isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class IdentityEncoder(nn.Module):
    """
    Pass precomputed modality embeddings directly into the contrastive/scorer
    stack. Used for SCM scorer-only pretraining.
    """
    def __init__(self, emb_dim: int):
        super().__init__()
        self.emb_dim = int(emb_dim)

    def forward(self, x):
        if x.shape[-1] != self.emb_dim:
            raise ValueError(
                f"IdentityEncoder expected last dim {self.emb_dim}, got {x.shape[-1]}."
            )
        return x.float()


class ResidualAdapterEncoder(nn.Module):
    """
    Near-identity modality adapter for SCM pretraining.

    This preserves the raw SCM embedding path and learns a small residual
    correction before the contrastive objective/scorer.
    """
    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        residual_scale_init: float = 1e-3,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.input_norm = nn.LayerNorm(self.emb_dim) if bool(use_layernorm) else nn.Identity()
        self.adapter = nn.Sequential(
            nn.Linear(self.emb_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.emb_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))

        self.apply(self._init_weights)
        self.init_near_identity_()

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_near_identity_(self) -> None:
        linears = [m for m in self.adapter if isinstance(m, nn.Linear)]
        if not linears:
            return
        with torch.no_grad():
            last = linears[-1]
            last.weight.zero_()
            if last.bias is not None:
                last.bias.zero_()

    def forward(self, x):
        if x.shape[-1] != self.emb_dim:
            raise ValueError(
                f"ResidualAdapterEncoder expected last dim {self.emb_dim}, got {x.shape[-1]}."
            )
        x = x.float()
        residual = self.adapter(self.input_norm(x))
        return x + self.residual_scale * residual


"""
Symile-MIMIC 
"""
class CXREncoder(nn.Module):
    def __init__(
        self,
        resnet_params: dict = {
            "type": "resnet",
            "weights": None,  # "IMAGENET1K_V2"
            "norm_type": "batchnorm",
        },
        emb_dim: int = 8192,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        encoder_type = str(resnet_params.get("type", "resnet")).lower()

        if encoder_type == "resnet":
            self.encoder_type = "resnet"
            self.resnet = models.resnet50(weights=resnet_params.get("weights", None))
            if bool(resnet_params.get("disable_skip_connections", False)):
                _disable_resnet_skip_connections_(self.resnet)
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.emb_dim, bias=True)

            nn.init.kaiming_normal_(self.resnet.fc.weight, mode="fan_out")
            nn.init.zeros_(self.resnet.fc.bias)
        elif encoder_type in {"transformer", "vit"}:
            self.encoder_type = "transformer"
            vit_name = str(resnet_params.get("vit", resnet_params.get("name", "vit_b_32")))
            dropout = float(resnet_params.get("dropout", resnet_params.get("vit_dropout", 0.2)))
            image_size = int(resnet_params.get("image_size", 224))
            weights = resnet_params.get("weights", None)
            in_channels = int(resnet_params.get("in_channels", 3))

            self.vit = getattr(models, vit_name)(
                weights=weights,
                dropout=dropout,
                image_size=image_size,
                num_classes=1000,
            )
            vit_hidden_dim = self.vit.heads.head.in_features
            self.vit.heads.head = nn.Linear(vit_hidden_dim, self.emb_dim)
            self.vit.heads.head.apply(_init_encoder_weights)
            if in_channels != 3:
                patch_size = int(vit_name.split("_")[-1])
                hidden_dim = self.vit.conv_proj.out_channels
                self.vit.conv_proj = nn.Conv2d(
                    in_channels,
                    hidden_dim,
                    kernel_size=(patch_size, patch_size),
                    stride=(patch_size, patch_size),
                )
                self.vit.conv_proj.apply(_init_encoder_weights)
        else:
            raise ValueError(f"Unsupported CXR encoder type={encoder_type!r}. Expected 'resnet' or 'transformer'.")


    def init_near_identity_(self, noise_scale: float = 1e-3) -> None:
        """
        Make the final projection head of the ResNet start as a near-identity
        padded/truncated map.
        """
        if self.encoder_type != "resnet":
            return
        _init_linear_near_identity_(self.resnet.fc, noise_scale=noise_scale)

    def forward(self, x):
        if self.encoder_type == "resnet":
            return self.resnet(x)
        return self.vit(x)

class ECGEncoder(nn.Module):
    def __init__(
        self, 
        resnet_params: dict = {
            "type": "resnet",
            "weights": None,  # "IMAGENET1K_V1"
            "norm_type": "batchnorm",
        },
        emb_dim: int = 8192,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        encoder_type = str(resnet_params.get("type", "resnet")).lower()

        if encoder_type == "resnet":
            self.encoder_type = "resnet"
            self.resnet = models.resnet18(weights=resnet_params.get("weights", None))
            if bool(resnet_params.get("disable_skip_connections", False)):
                _disable_resnet_skip_connections_(self.resnet)
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.emb_dim, bias=True)

            nn.init.kaiming_normal_(self.resnet.fc.weight, mode="fan_out")
            nn.init.zeros_(self.resnet.fc.bias)
            nn.init.kaiming_normal_(self.resnet.conv1.weight, mode="fan_out")
        elif encoder_type == "mlp":
            self.encoder_type = "mlp"
            input_dim = int(resnet_params.get("input_dim", 5000 * 12))
            hidden_dims = list(resnet_params.get("hidden_dims", [1024, 2048, 1024]))
            hidden_dropouts = list(resnet_params.get("hidden_dropouts", [0.2, 0.2, 0.2]))
            if len(hidden_dims) != len(hidden_dropouts):
                raise ValueError(
                    "ECG MLP expects hidden_dims and hidden_dropouts to have the same length; "
                    f"got {len(hidden_dims)} and {len(hidden_dropouts)}."
                )

            layers = []
            prev_dim = input_dim
            for hidden_dim, hidden_dropout in zip(hidden_dims, hidden_dropouts):
                layers.extend([
                    nn.Linear(prev_dim, int(hidden_dim)),
                    nn.ReLU(),
                    nn.LayerNorm(int(hidden_dim)),
                    nn.Dropout(float(hidden_dropout)),
                ])
                prev_dim = int(hidden_dim)
            layers.append(nn.Linear(prev_dim, self.emb_dim))
            self.encoder = nn.Sequential(*layers)
            self.encoder.apply(_init_encoder_weights)
        else:
            raise ValueError(f"Unsupported ECG encoder type={encoder_type!r}. Expected 'resnet' or 'mlp'.")

    def init_near_identity_(self, noise_scale: float = 1e-3) -> None:
        """
        Make the final projection head of the ResNet start as a near-identity
        padded/truncated map.
        """
        if self.encoder_type != "resnet":
            return
        _init_linear_near_identity_(self.resnet.fc, noise_scale=noise_scale)

    def forward(self, x):
        if self.encoder_type == "resnet":
            return self.resnet(x)
        return self.encoder(x.float().flatten(start_dim=1))

class LabsEncoder(nn.Module):
    def __init__(
        self,
        emb_dim: int = 8192,
    ):
        super().__init__()
        self.input_dim = 100
        self.emb_dim = emb_dim
        
        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, emb_dim)
        self.act = nn.GELU()
        self.residual_proj = nn.Linear(self.input_dim, emb_dim)
        self.residual_norm = nn.LayerNorm(emb_dim)
        self.residual_drop = nn.Dropout(0.0)

        self.apply(self._init_weights)
        self.init_residual_identity_()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # nn.init.kaiming_normal_(m.weight, mode="fan_out")
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_residual_identity_(self, branch_scale: float = 1e-3) -> None:
        if not isinstance(self.residual_proj, nn.Linear):
            raise TypeError("LabsEncoder.residual_proj is expected to be a Linear layer.")

        with torch.no_grad():
            proj = self.residual_proj
            proj.weight.zero_()
            k = min(proj.out_features, proj.in_features)
            proj.weight[:k, :k] = torch.eye(k, device=proj.weight.device, dtype=proj.weight.dtype)
            if proj.bias is not None:
                proj.bias.zero_()

            linears = [self.fc1, self.fc2, self.fc3]
            for layer in linears:
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()

            first = linears[0]
            last = linears[-1]
            k_first = min(first.out_features, first.in_features)
            k_last = min(last.out_features, last.in_features)
            first.weight[:k_first, :k_first] = branch_scale * torch.eye(
                k_first, device=first.weight.device, dtype=first.weight.dtype
            )
            last.weight[:k_last, :k_last] = branch_scale * torch.eye(
                k_last, device=last.weight.device, dtype=last.weight.dtype
            )

    def forward(self, x):
        x_in = x
        x = self.fc1(x_in)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        residual = self.residual_drop(self.residual_norm(self.residual_proj(x_in)))
        return residual + x



"""
UK Biobank
"""
class UKBTabularEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list[int] = [512, 1024, 512],
        hidden_dropouts: list[float] = [0.1, 0.1, 0.1],
        emb_dim: int = 8192,
        combine_eids_as: str = "intersect",
        shared_adapter: nn.Module = None,
        modality_name: str = None,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.emb_dim = emb_dim
        self.modality_name = modality_name

        self.combine_eids_as = combine_eids_as
        if self.combine_eids_as == "union":
            # also pass binary missing mask as input to the MLP
            input_dim = input_dim * 2
        effective_input_dim = int(input_dim)
        self.residual_proj = nn.Linear(effective_input_dim, emb_dim, bias=True)
        self.residual_drop = nn.Dropout(0.0)
        self.residual_norm = nn.LayerNorm(emb_dim)

        layers = []
        prev = effective_input_dim
        for hidden_dim, hidden_dropout in zip(hidden_dims, hidden_dropouts):
            layers.append(nn.Linear(prev, hidden_dim, bias=True))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(hidden_dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, emb_dim, bias=True))
        self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)
        self.init_residual_identity_()

        if shared_adapter is not None:
            self.mlp = nn.Sequential(
                self.mlp,
                shared_adapter,
            )
            self.shared_adapter = shared_adapter

    def _init_weights(
        self,
        m
    ) -> None: 
        if isinstance(m, (torch.nn.LayerNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def init_residual_identity_(self, branch_scale: float = 1e-3) -> None:
        """
        Initialize the encoder so the residual shortcut carries an identity-like
        map and the nonlinear MLP branch starts near zero.
        """
        if not isinstance(self.residual_proj, nn.Linear):
            raise TypeError("UKBTabularEncoder_MLP.residual_proj is expected to be a Linear layer.")

        linears = [m for m in self.mlp if isinstance(m, nn.Linear)]
        if len(linears) == 0:
            raise TypeError("UKBTabularEncoder_MLP.mlp does not contain any Linear layers.")

        with torch.no_grad():
            proj = self.residual_proj
            proj.weight.zero_()
            k = min(proj.out_features, proj.in_features)
            proj.weight[:k, :k] = torch.eye(k, device=proj.weight.device, dtype=proj.weight.dtype)
            if proj.bias is not None:
                proj.bias.zero_()

            for layer in linears:
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()

            first = linears[0]
            last = linears[-1]
            k_first = min(first.out_features, first.in_features)
            k_last = min(last.out_features, last.in_features)
            first.weight[:k_first, :k_first] = branch_scale * torch.eye(
                k_first, device=first.weight.device, dtype=first.weight.dtype
            )
            last.weight[:k_last, :k_last] = branch_scale * torch.eye(
                k_last, device=last.weight.device, dtype=last.weight.dtype
            )

    def init_residual_identity_nonzero(self, branch_scale: float = 1e-2) -> None:
        """
        Initialize an identity-like shortcut and a fully nonzero MLP branch.

        Scaling only the final MLP layer keeps the initial nonlinear correction
        small without blocking activations or gradients in intermediate layers.
        """
        if not isinstance(self.residual_proj, nn.Linear):
            raise TypeError("UKBTabularEncoder.residual_proj is expected to be a Linear layer.")

        linears = [module for module in self.mlp if isinstance(module, nn.Linear)]
        if not linears:
            raise TypeError("UKBTabularEncoder.mlp does not contain any Linear layers.")

        with torch.no_grad():
            proj = self.residual_proj
            proj.weight.zero_()
            k = min(proj.out_features, proj.in_features)
            proj.weight[:k, :k] = torch.eye(
                k, device=proj.weight.device, dtype=proj.weight.dtype
            )
            if proj.bias is not None:
                proj.bias.zero_()

            for layer in linears:
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    layer.bias.zero_()

            linears[-1].weight.mul_(branch_scale)

    def init_near_identity_(self, noise_scale: float = 1e-3) -> None:
        """
        Initialize the plain labs MLP as a near-identity stack while keeping
        the GELU activations in place.
        """
        _init_linear_near_identity_(self.mlp[0], noise_scale=noise_scale)
        _init_linear_near_identity_(self.mlp[4], noise_scale=noise_scale)
        _init_linear_near_identity_(self.mlp[8], noise_scale=noise_scale)
        _init_linear_near_identity_(self.mlp[12], noise_scale=noise_scale)

    def forward(self, x):
        if self.combine_eids_as == "union":
            nanmask = torch.isnan(x).float()
            x = torch.nan_to_num(x, nan=0.0)
            x = torch.cat([x, nanmask], dim=1)
        if torch.isnan(x).any():
            # raise ValueError("NaN values present in input")
            x = torch.nan_to_num(x, nan=0.0)

        residual = self.residual_drop(self.residual_norm(self.residual_proj(x)))
        return residual + self.mlp(x)

"""
MIMIC (radiology reports + time-series + demographics)
"""
class MimicRadBERTEncoder(nn.Module):
    def __init__(
        self,
        model_params: dict = {
            "text_model_id": "StanfordAIMI/RadBERT",
            "proj_hidden_dim": 512,
            "dropout": 0.0,
            "lora": False,
        },
        emb_dim: int = 8192,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        text_model_id = str(model_params.get("text_model_id", "StanfordAIMI/RadBERT"))
        self.text_encoder = AutoModel.from_pretrained(text_model_id)
        self.input_dim = int(self.text_encoder.config.hidden_size)
        proj_hidden_dim = int(model_params.get("proj_hidden_dim", 512))
        dropout = float(model_params.get("dropout", 0.0))
        branch_initialization = str(
            model_params.get("branch_initialization", "sparse_identity")
        ).lower()
        branch_scale = float(model_params.get("branch_scale", 1e-3))

        for p in self.text_encoder.parameters():
            p.requires_grad = False
        if model_params["lora"]:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=8,
                lora_alpha=8,
                lora_dropout=0.1,
                target_modules=["query", "value"],
                bias="none",
            )
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)

        self.residual_proj = nn.Linear(self.input_dim, self.emb_dim)
        self.residual_dropout = nn.Dropout(dropout)

        self.proj = nn.Sequential(
            nn.Linear(self.input_dim, proj_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden_dim, self.emb_dim),
        )
        self.proj.apply(self._init_weights)
        self.residual_proj.apply(self._init_weights)
        self.init_residual_identity_(
            branch_scale=branch_scale,
            branch_initialization=branch_initialization,
        )

    def _init_weights(self, m) -> None:
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def init_residual_identity_(
        self,
        branch_scale: float = 1e-3,
        branch_initialization: str = "sparse_identity",
    ) -> None:
        if not isinstance(self.residual_proj, nn.Linear):
            raise TypeError("MimicRadBERTEncoder.residual_proj is expected to be a Linear layer.")

        linears = [m for m in self.proj if isinstance(m, nn.Linear)]
        with torch.no_grad():
            self.residual_proj.weight.zero_()
            k = min(self.residual_proj.out_features, self.residual_proj.in_features)
            self.residual_proj.weight[:k, :k] = torch.eye(
                k,
                device=self.residual_proj.weight.device,
                dtype=self.residual_proj.weight.dtype,
            )
            if self.residual_proj.bias is not None:
                self.residual_proj.bias.zero_()

            if branch_initialization == "scaled_final":
                linears[-1].weight.mul_(branch_scale)
                if linears[-1].bias is not None:
                    linears[-1].bias.zero_()
            elif branch_initialization == "sparse_identity":
                for layer in linears:
                    layer.weight.zero_()
                    if layer.bias is not None:
                        layer.bias.zero_()
                for layer in (linears[0], linears[-1]):
                    k = min(layer.out_features, layer.in_features)
                    layer.weight[:k, :k] = branch_scale * torch.eye(
                        k, device=layer.weight.device, dtype=layer.weight.dtype
                    )
            else:
                raise ValueError(
                    "MimicRadBERTEncoder branch_initialization must be "
                    "'sparse_identity' or 'scaled_final'."
                )

    def forward(self, x):
        if torch.is_tensor(x):
            pooled = x
        elif isinstance(x, dict) and "pooled_features" in x:
            pooled = x["pooled_features"]
        elif isinstance(x, dict):
            input_ids = x["input_ids"]
            attention_mask = x["attention_mask"].to(dtype=torch.long)
            out = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
            weights = attention_mask.to(dtype=out.dtype).unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1.0)
            pooled = (out * weights).sum(dim=1) / denom
        else:
            raise TypeError(
                "MimicRadBERTEncoder expects tokenized dict input, pooled feature dict, or pooled tensor."
            )
        return self.residual_dropout(self.residual_proj(pooled)) + self.proj(pooled)


class InspectVisionEncoder(nn.Module):
    """Small 3D residual aggregator for precomputed INSPECT vision maps."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 64,
        num_blocks: int = 1,
        emb_dim: int = 128,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.emb_dim = int(emb_dim)
        hidden_dim = int(hidden_dim)
        num_blocks = int(num_blocks)
        if hidden_dim <= 0:
            raise ValueError("InspectVisionEncoder hidden_dim must be positive.")
        if num_blocks <= 0:
            raise ValueError("InspectVisionEncoder num_blocks must be positive.")

        self.stem = nn.Sequential(
            nn.Conv3d(self.input_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[_InspectVisionResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Conv3d(
            hidden_dim, self.emb_dim, kernel_size=1, bias=True
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.apply(_init_encoder_weights)

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != self.input_dim:
            raise ValueError(
                "InspectVisionEncoder expected input shaped "
                f"(B,{self.input_dim},D,H,W), got {tuple(x.shape)}."
            )
        x = self.stem(x.float())
        x = self.blocks(x)
        x = self.output_proj(x)
        return self.pool(x).flatten(1)


class _InspectVisionResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(1, channels)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.activation(x + residual)


class MimicMLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = [512, 512],
        hidden_dropouts: list[float] = [0.1, 0.1],
        emb_dim: int = 8192,
        branch_initialization: str = "sparse_identity",
        branch_scale: float = 1e-3,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.emb_dim = int(emb_dim)

        layers = []
        prev = self.input_dim
        for hidden_dim, hidden_dropout in zip(hidden_dims, hidden_dropouts):
            layers.append(nn.Linear(prev, hidden_dim, bias=True))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Dropout(hidden_dropout))
            prev = hidden_dim
        layers.append(nn.Linear(prev, self.emb_dim, bias=True))
        self.mlp = nn.Sequential(*layers)

        self.residual_proj = nn.Linear(self.input_dim, self.emb_dim, bias=True)
        self.residual_drop = nn.Dropout(0.0)
        self.residual_norm = nn.LayerNorm(self.emb_dim)

        self.apply(self._init_weights)
        self.init_residual_identity_(
            branch_scale=float(branch_scale),
            branch_initialization=str(branch_initialization).lower(),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def init_residual_identity_(
        self,
        branch_scale: float = 1e-3,
        branch_initialization: str = "sparse_identity",
    ) -> None:
        if not isinstance(self.residual_proj, nn.Linear):
            raise TypeError("MimicMLPEncoder.residual_proj is expected to be a Linear layer.")
        linears = [m for m in self.mlp if isinstance(m, nn.Linear)]
        with torch.no_grad():
            self.residual_proj.weight.zero_()
            k = min(self.residual_proj.out_features, self.residual_proj.in_features)
            self.residual_proj.weight[:k, :k] = torch.eye(
                k,
                device=self.residual_proj.weight.device,
                dtype=self.residual_proj.weight.dtype,
            )
            if self.residual_proj.bias is not None:
                self.residual_proj.bias.zero_()

            if branch_initialization == "scaled_final":
                linears[-1].weight.mul_(branch_scale)
                if linears[-1].bias is not None:
                    linears[-1].bias.zero_()
            elif branch_initialization == "sparse_identity":
                for layer in linears:
                    layer.weight.zero_()
                    if layer.bias is not None:
                        layer.bias.zero_()
                for layer in (linears[0], linears[-1]):
                    k = min(layer.out_features, layer.in_features)
                    layer.weight[:k, :k] = branch_scale * torch.eye(
                        k, device=layer.weight.device, dtype=layer.weight.dtype
                    )
            else:
                raise ValueError(
                    "MimicMLPEncoder branch_initialization must be "
                    "'sparse_identity' or 'scaled_final'."
                )

    def forward(self, x):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)
        residual = self.residual_drop(self.residual_norm(self.residual_proj(x)))
        return residual + self.mlp(x)


"""
Synthetic XNOR
"""
class SyntheticXNOREncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        emb_dim: int = 8192,
        input_layout: str = "1d",
        sequence_encoder_params: dict = {
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.0,
            "dim_feedforward_mult": 4,
            "layer_type": "transformer",
        },
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.emb_dim = emb_dim
        self.input_layout = str(input_layout)

        self.input_proj = nn.Linear(input_dim, emb_dim)
        self.residual_proj = nn.Linear(input_dim, emb_dim)

        if self.input_layout == "1d":
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim)
            )
            self.apply(self._init_weights)
        elif self.input_layout == "2d":
            nhead = sequence_encoder_params["nhead"]
            num_layers = sequence_encoder_params["num_layers"]
            dropout = 0.0
            ff_mult = sequence_encoder_params["dim_feedforward_mult"]
            seq_len = sequence_encoder_params["seq_len"]
            layer_type = str(sequence_encoder_params.get("layer_type", "transformer"))
            self.seq_len = seq_len

            self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, emb_dim))
            nn.init.normal_(self.pos_embed, std=0.02)

            self.seq_input_proj = nn.Linear(self.input_dim, self.emb_dim)
            if layer_type == "transformer":
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.emb_dim,
                    nhead=nhead,
                    dim_feedforward=max(self.emb_dim, ff_mult * self.emb_dim),
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="gelu",
                )
            elif layer_type == "tokenwise_mlp":
                from utils import TokenwiseMLPEncoderLayer
                encoder_layer = TokenwiseMLPEncoderLayer(
                    d_model=self.emb_dim,
                    nhead=nhead,
                    dim_feedforward=max(self.emb_dim, ff_mult * self.emb_dim),
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="relu",
                )
            elif layer_type == "uniform_attention":
                from utils import UniformAttentionEncoderLayer
                encoder_layer = UniformAttentionEncoderLayer(
                    d_model=self.emb_dim,
                    nhead=nhead,
                    dim_feedforward=max(self.emb_dim, ff_mult * self.emb_dim),
                    dropout=dropout,
                    batch_first=True,
                    norm_first=True,
                    activation="relu",
                )
            else:
                raise ValueError(
                    f"Unknown sequence_encoder_params.layer_type: {layer_type}. "
                    "Expected one of ['transformer', 'tokenwise_mlp', 'uniform_attention']."
                )

            self.seq_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.seq_out_proj = nn.Linear(self.emb_dim, self.emb_dim)

            self.apply(self._init_weights)

            #for layer in self.seq_encoder.layers:
            #    layer.self_attn.out_proj.weight.data.mul_(1e-3)
            #    layer.self_attn.out_proj.bias.data.zero_()
            #    layer.linear2.weight.data.mul_(1e-3)
            #    layer.linear2.bias.data.zero_()

        #self.apply_burkholz_relu_init(self.mlp, mode="orthogonal")
        self.init_residual_identity_()
    
    def _init_weights(
        self,
        m
    ) -> None: 
        if isinstance(m, (torch.nn.LayerNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    @staticmethod
    def _init_linear_near_identity_(
        layer: nn.Linear,
        noise_scale: float = 1e-3,
    ) -> None:
        """
        Initialize a Linear layer as the closest identity-like map allowed by
        its shape, plus a tiny perturbation.

        For non-square layers this becomes a padded/truncated identity. This is
        useful when input_dim != emb_dim but we still want to preserve input
        geometry as much as possible at initialization.
        """
        if not isinstance(layer, nn.Linear):
            raise TypeError("Expected nn.Linear in _init_linear_near_identity_.")

        with torch.no_grad():
            layer.weight.zero_()
            k = min(layer.out_features, layer.in_features)
            layer.weight[:k, :k] = torch.eye(
                k, device=layer.weight.device, dtype=layer.weight.dtype
            )
            if noise_scale > 0.0:
                layer.weight.add_(noise_scale * torch.randn_like(layer.weight))
            if layer.bias is not None:
                layer.bias.zero_()

    def init_identity_(self) -> None:
        """
        Reconfigure the MLP to be an identity-like map.

        The ReLU activations are replaced by Identity modules. The first Linear
        layer is initialized as a padded/truncated identity when
        input_dim != emb_dim, while the remaining square Linear layers are set
        to exact identity.
        """
        if not isinstance(self.mlp[0], nn.Linear) or not isinstance(self.mlp[2], nn.Linear) or not isinstance(self.mlp[4], nn.Linear):
            raise TypeError("SyntheticXNOREncoder.mlp does not have the expected Linear/ReLU/Linear/ReLU/Linear structure.")

        self.mlp[1] = nn.Identity()
        self.mlp[3] = nn.Identity()

        with torch.no_grad():
            first = self.mlp[0]
            first.weight.zero_()
            k = min(first.out_features, first.in_features)
            first.weight[:k, :k] = torch.eye(k, device=first.weight.device, dtype=first.weight.dtype)
            if first.bias is not None:
                first.bias.zero_()

            for idx in (2, 4):
                layer = self.mlp[idx]
                layer.weight.zero_()
                layer.weight.add_(torch.eye(layer.out_features, device=layer.weight.device, dtype=layer.weight.dtype))
                if layer.bias is not None:
                    layer.bias.zero_()

    def init_near_identity_(self, noise_scale: float = 1e-3) -> None:
        """
        Initialize the plain MLP path as a near-identity map.

        This keeps the ReLU activations in place, but makes every Linear layer
        start as the closest identity-like transform allowed by its shape,
        perturbed only by tiny noise.
        """
        if not isinstance(self.mlp[0], nn.Linear) or not isinstance(self.mlp[2], nn.Linear) or not isinstance(self.mlp[4], nn.Linear):
            raise TypeError("SyntheticXNOREncoder.mlp does not have the expected Linear/ReLU/Linear/ReLU/Linear structure.")

        for idx in (0, 2, 4):
            self._init_linear_near_identity_(self.mlp[idx], noise_scale=noise_scale)

    def init_residual_identity_(self, branch_scale: float = 1e-3) -> None:
        """
        Initialize the residual encoder so the shortcut carries an identity-like
        map and the nonlinear MLP branch starts near zero.

        This is useful for residual formulations like:
            y = residual_proj(x) + mlp(x)
        where we want to preserve input geometry at initialization while still
        keeping a trainable nonlinear branch.
        """
        if not isinstance(self.residual_proj, nn.Linear):
            raise TypeError("SyntheticXNOREncoder.residual_proj is expected to be a Linear layer.")

        with torch.no_grad():
            proj = self.residual_proj
            proj.weight.zero_()
            k = min(proj.out_features, proj.in_features)
            proj.weight[:k, :k] = torch.eye(k, device=proj.weight.device, dtype=proj.weight.dtype)
            if proj.bias is not None:
                proj.bias.zero_()

            if self.input_layout == "1d":
                if not isinstance(self.mlp[0], nn.Linear) or not isinstance(self.mlp[2], nn.Linear) or not isinstance(self.mlp[4], nn.Linear):
                    raise TypeError("SyntheticXNOREncoder.mlp does not have the expected Linear/Act/Linear/Act/Linear structure.")

                for idx in (0, 2, 4):
                    layer = self.mlp[idx]
                    layer.weight.zero_()
                    if layer.bias is not None:
                        layer.bias.zero_()

                # Keep a tiny non-zero branch so gradients can start shaping it,
                # while the residual path dominates at initialization.
                first = self.mlp[0]
                last = self.mlp[4]
                k_first = min(first.out_features, first.in_features)
                k_last = min(last.out_features, last.in_features)
                first.weight[:k_first, :k_first] = branch_scale * torch.eye(
                    k_first, device=first.weight.device, dtype=first.weight.dtype
                )
                last.weight[:k_last, :k_last] = branch_scale * torch.eye(
                    k_last, device=last.weight.device, dtype=last.weight.dtype
                )
            elif self.input_layout == "2d":
                # Keep transformer branch near zero at initialization while
                # residual path carries identity-like mapping.
                self.seq_input_proj.weight.zero_()
                k_seq = min(self.seq_input_proj.out_features, self.seq_input_proj.in_features)
                self.seq_input_proj.weight[:k_seq, :k_seq] = branch_scale * torch.eye(
                    k_seq,
                    device=self.seq_input_proj.weight.device,
                    dtype=self.seq_input_proj.weight.dtype,
                )
                if self.seq_input_proj.bias is not None:
                    self.seq_input_proj.bias.zero_()
            else:
                raise ValueError(f"Unknown input_layout: {self.input_layout}")

    def apply_burkholz_relu_init(self, model: nn.Module, mode: str = "orthogonal"):
        from utils import burkholz_relu_init_linear
        for m in model.modules():
            if isinstance(m, nn.Linear):
                burkholz_relu_init_linear(m, mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_layout == "1d":
            z = self.mlp(x)
            return self.residual_proj(x) + z
        elif self.input_layout == "2d":
            z_tokens = self.seq_input_proj(x)
            z_tokens = z_tokens + self.pos_embed

            residual = self.residual_proj(x)
            if residual.ndim == 2:
                residual = residual.unsqueeze(1)
            z_tokens = z_tokens + residual

            z_tokens = self.seq_encoder(z_tokens)
            z = self.seq_out_proj(z_tokens.mean(dim=1))
            return z
        else:
            raise ValueError(f"Unknown input_layout: {self.input_layout}")
