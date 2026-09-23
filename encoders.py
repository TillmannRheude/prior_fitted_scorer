import torch
import torch.nn as nn

from torchvision import models
from transformers import AutoModel

from helpers import _init_linear_near_identity_


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
        if resnet_params["type"] != "resnet":
            raise ValueError("CXREncoder supports the ResNet used in the paper.")
        self.resnet = models.resnet50(weights=resnet_params["weights"])
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.emb_dim, bias=True)
        nn.init.kaiming_normal_(self.resnet.fc.weight, mode="fan_out")
        nn.init.zeros_(self.resnet.fc.bias)

    def init_near_identity_(self, noise_scale: float = 1e-3) -> None:
        """
        Make the final projection head of the ResNet start as a near-identity
        padded/truncated map.
        """
        _init_linear_near_identity_(self.resnet.fc, noise_scale=noise_scale)

    def forward(self, x):
        return self.resnet(x)

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
        if resnet_params["type"] != "resnet":
            raise ValueError("ECGEncoder supports the ResNet used in the paper.")
        self.resnet = models.resnet18(weights=resnet_params["weights"])
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.emb_dim, bias=True)
        nn.init.kaiming_normal_(self.resnet.fc.weight, mode="fan_out")
        nn.init.zeros_(self.resnet.fc.bias)
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode="fan_out")

    def init_near_identity_(self, noise_scale: float = 1e-3) -> None:
        """
        Make the final projection head of the ResNet start as a near-identity
        padded/truncated map.
        """
        _init_linear_near_identity_(self.resnet.fc, noise_scale=noise_scale)

    def forward(self, x):
        return self.resnet(x)

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
