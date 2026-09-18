from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum

import torch
from torch import Tensor
import torch.nn.functional as F


class MechanismType(str, Enum):
    LINEAR = "linear"
    TANH = "tanh"
    SINE = "sine"
    MULTIPLICATIVE = "multiplicative"
    RELU = "relu"
    GELU = "gelu"
    QUADRATIC = "quadratic"
    MLP = "mlp"


# Latent-to-modality roles for three modalities (A, B, C).
_ROLE_MASKS: dict[str, tuple[bool, bool, bool]] = {
    "shared_all": (True, True, True),
    "abc": (True, True, True),
    "ab": (True, True, False),
    "ac": (True, False, True),
    "bc": (False, True, True),
    "a": (True, False, False),
    "b": (False, True, False),
    "c": (False, False, True),
    "semantic_root": (False, False, False),
    "ab_anchor": (False, False, False),
    "ac_anchor": (False, False, False),
    "bc_anchor": (False, False, False),
    "a_child": (True, False, False),
    "b_child": (False, True, False),
    "c_child": (False, False, True),
    "unused": (False, False, False),
}

_ALLOCATION_KEYS = {
    "shared_all",
    "abc",
    "pairwise",
    "ab",
    "ac",
    "bc",
    "private",
    "a",
    "b",
    "c",
    "unused",
}


def _fractions_to_counts(fractions: list[float], total: int) -> list[int]:
    """Largest-remainder conversion of fractions to integer counts summing to ``total``."""
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if any(f < 0 for f in fractions):
        raise ValueError(f"fractions must be non-negative, got {fractions}")

    raw = [f * total for f in fractions]
    counts = [int(math.floor(x)) for x in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(raw)),
        key=lambda i: (raw[i] - counts[i], -i),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


@dataclass
class SCMConfig:
    n_modalities: int = 3
    n_latents: int = 8
    latent_dim: int = 4
    embedding_dim: int = 32

    edge_probability: float = 0.3
    modality_connection_probability: float = 0.5

    # "static": use scalar values above.
    # "sampled": sample per SCM from the corresponding *_choices below.
    parameter_sampling_mode: str = "static"
    n_latents_choices: list[int] | None = None
    latent_dim_choices: list[int] | None = None
    modality_readout_alpha_choices: list[float] | None = None
    edge_probability_choices: list[float] | None = None

    # "bernoulli": independent connections via modality_connection_probability.
    # "structured": allocate latents to shared_all / pairwise / private(/unused) roles.
    # "semantic_children": unobserved semantic roots cause modality-specific
    # children; modalities observe those children plus private latents.
    modality_mask_mode: str = "bernoulli"
    # "null": current direct shared_all behavior.
    # "semantic_anchor": structured shared_all slots become unobserved anchors
    # with modality-specific child latents where possible.
    shared_all_mode: str | None = None
    # "null": current direct pairwise behavior.
    # "semantic_anchor": structured pairwise slots become unobserved pair anchors
    # with two modality-specific child latents where possible.
    pairwise_mode: str | None = None
    # Fraction of latents used as unobserved semantic roots in semantic_children
    # mode. Each root consumes three additional latents as A/B/C children.
    semantic_root_fraction: float = 0.125
    # Fraction or count allocation for structured masks. Examples:
    #   {"shared_all": 0.25, "pairwise": 0.375, "private": 0.375}
    #   {"shared_all": 2, "ab": 1, "ac": 1, "bc": 1, "a": 1, "b": 1, "c": 1}
    # Integer-valued entries are treated as counts; otherwise as fractions of n_latents.
    # Remainder (if any) becomes unused. Fine-grained keys ab/ac/bc or a/b/c may replace
    # the grouped keys pairwise / private.
    modality_allocation: dict[str, float] | None = None
    # "static": use modality_allocation above.
    # "sampled": sample one allocation per SCM from modality_allocation_choices.
    modality_allocation_mode: str = "static"
    modality_allocation_choices: list[dict[str, float]] | None = None

    # Allowed structural-equation mechanism types. None / empty means all.
    mechanism_types: list[str] | None = None
    # "static": use mechanism_types above.
    # "sampled": sample one mechanism set per SCM from mechanism_type_choices.
    mechanism_types_mode: str = "static"
    mechanism_type_choices: list[list[str]] | None = None

    # Modality observation map: "linear" (masked sum of projections) or
    # "mlp" (small MLP over masked concatenated latents).
    # With readout "mlp", modality_readout_alpha mixes:
    #   x = (1 - alpha) * linear + alpha * mlp
    # so alpha=0 is pure linear, alpha=1 is pure MLP. Ignored for readout "linear".
    modality_readout: str = "linear"
    modality_readout_alpha: float = 1.0
    # Hidden width for MLP readout. None => max(embedding_dim, 2 * latent_dim).
    modality_mlp_hidden_dim: int | None = None

    latent_noise_std: float = 0.1
    modality_noise_std: list[float] | None = None
    modality_noise_std_choices: list[float] | list[list[float]] | None = None
    seed: int = 42
    shared_modality_projections: bool = False
    shared_projection_seed: int | None = None

    def resolved_mechanism_types(self) -> list[MechanismType]:
        if not self.mechanism_types:
            return [
                MechanismType.LINEAR,
                MechanismType.TANH,
                MechanismType.SINE,
                MechanismType.RELU,
                MechanismType.GELU,
                MechanismType.MLP,
            ]

        allowed = {mechanism.value: mechanism for mechanism in MechanismType}
        resolved: list[MechanismType] = []
        for name in self.mechanism_types:
            key = str(name).strip().lower()
            if key not in allowed:
                raise ValueError(
                    f"Unknown mechanism type {name!r}. "
                    f"Expected one of {sorted(allowed)}."
                )
            resolved.append(allowed[key])

        if not resolved:
            raise ValueError("mechanism_types must contain at least one mechanism.")
        return resolved

    def resolved_sampling_mode(self, value: str, name: str) -> str:
        mode = str(value).strip().lower()
        if mode not in {"static", "sampled"}:
            raise ValueError(
                f"Unknown {name}={value!r}. Expected 'static' or 'sampled'."
            )
        return mode

    def resolved_parameter_sampling_mode(self) -> str:
        return self.resolved_sampling_mode(
            self.parameter_sampling_mode,
            "parameter_sampling_mode",
        )

    def resolved_modality_allocation_mode(self) -> str:
        return self.resolved_sampling_mode(
            self.modality_allocation_mode,
            "modality_allocation_mode",
        )

    def resolved_mechanism_types_mode(self) -> str:
        return self.resolved_sampling_mode(
            self.mechanism_types_mode,
            "mechanism_types_mode",
        )

    def resolved_modality_mask_mode(self) -> str:
        mode = str(self.modality_mask_mode).strip().lower()
        allowed = {
            "bernoulli",
            "structured",
            "semantic_children",
        }
        if mode not in allowed:
            raise ValueError(
                f"Unknown modality_mask_mode={self.modality_mask_mode!r}. "
                f"Expected one of {sorted(allowed)}."
            )
        if self.n_modalities != 3 and mode in {
            "structured",
            "semantic_children",
        }:
            raise ValueError(
                f"{mode} modality masks currently require n_modalities=3."
            )
        return mode

    def resolved_shared_all_mode(self) -> str:
        return self._resolved_anchor_mode(self.shared_all_mode, "shared_all_mode")

    def resolved_pairwise_mode(self) -> str:
        return self._resolved_anchor_mode(self.pairwise_mode, "pairwise_mode")

    def _resolved_anchor_mode(self, value, name: str) -> str:
        if value is None:
            return "null"
        mode = str(value).strip().lower()
        if mode in {"", "none", "null"}:
            return "null"
        if mode not in {"semantic_anchor"}:
            raise ValueError(
                f"Unknown {name}={value!r}. "
                "Expected 'null' or 'semantic_anchor'."
            )
        return mode

    def resolved_modality_readout(self) -> str:
        readout = str(self.modality_readout).strip().lower()
        if readout not in {"linear", "mlp"}:
            raise ValueError(
                f"Unknown modality_readout={self.modality_readout!r}. "
                "Expected 'linear' or 'mlp'."
            )
        return readout

    def resolved_modality_readout_alpha(self) -> float:
        """Mixing weight toward the MLP readout: 0=linear, 1=mlp."""
        if self.resolved_modality_readout() == "linear":
            return 0.0
        alpha = float(self.modality_readout_alpha)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(
                f"modality_readout_alpha must be in [0, 1], got {alpha}."
            )
        return alpha

    def uses_linear_readout(self) -> bool:
        return self.resolved_modality_readout_alpha() < 1.0 - 1e-12

    def uses_mlp_readout(self) -> bool:
        return self.resolved_modality_readout_alpha() > 1e-12

    def resolved_modality_mlp_hidden_dim(self) -> int:
        if self.modality_mlp_hidden_dim is None:
            return max(int(self.embedding_dim), 2 * int(self.latent_dim))
        hidden = int(self.modality_mlp_hidden_dim)
        if hidden < 1:
            raise ValueError(f"modality_mlp_hidden_dim must be >= 1, got {hidden}.")
        return hidden

    def resolved_modality_noise_std(self) -> list[float]:
        if self.modality_noise_std is None:
            values = [0.3 for _ in range(int(self.n_modalities))]
        else:
            values = [float(value) for value in self.modality_noise_std]
        if len(values) != int(self.n_modalities):
            raise ValueError(
                f"modality_noise_std must have length {self.n_modalities}, got {values}."
            )
        if any(value < 0 for value in values):
            raise ValueError(f"modality_noise_std values must be non-negative, got {values}.")
        return values

@dataclass
class ModalityMLP:
    weight_1: Tensor
    bias_1: Tensor
    weight_2: Tensor
    bias_2: Tensor

    def clone(self) -> "ModalityMLP":
        return ModalityMLP(
            weight_1=self.weight_1.clone(),
            bias_1=self.bias_1.clone(),
            weight_2=self.weight_2.clone(),
            bias_2=self.bias_2.clone(),
        )


@dataclass
class StructuralEquation:
    parents: list[int]
    mechanism_type: MechanismType
    weights: list[Tensor]
    bias: Tensor
    mlp_weight_1: Tensor | None = None
    mlp_bias_1: Tensor | None = None
    mlp_weight_2: Tensor | None = None
    mlp_bias_2: Tensor | None = None

    def __call__(self, parent_values: list[Tensor]) -> Tensor:
        if not parent_values:
            raise ValueError("Structural equation requires parents.")

        transformed = [
            parent @ weight
            for parent, weight in zip(parent_values, self.weights)
        ]

        if self.mechanism_type == MechanismType.LINEAR:
            output = sum(transformed)

        elif self.mechanism_type == MechanismType.TANH:
            output = torch.tanh(sum(transformed))

        elif self.mechanism_type == MechanismType.SINE:
            output = torch.sin(sum(transformed))

        elif self.mechanism_type == MechanismType.MULTIPLICATIVE:
            output = transformed[0]
            for value in transformed[1:]:
                output = output * value

        elif self.mechanism_type == MechanismType.RELU:
            output = F.relu(sum(transformed))

        elif self.mechanism_type == MechanismType.GELU:
            output = F.gelu(sum(transformed))

        elif self.mechanism_type == MechanismType.QUADRATIC:
            linear = sum(transformed)
            output = linear + 0.25 * torch.sign(linear) * linear.square()

        elif self.mechanism_type == MechanismType.MLP:
            if (
                self.mlp_weight_1 is None
                or self.mlp_bias_1 is None
                or self.mlp_weight_2 is None
                or self.mlp_bias_2 is None
            ):
                raise ValueError("MLP structural equation is missing sampled MLP parameters.")
            parent_concat = torch.cat(parent_values, dim=-1)
            hidden = F.gelu(parent_concat @ self.mlp_weight_1 + self.mlp_bias_1)
            output = hidden @ self.mlp_weight_2 + self.mlp_bias_2

        else:
            raise ValueError(
                f"Unknown mechanism: {self.mechanism_type}"
            )

        return output + self.bias


class SampledSCM:
    """
    One sampled structural causal model.

    Latent variables form a DAG:

        Z_j := f_j(parents(Z_j), U_j)

    Modalities are generated as:

        X_m := g_m(selected latent variables, epsilon_m)

    where g_m is a linear mix of latent projections, an MLP over masked
    concatenated latents, or a convex combination
    ``(1 - alpha) * linear + alpha * mlp``.
    """

    def __init__(
        self,
        config: SCMConfig,
        scm_seed: int,
        modality_projections: list[list[Tensor]] | None = None,
        modality_mlps: list[ModalityMLP] | None = None,
    ):
        self.config = config
        self.scm_seed = scm_seed
        self.generator = torch.Generator().manual_seed(scm_seed)
        self.semantic_child_edges: list[tuple[int, int]] = []
        self.semantic_child_parent: dict[int, int] = {}

        self.modality_masks, self.latent_roles = self._sample_modality_masks()
        self.adjacency = self._sample_dag()
        self.structural_equations = self._sample_structural_equations()

        self.modality_readout = self.config.resolved_modality_readout()
        self.modality_readout_alpha = self.config.resolved_modality_readout_alpha()
        self.modality_projections: list[list[Tensor]] | None = None
        self.modality_projection_tensor: Tensor | None = None
        self.modality_mlps: list[ModalityMLP] | None = None

        if self.config.uses_linear_readout():
            if modality_projections is None:
                self.modality_projections = self._sample_modality_projections()
            else:
                self.modality_projections = [
                    [projection.clone() for projection in modality_projection]
                    for modality_projection in modality_projections
                ]
            self.modality_projection_tensor = self._stack_modality_projections()

        if self.config.uses_mlp_readout():
            if modality_mlps is None:
                self.modality_mlps = self._sample_modality_mlps()
            else:
                self.modality_mlps = [mlp.clone() for mlp in modality_mlps]

    def _randn(self, *shape: int) -> Tensor:
        return torch.randn(*shape, generator=self.generator)

    def _sample_dag(self) -> Tensor:
        """
        Samples an exchangeable DAG prior.

        A random topological order is drawn first, then each forward edge in
        that order is included independently with probability
        ``edge_probability``. This removes the index bias of a fixed
        lower-to-higher ordering while remaining acyclic.
        """
        n_latents = self.config.n_latents
        order = self._sample_topological_order(n_latents)
        self.topological_order = order

        adjacency = torch.zeros(
            n_latents,
            n_latents,
            dtype=torch.bool,
        )

        for child_pos in range(n_latents):
            child = int(order[child_pos].item())
            for parent_pos in range(child_pos):
                parent = int(order[parent_pos].item())
                if self._blocks_random_incoming_edge(child):
                    continue
                include_edge = (
                    torch.rand((), generator=self.generator).item()
                    < self.config.edge_probability
                )
                adjacency[parent, child] = include_edge

        for parent, child in self.semantic_child_edges:
            adjacency[parent, child] = True

        return adjacency

    def _blocks_random_incoming_edge(self, child: int) -> bool:
        if child in self.semantic_child_parent:
            return True
        if self.config.resolved_modality_mask_mode() != "semantic_children":
            return False
        return self.latent_roles[child] in {
            "semantic_root",
            "a_child",
            "b_child",
            "c_child",
        }

    def _sample_topological_order(self, n_latents: int) -> Tensor:
        if not self.semantic_child_edges:
            return torch.randperm(n_latents, generator=self.generator)

        forced_children: dict[int, list[int]] = {index: [] for index in range(n_latents)}
        indegree = [0 for _ in range(n_latents)]
        for parent, child in self.semantic_child_edges:
            forced_children[parent].append(child)
            indegree[child] += 1
            self.semantic_child_parent[child] = parent

        random_priority = torch.rand(n_latents, generator=self.generator).tolist()
        available = [index for index, degree in enumerate(indegree) if degree == 0]
        order: list[int] = []

        while available:
            available.sort(key=lambda index: random_priority[index])
            node = available.pop(0)
            order.append(node)
            for child in forced_children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    available.append(child)

        if len(order) != n_latents:
            raise RuntimeError("Forced semantic-child edges produced a cycle.")

        return torch.tensor(order, dtype=torch.long)

    def _sample_structural_equations(
        self,
    ) -> list[StructuralEquation | None]:

        equations: list[StructuralEquation | None] = []
        mechanism_types = self.config.resolved_mechanism_types()
        semantic_edge_weights = self._sample_correlated_semantic_anchor_edge_weights()

        for child in range(self.config.n_latents):
            parents = torch.where(
                self.adjacency[:, child]
            )[0].tolist()

            if not parents:
                equations.append(None)
                continue

            semantic_parent = self.semantic_child_parent.get(child)
            if semantic_parent is not None:
                parents = [semantic_parent]
                weights = [semantic_edge_weights[(semantic_parent, child)]]
            else:
                parent_weight_scale = math.sqrt(
                    self.config.latent_dim * len(parents)
                )
                weights = [
                    self._randn(
                        self.config.latent_dim,
                        self.config.latent_dim,
                    ) / parent_weight_scale
                    for _ in parents
                ]

            mechanism_index = torch.randint(
                len(mechanism_types),
                size=(1,),
                generator=self.generator,
            ).item()

            mechanism_type = mechanism_types[mechanism_index]

            bias = 0.1 * self._randn(self.config.latent_dim)
            mlp_weight_1 = None
            mlp_bias_1 = None
            mlp_weight_2 = None
            mlp_bias_2 = None

            if mechanism_type == MechanismType.MLP:
                input_dim = len(parents) * self.config.latent_dim
                hidden_dim = max(2 * self.config.latent_dim, 8)
                mlp_weight_1 = self._randn(
                    input_dim,
                    hidden_dim,
                ) / input_dim**0.5
                mlp_bias_1 = 0.1 * self._randn(hidden_dim)
                mlp_weight_2 = self._randn(
                    hidden_dim,
                    self.config.latent_dim,
                ) / hidden_dim**0.5
                mlp_bias_2 = 0.1 * self._randn(self.config.latent_dim)

            equations.append(
                StructuralEquation(
                    parents=parents,
                    mechanism_type=mechanism_type,
                    weights=weights,
                    bias=bias,
                    mlp_weight_1=mlp_weight_1,
                    mlp_bias_1=mlp_bias_1,
                    mlp_weight_2=mlp_weight_2,
                    mlp_bias_2=mlp_bias_2,
                )
            )

        return equations

    def _sample_correlated_semantic_anchor_edge_weights(self) -> dict[tuple[int, int], Tensor]:
        if not self.semantic_child_edges:
            return {}

        children_by_parent: dict[int, list[int]] = {}
        for parent, child in self.semantic_child_edges:
            children_by_parent.setdefault(int(parent), []).append(int(child))

        perturbation = 0.2
        scale = self.config.latent_dim**0.5
        edge_weights: dict[tuple[int, int], Tensor] = {}

        for parent, children in children_by_parent.items():
            base_weight = self._randn(
                self.config.latent_dim,
                self.config.latent_dim,
            ) / scale
            for child in children:
                delta = self._randn(
                    self.config.latent_dim,
                    self.config.latent_dim,
                ) / scale
                edge_weights[(parent, child)] = base_weight + perturbation * delta

        return edge_weights

    def _sample_modality_masks(self) -> tuple[Tensor, list[str]]:
        """
        mask[m, j] indicates whether latent Z_j affects modality m.

        Returns the boolean mask and a per-latent role label list.
        """
        mode = self.config.resolved_modality_mask_mode()
        if mode == "structured":
            masks, roles = self._sample_modality_masks_structured()
        elif mode == "semantic_children":
            masks, roles = self._sample_modality_masks_semantic_children()
        else:
            masks, roles = self._sample_modality_masks_bernoulli()

        # Ensure every modality receives at least one latent variable.
        for modality in range(self.config.n_modalities):
            if not masks[modality].any():
                latent_index = int(
                    torch.randint(
                        self.config.n_latents,
                        size=(1,),
                        generator=self.generator,
                    ).item()
                )
                masks[modality, latent_index] = True
                roles[latent_index] = self._role_from_mask(masks[:, latent_index])

        return masks, roles

    def _sample_modality_masks_bernoulli(self) -> tuple[Tensor, list[str]]:
        masks = torch.rand(
            self.config.n_modalities,
            self.config.n_latents,
            generator=self.generator,
        ) < self.config.modality_connection_probability
        roles = [self._role_from_mask(masks[:, j]) for j in range(self.config.n_latents)]
        return masks, roles

    def _sample_modality_masks_semantic_children(self) -> tuple[Tensor, list[str]]:
        """
        Create unobserved semantic roots with modality-specific descendants.

        For each root R_g, forced DAG edges R_g -> A_g/B_g/C_g are added later.
        Modalities observe only their child, not the root. Remaining latents are
        assigned as private modality factors.
        """
        n_latents = int(self.config.n_latents)
        if n_latents < 4:
            raise ValueError(
                "semantic_children modality masks require n_latents >= 4 "
                f"because each semantic group uses root+A/B/C child, got {n_latents}."
            )

        root_fraction = float(self.config.semantic_root_fraction)
        if root_fraction <= 0:
            raise ValueError(
                f"semantic_root_fraction must be > 0 in semantic_children mode, got {root_fraction}."
            )

        max_roots = n_latents // 4
        n_roots = int(round(root_fraction * n_latents))
        n_roots = max(1, min(max_roots, n_roots))

        masks = torch.zeros(
            self.config.n_modalities,
            n_latents,
            dtype=torch.bool,
        )
        roles = ["unused" for _ in range(n_latents)]
        self.semantic_child_edges = []

        permuted_indices = torch.randperm(n_latents, generator=self.generator).tolist()
        cursor = 0
        child_role_names = ("a_child", "b_child", "c_child")

        for root_index in range(n_roots):
            root = int(permuted_indices[cursor])
            children = [int(index) for index in permuted_indices[cursor + 1 : cursor + 4]]
            cursor += 4

            roles[root] = "semantic_root"
            for modality, child in enumerate(children):
                masks[modality, child] = True
                roles[child] = child_role_names[modality]
                self.semantic_child_edges.append((root, child))

        remaining = [int(index) for index in permuted_indices[cursor:]]
        if remaining:
            modality_order = torch.randperm(
                self.config.n_modalities,
                generator=self.generator,
            ).tolist()
            private_role_names = ("a", "b", "c")
            for offset, latent_index in enumerate(remaining):
                modality = int(modality_order[offset % self.config.n_modalities])
                masks[modality, latent_index] = True
                roles[latent_index] = private_role_names[modality]

        return masks, roles

    def _sample_modality_masks_structured(self) -> tuple[Tensor, list[str]]:
        role_counts = self._resolve_structured_role_counts()
        roles, semantic_anchor_edges = self._structured_roles_from_counts(role_counts)

        if len(roles) != self.config.n_latents:
            raise RuntimeError(
                f"Structured role allocation produced {len(roles)} roles, "
                f"expected {self.config.n_latents}."
            )

        if self.config.n_latents > 1:
            perm = torch.randperm(self.config.n_latents, generator=self.generator)
            roles = [roles[int(i)] for i in perm.tolist()]
            inverse_perm = [0 for _ in range(self.config.n_latents)]
            for new_index, old_index in enumerate(perm.tolist()):
                inverse_perm[int(old_index)] = new_index
            semantic_anchor_edges = [
                (inverse_perm[parent], inverse_perm[child])
                for parent, child in semantic_anchor_edges
            ]

        masks = torch.zeros(
            self.config.n_modalities,
            self.config.n_latents,
            dtype=torch.bool,
        )
        for latent_index, role in enumerate(roles):
            pattern = _ROLE_MASKS[role]
            for modality, connected in enumerate(pattern):
                masks[modality, latent_index] = connected

        self.semantic_child_edges.extend(semantic_anchor_edges)
        return masks, roles

    def _structured_roles_from_counts(
        self,
        role_counts: dict[str, int],
    ) -> tuple[list[str], list[tuple[int, int]]]:
        role_counts = dict(role_counts)
        roles: list[str] = []
        semantic_anchor_edges: list[tuple[int, int]] = []

        shared_count = int(role_counts.get("shared_all", 0))
        if self.config.resolved_shared_all_mode() == "semantic_anchor":
            shared_roles, shared_edges, leftover_shared = self._semantic_anchor_roles(
                count=shared_count,
                root_role="semantic_root",
                child_roles=("a_child", "b_child", "c_child"),
            )
            offset = len(roles)
            roles.extend(shared_roles)
            semantic_anchor_edges.extend(
                (offset + parent, offset + child)
                for parent, child in shared_edges
            )
            role_counts["shared_all"] = 0

            if leftover_shared:
                pair_counts = self._split_grouped_count(leftover_shared, ("ab", "ac", "bc"))
                for role, count in pair_counts.items():
                    role_counts[role] = int(role_counts.get(role, 0)) + int(count)
        elif shared_count:
            roles.extend(["shared_all"] * shared_count)
            role_counts["shared_all"] = 0

        if self.config.resolved_pairwise_mode() == "semantic_anchor":
            pair_specs = {
                "ab": ("ab_anchor", ("a_child", "b_child")),
                "ac": ("ac_anchor", ("a_child", "c_child")),
                "bc": ("bc_anchor", ("b_child", "c_child")),
            }
            for pair_role, (root_role, child_roles) in pair_specs.items():
                pair_count = int(role_counts.get(pair_role, 0))
                pair_roles, pair_edges, leftover_pair = self._semantic_anchor_roles(
                    count=pair_count,
                    root_role=root_role,
                    child_roles=child_roles,
                )
                offset = len(roles)
                roles.extend(pair_roles)
                semantic_anchor_edges.extend(
                    (offset + parent, offset + child)
                    for parent, child in pair_edges
                )
                role_counts[pair_role] = leftover_pair

        for role, count in role_counts.items():
            if role == "shared_all":
                continue
            roles.extend([role] * int(count))

        return roles, semantic_anchor_edges

    def _semantic_anchor_roles(
        self,
        count: int,
        root_role: str,
        child_roles: tuple[str, ...],
    ) -> tuple[list[str], list[tuple[int, int]], int]:
        group_size = 1 + len(child_roles)
        n_groups = int(count) // group_size
        leftover = int(count) - group_size * n_groups
        roles: list[str] = []
        edges: list[tuple[int, int]] = []
        for _ in range(n_groups):
            root_index = len(roles)
            roles.append(root_role)
            for child_role in child_roles:
                child_index = len(roles)
                roles.append(child_role)
                edges.append((root_index, child_index))
        return roles, edges, leftover

    def _resolve_structured_role_counts(self) -> dict[str, int]:
        allocation = self.config.modality_allocation
        if allocation is None:
            allocation = {
                "shared_all": 1.0 / 3.0,
                "pairwise": 1.0 / 3.0,
                "private": 1.0 / 3.0,
            }

        alloc = {
            str(key).strip().lower(): float(value)
            for key, value in dict(allocation).items()
        }
        unknown = set(alloc) - _ALLOCATION_KEYS
        if unknown:
            raise ValueError(
                f"Unknown modality_allocation keys {sorted(unknown)}. "
                f"Expected a subset of {sorted(_ALLOCATION_KEYS)}."
            )
        if any(value < 0 for value in alloc.values()):
            raise ValueError(f"modality_allocation values must be non-negative, got {alloc}")

        if "shared_all" in alloc and "abc" in alloc:
            raise ValueError("Use only one of modality_allocation keys 'shared_all' and 'abc'.")
        if "pairwise" in alloc and any(k in alloc for k in ("ab", "ac", "bc")):
            raise ValueError(
                "Use either grouped key 'pairwise' or fine-grained keys 'ab'/'ac'/'bc', not both."
            )
        if "private" in alloc and any(k in alloc for k in ("a", "b", "c")):
            raise ValueError(
                "Use either grouped key 'private' or fine-grained keys 'a'/'b'/'c', not both."
            )

        shared = alloc.get("shared_all", alloc.get("abc", 0.0))
        use_counts = self._allocation_values_are_counts(alloc)

        if use_counts:
            shared_count = int(round(shared))
            if "pairwise" in alloc:
                pair_counts = self._split_grouped_count(int(round(alloc["pairwise"])), ("ab", "ac", "bc"))
            else:
                pair_counts = {
                    "ab": int(round(alloc.get("ab", 0.0))),
                    "ac": int(round(alloc.get("ac", 0.0))),
                    "bc": int(round(alloc.get("bc", 0.0))),
                }
            if "private" in alloc:
                private_counts = self._split_grouped_count(int(round(alloc["private"])), ("a", "b", "c"))
            else:
                private_counts = {
                    "a": int(round(alloc.get("a", 0.0))),
                    "b": int(round(alloc.get("b", 0.0))),
                    "c": int(round(alloc.get("c", 0.0))),
                }
            unused_count = int(round(alloc["unused"])) if "unused" in alloc else None
            assigned = shared_count + sum(pair_counts.values()) + sum(private_counts.values())
            if unused_count is None:
                unused_count = self.config.n_latents - assigned
            if unused_count < 0 or assigned + unused_count != self.config.n_latents:
                raise ValueError(
                    "Structured count allocation must sum to n_latents. "
                    f"Got shared={shared_count}, pairwise={pair_counts}, "
                    f"private={private_counts}, unused={unused_count}, "
                    f"n_latents={self.config.n_latents}."
                )
            return {
                "shared_all": shared_count,
                **pair_counts,
                **private_counts,
                "unused": unused_count,
            }

        # Fraction mode.
        if "pairwise" in alloc:
            pair_fracs = self._split_grouped_fraction(float(alloc["pairwise"]), ("ab", "ac", "bc"))
        else:
            pair_fracs = {
                "ab": float(alloc.get("ab", 0.0)),
                "ac": float(alloc.get("ac", 0.0)),
                "bc": float(alloc.get("bc", 0.0)),
            }
        if "private" in alloc:
            private_fracs = self._split_grouped_fraction(float(alloc["private"]), ("a", "b", "c"))
        else:
            private_fracs = {
                "a": float(alloc.get("a", 0.0)),
                "b": float(alloc.get("b", 0.0)),
                "c": float(alloc.get("c", 0.0)),
            }

        role_keys = ["shared_all", "ab", "ac", "bc", "a", "b", "c", "unused"]
        core_fracs = [
            float(shared),
            pair_fracs["ab"],
            pair_fracs["ac"],
            pair_fracs["bc"],
            private_fracs["a"],
            private_fracs["b"],
            private_fracs["c"],
        ]
        if "unused" in alloc:
            fracs = core_fracs + [float(alloc["unused"])]
            specified = sum(fracs)
            if specified > 1.0 + 1e-6:
                raise ValueError(
                    f"modality_allocation fractions must sum to <= 1, got {specified}."
                )
        else:
            specified = sum(core_fracs)
            if specified > 1.0 + 1e-6:
                raise ValueError(
                    f"modality_allocation fractions must sum to <= 1, got {specified}."
                )
            fracs = core_fracs + [max(0.0, 1.0 - specified)]
        fracs = [max(0.0, float(value)) for value in fracs]

        counts = _fractions_to_counts(fracs, self.config.n_latents)
        return {role: count for role, count in zip(role_keys, counts)}

    def _allocation_values_are_counts(self, alloc: dict[str, float]) -> bool:
        values = list(alloc.values())
        if not values:
            return False
        all_integral = all(abs(value - round(value)) <= 1e-6 for value in values)
        if not all_integral:
            return False
        total = sum(values)
        # Fractions live in [0, 1] and usually sum to <= 1 (e.g. shared_all=1.0 => 100%).
        # Counts either exceed 1 in a single entry, sum to more than 1 across roles, or
        # explicitly include unused.
        return (
            max(values) > 1.0 + 1e-6
            or total > 1.0 + 1e-6
            or "unused" in alloc
        )

    def _split_grouped_count(self, total: int, keys: tuple[str, ...]) -> dict[str, int]:
        if total < 0:
            raise ValueError(f"Grouped count must be non-negative, got {total}")
        base = total // len(keys)
        counts = {key: base for key in keys}
        remainder = total - base * len(keys)
        if remainder > 0:
            order = torch.randperm(len(keys), generator=self.generator).tolist()
            for index in order[:remainder]:
                counts[keys[index]] += 1
        return counts

    def _split_grouped_fraction(
        self,
        total: float,
        keys: tuple[str, ...],
    ) -> dict[str, float]:
        if total < 0:
            raise ValueError(f"Grouped fraction must be non-negative, got {total}")
        share = total / len(keys)
        return {key: share for key in keys}

    @staticmethod
    def _role_from_mask(mask_column: Tensor) -> str:
        pattern = tuple(bool(x) for x in mask_column.tolist())
        for role, role_pattern in _ROLE_MASKS.items():
            if role == "abc":
                continue
            if role_pattern == pattern:
                return role
        return "unused"

    def _sample_modality_projections(self) -> list[list[Tensor]]:
        projections: list[list[Tensor]] = []

        for modality in range(self.config.n_modalities):
            modality_projections = []

            for latent in range(self.config.n_latents):
                projection = self._randn(
                    self.config.latent_dim,
                    self.config.embedding_dim,
                ) / self.config.latent_dim**0.5

                modality_projections.append(projection)

            projections.append(modality_projections)

        return projections

    def _stack_modality_projections(self) -> Tensor:
        if self.modality_projections is None:
            raise RuntimeError("Linear modality projections are not initialized.")
        return torch.stack(
            [
                torch.stack(modality_projections, dim=0)
                for modality_projections in self.modality_projections
            ],
            dim=0,
        )

    def _sample_modality_mlps(self) -> list[ModalityMLP]:
        input_dim = self.config.n_latents * self.config.latent_dim
        hidden_dim = self.config.resolved_modality_mlp_hidden_dim()
        mlps: list[ModalityMLP] = []
        for _ in range(self.config.n_modalities):
            mlps.append(
                ModalityMLP(
                    weight_1=self._randn(input_dim, hidden_dim) / input_dim**0.5,
                    bias_1=0.1 * self._randn(hidden_dim),
                    weight_2=self._randn(hidden_dim, self.config.embedding_dim)
                    / hidden_dim**0.5,
                    bias_2=0.1 * self._randn(self.config.embedding_dim),
                )
            )
        return mlps

    def sample_latents(self, n_samples: int) -> list[Tensor]:
        latents: dict[int, Tensor] = {}

        for latent_index in self.topological_order.tolist():
            exogenous_noise = self._randn(
                n_samples,
                self.config.latent_dim,
            )

            equation = self.structural_equations[latent_index]

            if equation is None:
                latent_value = exogenous_noise
            else:
                parent_values = [
                    latents[parent]
                    for parent in equation.parents
                ]

                deterministic_part = equation(parent_values)

                latent_value = (
                    deterministic_part
                    + self.config.latent_noise_std * exogenous_noise
                )

            latents[latent_index] = latent_value

        return [latents[index] for index in range(self.config.n_latents)]

    def sample_modalities(
        self,
        latents: list[Tensor],
    ) -> list[Tensor]:

        n_samples = latents[0].shape[0]
        latent_tensor = torch.stack(latents, dim=1)
        mask = self.modality_masks.to(
            device=latent_tensor.device,
            dtype=latent_tensor.dtype,
        )
        alpha = float(self.modality_readout_alpha)

        linear_modalities = None
        mlp_modalities = None
        if alpha < 1.0 - 1e-12:
            linear_modalities = self._linear_modality_readout(latent_tensor, mask)
        if alpha > 1e-12:
            mlp_modalities = self._mlp_modality_readout(
                latent_tensor, mask, n_samples=n_samples
            )

        if linear_modalities is None:
            modalities = mlp_modalities
        elif mlp_modalities is None:
            modalities = linear_modalities
        else:
            modalities = (1.0 - alpha) * linear_modalities + alpha * mlp_modalities

        modality_noise = torch.stack(
            [
                self._randn(n_samples, self.config.embedding_dim)
                for _ in range(self.config.n_modalities)
            ],
            dim=0,
        ).to(device=modalities.device, dtype=modalities.dtype)
        modality_noise_scale = torch.tensor(
            self.config.resolved_modality_noise_std(),
            device=modalities.device,
            dtype=modalities.dtype,
        )[:, None, None]
        modality_noise = modality_noise * modality_noise_scale

        modalities = F.normalize(modalities + modality_noise, dim=-1)

        return [modalities[index] for index in range(self.config.n_modalities)]

    def _linear_modality_readout(self, latent_tensor: Tensor, mask: Tensor) -> Tensor:
        if self.modality_projection_tensor is None:
            raise RuntimeError("Linear modality projections are not initialized.")
        projections = self.modality_projection_tensor.to(
            device=latent_tensor.device,
            dtype=latent_tensor.dtype,
        )
        return torch.einsum(
            "nkl,mkle,mk->mne",
            latent_tensor,
            projections,
            mask,
        )

    def _mlp_modality_readout(
        self,
        latent_tensor: Tensor,
        mask: Tensor,
        n_samples: int,
    ) -> Tensor:
        if self.modality_mlps is None:
            raise RuntimeError("MLP modality readouts are not initialized.")
        # Zero masked latents, then concat: (n, m, k*d)
        masked = latent_tensor[:, None, :, :] * mask[None, :, :, None]
        flat = masked.reshape(
            n_samples,
            self.config.n_modalities,
            self.config.n_latents * self.config.latent_dim,
        )
        modality_outputs = []
        for modality in range(self.config.n_modalities):
            mlp = self.modality_mlps[modality]
            hidden = F.gelu(
                flat[:, modality] @ mlp.weight_1.to(
                    device=flat.device, dtype=flat.dtype
                )
                + mlp.bias_1.to(device=flat.device, dtype=flat.dtype)
            )
            modality_outputs.append(
                hidden @ mlp.weight_2.to(device=flat.device, dtype=flat.dtype)
                + mlp.bias_2.to(device=flat.device, dtype=flat.dtype)
            )
        return torch.stack(modality_outputs, dim=0)

    def generate_dataset(
        self,
        n_samples: int,
    ) -> dict[str, Tensor]:

        latents = self.sample_latents(n_samples)
        modalities = self.sample_modalities(latents)

        output = {
            f"modality_{index}": modality
            for index, modality in enumerate(modalities)
        }

        output.update({
            f"latent_{index}": latent
            for index, latent in enumerate(latents)
        })

        return output

    def describe(self) -> dict:
        return {
            "seed": self.scm_seed,
            "n_latents": self.config.n_latents,
            "latent_dim": self.config.latent_dim,
            "latent_noise_std": self.config.latent_noise_std,
            "modality_noise_std": self.config.resolved_modality_noise_std(),
            "edge_probability": self.config.edge_probability,
            "shared_modality_projections": self.config.shared_modality_projections,
            "modality_readout": self.modality_readout,
            "modality_readout_alpha": self.modality_readout_alpha,
            "adjacency": self.adjacency.clone(),
            "topological_order": self.topological_order.clone(),
            "modality_masks": self.modality_masks.clone(),
            "latent_roles": list(self.latent_roles),
            "modality_mask_mode": self.config.resolved_modality_mask_mode(),
            "semantic_root_fraction": self.config.semantic_root_fraction,
            "semantic_child_edges": list(self.semantic_child_edges),
            "mechanisms": [
                equation.mechanism_type.value
                if equation is not None
                else "exogenous"
                for equation in self.structural_equations
            ],
        }


class SCMPrior:
    """
    Samples multiple datasets from multiple independently sampled SCMs.
    """

    def __init__(self, config: SCMConfig):
        self.config = config
        self.shared_modality_projection_cache: dict[
            tuple[int, int, int, int], list[list[Tensor]]
        ] = {}
        self.shared_modality_mlp_cache: dict[
            tuple[int, int, int, int, int], list[ModalityMLP]
        ] = {}

    def _shared_projection_generator(self, config: SCMConfig) -> torch.Generator:
        base_seed = (
            config.seed + 123_456
            if config.shared_projection_seed is None
            else int(config.shared_projection_seed)
        )
        seed = (
            int(base_seed)
            + 1_000_003 * int(config.n_latents)
            + 10_007 * int(config.latent_dim)
            + 101 * int(config.embedding_dim)
            + int(config.resolved_modality_mlp_hidden_dim())
        )
        return torch.Generator().manual_seed(seed)

    def _sample_shared_modality_projections(self, config: SCMConfig) -> list[list[Tensor]]:
        generator = self._shared_projection_generator(config)
        projections: list[list[Tensor]] = []

        for modality in range(config.n_modalities):
            modality_projections = []
            for latent in range(config.n_latents):
                projection = torch.randn(
                    config.latent_dim,
                    config.embedding_dim,
                    generator=generator,
                ) / config.latent_dim**0.5
                modality_projections.append(projection)
            projections.append(modality_projections)

        return projections

    def _sample_shared_modality_mlps(self, config: SCMConfig) -> list[ModalityMLP]:
        generator = self._shared_projection_generator(config)
        input_dim = config.n_latents * config.latent_dim
        hidden_dim = config.resolved_modality_mlp_hidden_dim()
        mlps: list[ModalityMLP] = []
        for _ in range(config.n_modalities):
            mlps.append(
                ModalityMLP(
                    weight_1=torch.randn(
                        input_dim, hidden_dim, generator=generator
                    )
                    / input_dim**0.5,
                    bias_1=0.1
                    * torch.randn(hidden_dim, generator=generator),
                    weight_2=torch.randn(
                        hidden_dim,
                        config.embedding_dim,
                        generator=generator,
                    )
                    / hidden_dim**0.5,
                    bias_2=0.1
                    * torch.randn(
                        config.embedding_dim, generator=generator
                    ),
                )
            )
        return mlps

    def _choice(self, values, generator: torch.Generator, name: str):
        if not values:
            raise ValueError(f"{name} must be non-empty when sampled mode is active.")
        index = int(torch.randint(len(values), size=(1,), generator=generator).item())
        return values[index]

    def _sample_effective_config(self, scm_index: int) -> SCMConfig:
        generator = torch.Generator().manual_seed(
            int(self.config.seed) + 9_999_937 * int(scm_index) + 31_337
        )

        updates = {}
        if self.config.resolved_parameter_sampling_mode() == "sampled":
            updates["n_latents"] = int(
                self._choice(
                    self.config.n_latents_choices or [8, 16, 32, 64],
                    generator,
                    "n_latents_choices",
                )
            )
            updates["latent_dim"] = int(
                self._choice(
                    self.config.latent_dim_choices or [2, 4, 8, 16],
                    generator,
                    "latent_dim_choices",
                )
            )
            updates["modality_readout_alpha"] = float(
                self._choice(
                    self.config.modality_readout_alpha_choices
                    or [0.2, 0.4, 0.6, 0.8],
                    generator,
                    "modality_readout_alpha_choices",
                )
            )
            updates["edge_probability"] = float(
                self._choice(
                    self.config.edge_probability_choices or [0.2, 0.3, 0.4],
                    generator,
                    "edge_probability_choices",
                )
            )
            updates["modality_noise_std"] = self._sample_modality_noise_std(generator)

        if self.config.resolved_modality_allocation_mode() == "sampled":
            allocation_choices = self.config.modality_allocation_choices or [
                {"shared_all": 0.8, "pairwise": 0.1, "private": 0.1},
                {"shared_all": 0.5, "pairwise": 0.25, "private": 0.25},
                {"shared_all": 0.25, "pairwise": 0.375, "private": 0.375},
                {"shared_all": 0.1, "pairwise": 0.45, "private": 0.45},
            ]
            updates["modality_allocation"] = {
                str(key): float(value)
                for key, value in dict(
                    self._choice(
                        allocation_choices,
                        generator,
                        "modality_allocation_choices",
                    )
                ).items()
            }

        if self.config.resolved_mechanism_types_mode() == "sampled":
            mechanism_type_choices = self.config.mechanism_type_choices or [
                ["linear", "tanh", "sine"],
                ["linear", "relu", "gelu"],
                ["tanh", "sine", "relu", "gelu"],
                ["linear", "tanh", "sine", "relu", "gelu", "mlp"],
                ["mlp", "linear", "gelu"],
            ]
            updates["mechanism_types"] = [
                str(name)
                for name in self._choice(
                    mechanism_type_choices,
                    generator,
                    "mechanism_type_choices",
                )
            ]

        return replace(self.config, **updates)

    def _sample_modality_noise_std(self, generator: torch.Generator) -> list[float]:
        choices = self.config.modality_noise_std_choices or [0.2, 0.3, 0.4, 0.5]
        if not choices:
            raise ValueError("modality_noise_std_choices must be non-empty in sampled mode.")

        first = choices[0]
        if isinstance(first, (list, tuple)):
            sampled = self._choice(choices, generator, "modality_noise_std_choices")
            values = [float(value) for value in sampled]
        else:
            values = [
                float(self._choice(choices, generator, "modality_noise_std_choices"))
                for _ in range(int(self.config.n_modalities))
            ]

        if len(values) != int(self.config.n_modalities):
            raise ValueError(
                f"Sampled modality noise must have length {self.config.n_modalities}, got {values}."
            )
        return values

    def _shared_projection_key(self, config: SCMConfig) -> tuple[int, int, int, int]:
        return (
            int(config.n_modalities),
            int(config.n_latents),
            int(config.latent_dim),
            int(config.embedding_dim),
        )

    def _shared_mlp_key(self, config: SCMConfig) -> tuple[int, int, int, int, int]:
        return (
            int(config.n_modalities),
            int(config.n_latents),
            int(config.latent_dim),
            int(config.embedding_dim),
            int(config.resolved_modality_mlp_hidden_dim()),
        )

    def _shared_components(
        self,
        config: SCMConfig,
    ) -> tuple[list[list[Tensor]] | None, list[ModalityMLP] | None]:
        modality_projections = None
        modality_mlps = None
        if config.shared_modality_projections:
            if config.uses_linear_readout():
                projection_key = self._shared_projection_key(config)
                if projection_key not in self.shared_modality_projection_cache:
                    self.shared_modality_projection_cache[projection_key] = (
                        self._sample_shared_modality_projections(config)
                    )
                modality_projections = self.shared_modality_projection_cache[projection_key]
            if config.uses_mlp_readout():
                mlp_key = self._shared_mlp_key(config)
                if mlp_key not in self.shared_modality_mlp_cache:
                    self.shared_modality_mlp_cache[mlp_key] = (
                        self._sample_shared_modality_mlps(config)
                    )
                modality_mlps = self.shared_modality_mlp_cache[mlp_key]
        return modality_projections, modality_mlps

    def sample_scm(self, scm_index: int) -> SampledSCM:
        effective_config = self._sample_effective_config(scm_index)
        modality_projections, modality_mlps = self._shared_components(effective_config)
        scm_seed = effective_config.seed + 10_000 * scm_index
        return SampledSCM(
            config=effective_config,
            scm_seed=scm_seed,
            modality_projections=modality_projections,
            modality_mlps=modality_mlps,
        )

    def sample_dataset(
        self,
        scm_index: int,
        n_samples: int,
    ) -> tuple[dict[str, Tensor], dict]:

        scm = self.sample_scm(scm_index)
        dataset = scm.generate_dataset(n_samples)
        metadata = scm.describe()

        return dataset, metadata


if __name__ == "__main__":
    config = SCMConfig(
        n_modalities=3,
        n_latents=8,
        latent_dim=4,
        embedding_dim=32,
    )

    prior = SCMPrior(config)

    for scm_index in range(3):
        dataset, metadata = prior.sample_dataset(
            scm_index=scm_index,
            n_samples=5_000,
        )

        print(f"\nSCM {scm_index}")
        print("Mechanisms:", metadata["mechanisms"])
        print("Modality masks:")
        print(metadata["modality_masks"])

        for modality_index in range(config.n_modalities):
            print(
                f"modality_{modality_index}:",
                dataset[f"modality_{modality_index}"].shape,
            )
