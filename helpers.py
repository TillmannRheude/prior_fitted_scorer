import torch.nn as nn
import torch
import pickle

def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)


def _init_linear_near_identity_(layer: nn.Linear, noise_scale: float = 1e-3) -> None:
    """
    Initialize a Linear layer as the closest identity-like map allowed by its
    shape, plus tiny noise.
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


def _load_torch_checkpoint(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        # Lightning checkpoints can include OmegaConf objects in hyperparameters,
        # which are rejected by PyTorch's weights_only loader. The checkpoint is
        # produced locally by this project, so fall back to normal checkpoint load.
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_scorer_state_dict(checkpoint: dict, target_state_dict: dict) -> dict:
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Expected checkpoint or checkpoint['state_dict'] to be a state dict.")

    for prefix in (
        "model.transformer.",
        "model.scorer.",
        "transformer.",
        "scorer.",
    ):
        scorer_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if scorer_state:
            return scorer_state

    return {
        key: value
        for key, value in state_dict.items()
        if key in target_state_dict
    }


def _get_scorer(lightning_module):
    model = getattr(lightning_module, "model", None)
    scorer = getattr(model, "scorer", None)
    if scorer is None:
        scorer = getattr(model, "transformer", None)
    return scorer


def _maybe_load_pretrained_scorer(lightning_module, cfg: dict):
    model_cfg = cfg.get("modelname", {})
    checkpoint_path = model_cfg.get("pretrained_scorer_ckpt", None)
    if checkpoint_path in [None, "None", "null", ""]:
        return lightning_module

    if cfg["modelname"]["modelname"] not in {"transformer", "mlp", "barybind"}:
        raise ValueError(
            "pretrained_scorer_ckpt requires a learned scorer."
        )

    scorer = _get_scorer(lightning_module)
    if scorer is None:
        raise ValueError("Could not locate the learned scorer for pretraining.")

    checkpoint = _load_torch_checkpoint(str(checkpoint_path))
    scorer_state = _extract_scorer_state_dict(checkpoint, scorer.state_dict())
    if not scorer_state:
        raise ValueError(f"No learned scorer weights found in checkpoint: {checkpoint_path}")

    strict = _as_bool(model_cfg.get("pretrained_scorer_strict", True))
    skipped = []
    if not strict:
        target_state = scorer.state_dict()
        filtered_state = {}
        for key, value in scorer_state.items():
            if key not in target_state:
                skipped.append((key, "unexpected"))
                continue
            if tuple(value.shape) != tuple(target_state[key].shape):
                skipped.append((key, f"shape {tuple(value.shape)} != {tuple(target_state[key].shape)}"))
                continue
            filtered_state[key] = value
        scorer_state = filtered_state
        if not scorer_state:
            raise ValueError(
                f"No shape-compatible scorer weights found in checkpoint: {checkpoint_path}"
            )

    incompatible = scorer.load_state_dict(scorer_state, strict=strict)

    if _as_bool(model_cfg.get("pretrained_load_logit_scale", False)):
        state_dict = checkpoint.get("state_dict", checkpoint)
        if "logit_scale" not in state_dict:
            raise ValueError("pretrained_load_logit_scale=True, but checkpoint has no 'logit_scale'.")
        if getattr(lightning_module, "logit_scale", None) is None:
            raise ValueError("pretrained_load_logit_scale=True, but target module has no logit_scale parameter.")
        with torch.no_grad():
            lightning_module.logit_scale.copy_(state_dict["logit_scale"].detach().cpu())

    print(
        f"Loaded pretrained scorer from {checkpoint_path} "
        f"({len(scorer_state)} tensors, strict={strict})."
    )
    if not strict:
        print(
            "Pretrained scorer load incompatibilities: "
            f"missing={list(incompatible.missing_keys)}, unexpected={list(incompatible.unexpected_keys)}"
        )
        if skipped:
            print(f"Skipped {len(skipped)} incompatible pretrained scorer tensors: {skipped}")
    return lightning_module


def _maybe_freeze_scorer(lightning_module, cfg: dict):
    """Freeze learned scorer weights when modelname.freeze_scorer=true."""
    if not _as_bool(cfg.get("modelname", {}).get("freeze_scorer", False)):
        return lightning_module
    if cfg["modelname"]["modelname"] == "barybind":
        raise ValueError(
            "BaryBind jointly trains its WB maps, potentials, and matching head; "
            "freeze_scorer is therefore unsupported."
        )

    scorer = _get_scorer(lightning_module)
    if scorer is None:
        raise ValueError("freeze_scorer=True, but no learned scorer was found.")

    n_params = 0
    for param in scorer.parameters():
        param.requires_grad_(False)
        n_params += param.numel()
    print(f"Froze learned scorer ({n_params} parameters, requires_grad=False).")
    return lightning_module


def _finalize_learned_scorer(lightning_module, cfg: dict):
    lightning_module = _maybe_load_pretrained_scorer(lightning_module, cfg)
    return _maybe_freeze_scorer(lightning_module, cfg)


def build_model(cfg: dict):
    from architecture import Contrastive_Model, MLPScorer_Model, TransformerScorer_Model
    from losses.barybind import BaryBindModel
    from encoders import (
        CXREncoder,
        ECGEncoder,
        IdentityEncoder,
        InspectVisionEncoder,
        LabsEncoder,
        MimicMLPEncoder,
        MimicRadBERTEncoder,
        UKBTabularEncoder,
    )
    from lightningmodules.symile_mimic import SymileMIMICModel
    from lightningmodules.ukb import UKBModel
    from lightningmodules.mimic import MIMICModel
    from lightningmodules.scm import SCMPretrainModel
    from lightningmodules.inspect import INSPECTModel
    from datasets.symile_mimic import Dataset_SymileMimic

    params_optimizer = {
        "name": cfg["optimizer"]["name"],
        "lr": cfg["optimizer"]["lr"],
        "weight_decay": cfg["optimizer"]["weight_decay"],
        "eps": cfg["optimizer"]["eps"],
        "betas": cfg["optimizer"]["betas"],
        "warmup_steps": cfg["optimizer"]["warmup_steps"],
        "scorer_lr": cfg["modelname"].get("scorer_lr", None),
        "scorer_warmup_start_epoch": cfg["modelname"].get("scorer_warmup_start_epoch", 0),
        "scorer_warmup_epochs": cfg["modelname"].get("scorer_warmup_epochs", 0),
        "scorer_warmup_curve": cfg["modelname"].get("scorer_warmup_curve", "cosine"),
        "potential_lr_multiplier": cfg["modelname"].get(
            "barybind_params", {}
        ).get("potential_lr_multiplier", 2.0),
    }
    params_method = {
        "modelname": cfg["modelname"]["modelname"],
        "negative_sampling": cfg["modelname"]["negative_sampling"],
        "candidate_dependent": cfg["modelname"].get("candidate_dependent", True),
        "logit_scale_init": cfg["modelname"]["logit_scale_init"],
        "batch_size": cfg["batch_size"],
        "embedding_norm": cfg["modelname"]["embedding_norm"],
        "bias_init_mult": cfg["modelname"]["bias_init_mult"] if "bias_init_mult" in cfg["modelname"] else 1.0,
        "pair_num_negatives": cfg["modelname"]["pair_num_negatives"],
        "validate_retrieval": _as_bool(cfg.get("validate_retrieval", True)),
    }
    params_retrival_ds = {
        "batch_size": cfg["batch_size"],
        "split_nr": cfg["split_nr"],
        "retrieval_mode": cfg.get("encoders", {}).get("retrieval_mode", "global"),
        "preselected_num_candidates": cfg.get("encoders", {}).get("preselected_num_candidates", 10),
        "retrieval_seed": cfg.get("seed", 420),
        "data_dir": cfg.get("datamodule", {}).get("data_dir", Dataset_SymileMimic.DEFAULT_DATA_DIR),
        "num_workers": cfg.get("datamodule", {}).get("num_workers", 4),
        "text_model_id": cfg.get("encoders", {}).get("report", {}).get(
            "text_model_id", "StanfordAIMI/RadBERT"
        ),
        "max_text_length": cfg.get("encoders", {}).get("report", {}).get(
            "max_text_length", 256
        ),
        "same_label_negatives": cfg.get("encoders", {}).get(
            "same_label_negatives", False
        ),
    }
    modelname = cfg["modelname"]["modelname"]

    def build_scorer_model(contrastive_model, max_modalities=3):
        if modelname == "barybind":
            if max_modalities != 3:
                raise ValueError("BaryBind currently supports exactly three modalities.")
            barybind_params = dict(
                cfg["modelname"].get("barybind_params", {})
            )
            barybind_params["retrieval_anchor_idx"] = int(
                cfg.get("encoders", {}).get("candidate_idx", 0)
            )
            return BaryBindModel(
                contrastive_model=contrastive_model,
                params=barybind_params,
            )
        if modelname == "transformer":
            return TransformerScorer_Model(
                contrastive_model=contrastive_model,
                transformer_params=cfg["modelname"]["transformer_params"],
                proj_output_dim=1,
                candidate_dependent=cfg["modelname"].get(
                    "candidate_dependent", True
                ),
            )
        if modelname == "mlp":
            return MLPScorer_Model(
                contrastive_model=contrastive_model,
                mlp_params=cfg["modelname"]["mlp_params"],
                proj_output_dim=1,
                max_modalities=max_modalities,
            )
        return contrastive_model

    # SCM scorer-only pretraining
    if cfg["dataset_name"] == "scm_pretrain":
        if modelname == "barybind":
            raise ValueError(
                "BaryBind is an end-to-end real-data objective and is not "
                "supported by the scorer-only SCM pretraining pipeline."
            )
        n_modalities = int(cfg["encoders"].get("n_modalities", 3))
        if n_modalities != 3:
            raise ValueError("scm_pretrain currently supports exactly three modalities.")

        embedding_dim = int(cfg["encoders"]["embedding_dim"])
        model_emb_dim = int(cfg["modelname"]["emb_dim"])
        if embedding_dim != model_emb_dim:
            raise ValueError(
                "For scm_pretrain, encoders.embedding_dim must match modelname.emb_dim "
                f"because IdentityEncoder forwards SCM embeddings directly. Got {embedding_dim} and {model_emb_dim}."
            )

        encoders = nn.ModuleList(
            [IdentityEncoder(emb_dim=embedding_dim) for _ in range(n_modalities)]
        )
        model = build_scorer_model(
            Contrastive_Model(encoders=encoders),
            max_modalities=n_modalities,
        )

        lightning_module = SCMPretrainModel(
            params_optimizer=params_optimizer,
            params_method=params_method,
            model=model,
            modelname=modelname,
            params_retrival_ds=params_retrival_ds,
        )
        return _finalize_learned_scorer(lightning_module, cfg)

    # Symile-MIMIC
    if cfg["dataset_name"] == "symile_mimic":
        encoders = nn.ModuleList([
            CXREncoder(
                resnet_params=cfg["encoders"].get("cxr", {}),
                emb_dim=cfg["modelname"]["emb_dim"],
            ),
            ECGEncoder(
                resnet_params=cfg["encoders"].get("ecg", {}),
                emb_dim=cfg["modelname"]["emb_dim"],
            ),
            LabsEncoder(
                emb_dim=cfg["modelname"]["emb_dim"],
            ),
        ])
        model = build_scorer_model(Contrastive_Model(encoders=encoders))

        lightning_module = SymileMIMICModel(
            params_optimizer=params_optimizer,
            params_method=params_method,
            model=model,
            modelname=modelname,
            params_retrival_ds=params_retrival_ds,
            candidate_idx=cfg["encoders"].get("candidate_idx", 0),
        )
        return _finalize_learned_scorer(lightning_module, cfg)

    if cfg["dataset_name"] == "ukb":
        input_dims_modalities = {
            "nmr": 249,
            "ehr": 3584,
            "olink": 1463,
        }
        modalities = cfg["encoders"]["modalities"]
        encoders = nn.ModuleList([])

        for mod in modalities:
            encoders.append(
                UKBTabularEncoder(
                    input_dim=input_dims_modalities[mod],
                    hidden_dims=cfg["encoders"][mod]["mlp"]["hidden_dims"],
                    hidden_dropouts=cfg["encoders"][mod]["mlp"]["hidden_dropouts"],
                    emb_dim=cfg["modelname"]["emb_dim"],
                    combine_eids_as=cfg["datamodule"]["combine_eids_as"],
                    modality_name=mod,
                )
            )
        model = build_scorer_model(
            Contrastive_Model(encoders=encoders),
            max_modalities=len(encoders),
        )

        lightning_module = UKBModel(
            params_optimizer=params_optimizer,
            params_method=params_method,
            model=model,
            candidate_idx=cfg["encoders"]["candidate_idx"],
            modelname=modelname,
            params_retrival_ds=params_retrival_ds,
            modalities=cfg["encoders"]["modalities"],
        )
        return _finalize_learned_scorer(lightning_module, cfg)
    
    # MIMIC (new trimodal variant)
    if cfg["dataset_name"] == "mimic":
        import pandas as pd
        timeseries_input_dim = int(cfg["encoders"]["timeseries"]["input_dim"])
        if timeseries_input_dim <= 0:
            probe_df = pd.read_csv(cfg["encoders"]["train_split_csv"], nrows=1)
            timeseries_input_dim = int(len([c for c in probe_df.columns if c.startswith("ts_item")]))
            if timeseries_input_dim <= 0:
                raise ValueError("Could not infer timeseries input_dim from train_split_csv; please set encoders.timeseries.input_dim.")

        ts_hidden_dims = cfg["encoders"]["timeseries"]["mlp"]["hidden_dims"]
        ts_hidden_dropouts = cfg["encoders"]["timeseries"]["mlp"]["hidden_dropouts"]
        demo_hidden_dims = cfg["encoders"]["demographics"]["mlp"]["hidden_dims"]
        demo_hidden_dropouts = cfg["encoders"]["demographics"]["mlp"]["hidden_dropouts"]
        if len(ts_hidden_dims) != len(ts_hidden_dropouts):
            raise ValueError("mimic.timeseries.mlp.hidden_dims and hidden_dropouts must have the same length.")
        if len(demo_hidden_dims) != len(demo_hidden_dropouts):
            raise ValueError("mimic.demographics.mlp.hidden_dims and hidden_dropouts must have the same length.")

        encoders = nn.ModuleList([
            MimicRadBERTEncoder(
                model_params=cfg["encoders"].get("radiology", {}),
                emb_dim=cfg["modelname"]["emb_dim"],
            ),
            MimicMLPEncoder(
                input_dim=timeseries_input_dim,
                hidden_dims=ts_hidden_dims,
                hidden_dropouts=ts_hidden_dropouts,
                emb_dim=cfg["modelname"]["emb_dim"],
            ),
            MimicMLPEncoder(
                input_dim=cfg["encoders"]["demographics"]["input_dim"],
                hidden_dims=demo_hidden_dims,
                hidden_dropouts=demo_hidden_dropouts,
                emb_dim=cfg["modelname"]["emb_dim"],
            ),
        ])
        model = build_scorer_model(Contrastive_Model(encoders=encoders))
        lightning_module = MIMICModel(
            params_optimizer=params_optimizer,
            params_method=params_method,
            model=model,
            modelname=modelname,
            params_retrival_ds=params_retrival_ds,
            candidate_idx=cfg["encoders"].get("candidate_idx", 0),
        )
        return _finalize_learned_scorer(lightning_module, cfg)

    # INSPECT CTPA vision, MOTOR EHR, and radiology reports.
    if cfg["dataset_name"] == "inspect":
        vision_cfg = cfg["encoders"]["vision"]
        ehr_cfg = cfg["encoders"]["ehr"]
        encoders = nn.ModuleList([
            InspectVisionEncoder(
                input_dim=int(vision_cfg["input_dim"]),
                hidden_dim=int(vision_cfg.get("hidden_dim", 64)),
                num_blocks=int(vision_cfg.get("num_blocks", 1)),
                emb_dim=cfg["modelname"]["emb_dim"],
            ),
            MimicMLPEncoder(
                input_dim=int(ehr_cfg["input_dim"]),
                hidden_dims=ehr_cfg["mlp"]["hidden_dims"],
                hidden_dropouts=ehr_cfg["mlp"]["hidden_dropouts"],
                emb_dim=cfg["modelname"]["emb_dim"],
                branch_initialization=ehr_cfg.get(
                    "branch_initialization", "scaled_final"
                ),
                branch_scale=float(ehr_cfg.get("branch_scale", 1e-3)),
            ),
            MimicRadBERTEncoder(
                model_params=cfg["encoders"]["report"],
                emb_dim=cfg["modelname"]["emb_dim"],
            ),
        ])
        model = build_scorer_model(Contrastive_Model(encoders=encoders))
        lightning_module = INSPECTModel(
            params_optimizer=params_optimizer,
            params_method=params_method,
            model=model,
            modelname=modelname,
            params_retrival_ds=params_retrival_ds,
            candidate_idx=cfg["encoders"].get("candidate_idx", 0),
        )
        return _finalize_learned_scorer(lightning_module, cfg)

    raise ValueError(f"Model {cfg['dataset_name']} not implemented.")



def build_datamodule(cfg: dict):
    from datasets.symile_mimic import Dataset_SymileMimic
    from lightningdatamodules.symile_mimic import DataModule_SymileMimic
    from lightningdatamodules.mimic import DataModule_MIMIC
    from lightningdatamodules.inspect import INSPECTDataModule
    from lightningdatamodules.scm import DataModule_SCMPretrain
    from scm.framework import SCMConfig

    # SCM scorer-only pretraining
    if cfg["dataset_name"] == "scm_pretrain":
        dm_cfg = cfg.get("datamodule", {})
        enc_cfg = cfg.get("encoders", {})
        mechanism_types = enc_cfg.get("mechanism_types", None)
        if mechanism_types is not None:
            mechanism_types = [str(name) for name in mechanism_types]
        mechanism_type_choices = enc_cfg.get("mechanism_type_choices", None)
        if mechanism_type_choices is not None:
            mechanism_type_choices = [
                [str(name) for name in mechanism_set]
                for mechanism_set in mechanism_type_choices
            ]

        allocation_presets = {
            "shared50_pair25_private25": {
                "shared_all": 0.5,
                "pairwise": 0.25,
                "private": 0.25,
            },
            "shared25_pair375_private375": {
                "shared_all": 0.25,
                "pairwise": 0.375,
                "private": 0.375,
            },
            "shared20_pair40_private40": {
                "shared_all": 0.2,
                "pairwise": 0.4,
                "private": 0.4,
            },
        }
        allocation_preset = enc_cfg.get("modality_allocation_preset", None)
        modality_allocation = enc_cfg.get("modality_allocation", None)
        if allocation_preset not in [None, "None", "null", ""]:
            allocation_key = str(allocation_preset)
            if allocation_key not in allocation_presets:
                raise ValueError(
                    f"Unknown modality_allocation_preset={allocation_key!r}. "
                    f"Expected one of {sorted(allocation_presets)}."
                )
            modality_allocation = allocation_presets[allocation_key]
        if modality_allocation is not None:
            modality_allocation = {
                str(key): float(value)
                for key, value in dict(modality_allocation).items()
            }
        modality_allocation_choices = enc_cfg.get("modality_allocation_choices", None)
        if modality_allocation_choices is not None:
            modality_allocation_choices = [
                {
                    str(key): float(value)
                    for key, value in dict(allocation).items()
                }
                for allocation in modality_allocation_choices
            ]

        modality_mlp_hidden_dim = enc_cfg.get("modality_mlp_hidden_dim", None)
        if modality_mlp_hidden_dim is not None:
            modality_mlp_hidden_dim = int(modality_mlp_hidden_dim)

        scm_config = SCMConfig(
            n_modalities=int(enc_cfg.get("n_modalities", 3)),
            n_latents=int(enc_cfg.get("n_latents", 8)),
            latent_dim=int(enc_cfg.get("latent_dim", 4)),
            embedding_dim=int(enc_cfg["embedding_dim"]),
            edge_probability=float(enc_cfg.get("edge_probability", 0.3)),
            modality_connection_probability=float(enc_cfg.get("modality_connection_probability", 0.5)),
            parameter_sampling_mode=str(enc_cfg.get("parameter_sampling_mode", "static")),
            n_latents_choices=enc_cfg.get("n_latents_choices", None),
            latent_dim_choices=enc_cfg.get("latent_dim_choices", None),
            modality_readout_alpha_choices=enc_cfg.get("modality_readout_alpha_choices", None),
            edge_probability_choices=enc_cfg.get("edge_probability_choices", None),
            modality_mask_mode=str(enc_cfg.get("modality_mask_mode", "bernoulli")),
            shared_all_mode=enc_cfg.get("shared_all_mode", None),
            pairwise_mode=enc_cfg.get("pairwise_mode", None),
            modality_allocation=modality_allocation,
            modality_allocation_mode=str(enc_cfg.get("modality_allocation_mode", "static")),
            modality_allocation_choices=modality_allocation_choices,
            mechanism_types=mechanism_types,
            mechanism_types_mode=str(enc_cfg.get("mechanism_types_mode", "static")),
            mechanism_type_choices=mechanism_type_choices,
            modality_readout=str(enc_cfg.get("modality_readout", "linear")),
            modality_readout_alpha=float(enc_cfg.get("modality_readout_alpha", 1.0)),
            modality_mlp_hidden_dim=modality_mlp_hidden_dim,
            latent_noise_std=float(enc_cfg.get("latent_noise_std", 0.1)),
            modality_noise_std=enc_cfg.get("modality_noise_std", None),
            modality_noise_std_choices=enc_cfg.get("modality_noise_std_choices", None),
            seed=int(cfg.get("seed", 420)),
            shared_modality_projections=_as_bool(enc_cfg.get("shared_modality_projections", False)),
            shared_projection_seed=enc_cfg.get("shared_projection_seed", None),
        )
        return DataModule_SCMPretrain(
            n_train_scms=dm_cfg.get("n_train_scms", 10_000),
            n_val_scms=dm_cfg.get("n_val_scms", 1_000),
            n_test_scms=dm_cfg.get("n_test_scms", 1_000),
            samples_per_scm=dm_cfg.get("samples_per_scm", cfg["batch_size"]),
            scms_per_batch=dm_cfg.get("scms_per_batch", 1),
            num_workers=dm_cfg.get("num_workers", 0),
            seed=cfg.get("seed", 420),
            scm_config=scm_config,
            control_mode=dm_cfg.get("control_mode", "none"),
        )

    # Symile-MIMIC
    if cfg["dataset_name"] == "symile_mimic":
        dm_cfg = cfg.get("datamodule", {})
        return DataModule_SymileMimic(
            batch_size=cfg["batch_size"],
            split_nr=cfg["split_nr"],
            data_dir=dm_cfg.get("data_dir", Dataset_SymileMimic.DEFAULT_DATA_DIR),
            num_workers=dm_cfg.get("num_workers", 4),
        )
    
    # UKBB 
    if cfg["dataset_name"] == "ukb":
        # lazy import if private UDM not installed 
        from udm.general_datamodule import GeneralDatamodule
        datamodule_params = {key: value for key, value in cfg.datamodule.items()}
        return GeneralDatamodule(**datamodule_params)

    # MIMIC (new trimodal variant)
    if cfg["dataset_name"] == "mimic":
        return DataModule_MIMIC(
            train_split_csv=cfg["encoders"]["train_split_csv"],
            val_split_csv=cfg["encoders"]["val_split_csv"],
            test_split_csv=cfg["encoders"]["test_split_csv"],
            mimic_root=cfg["encoders"]["mimic_root"],
            text_model_id=cfg["encoders"]["radiology"].get("text_model_id", "StanfordAIMI/RadBERT"),
            max_text_length=cfg["encoders"]["radiology"].get("max_text_length", 256),
            load_radiology_text=cfg["encoders"].get("load_radiology_text", True),
            require_all_modalities=cfg["encoders"].get("require_all_modalities", False),
            radiology_chunk_size=cfg["encoders"].get("radiology_chunk_size", 250000),
            batch_size=cfg["batch_size"],
            num_workers=cfg["encoders"].get("num_workers", 4),
        )

    if cfg["dataset_name"] == "inspect":
        dm_cfg = cfg["datamodule"]
        report_cfg = cfg["encoders"]["report"]
        return INSPECTDataModule(
            data_dir=dm_cfg["data_dir"],
            batch_size=cfg["batch_size"],
            split_nr=cfg["split_nr"],
            num_workers=dm_cfg.get("num_workers", 4),
            text_model_id=report_cfg.get(
                "text_model_id", "StanfordAIMI/RadBERT"
            ),
            max_text_length=report_cfg.get("max_text_length", 256),
            pin_memory=dm_cfg.get("pin_memory", True),
            drop_last=dm_cfg.get("drop_last", True),
            shuffle=dm_cfg.get("shuffle", True),
        )

    raise ValueError(f"Data module {cfg['dataset_name']} not implemented.")



def set_all_seeds(seed):
    import random
    import numpy as np
    import pytorch_lightning as pl

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    pl.seed_everything(seed)

    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    #torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
