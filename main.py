import os

import hydra
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.plugins.environments import LightningEnvironment

from helpers import set_all_seeds, build_model, build_datamodule

try:
    from udm.plugins.tabular.tabular_plugin import TabularPlugin

    if not hasattr(TabularPlugin, "_original_setup"):
        TabularPlugin._original_setup = TabularPlugin._setup

        def patched_setup(self):
            if self.tabular_df is None:
                self.tabular_df = self.load_tabular_df()
            self._original_setup()

        TabularPlugin._setup = patched_setup
except ImportError:
    pass

os.environ["WANDB_SILENT"] = "true"
torch.set_float32_matmul_precision("high")


@hydra.main(config_path="config", config_name="config")
def main(cfg) -> None:
    set_all_seeds(seed=cfg["seed"])
    wandb.finish()
    group = cfg["wandb"]["group"]
    logger = WandbLogger(
        project=cfg["wandb"]["project"],
        dir="wandb/",
        group=None if group == "None" else group,
    )

    model = build_model(cfg)
    datamodule = build_datamodule(cfg)

    save_checkpoint = cfg["ckpt_name"] not in (None, "None")
    checkpoint_callback = None
    if save_checkpoint:
        checkpoint_path = os.path.join(cfg["ckpt_dir"], f'{cfg["ckpt_name"]}.ckpt')
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        checkpoint_callback = ModelCheckpoint(
            monitor=cfg["encoders"]["monitor"]["metric"],
            mode=cfg["encoders"]["monitor"]["mode"],
            dirpath=cfg["ckpt_dir"],
            filename=cfg["ckpt_name"],
            save_top_k=1,
        )

    monitor = (
        cfg["encoders"]["monitor"]
        if save_checkpoint
        else {"metric": cfg["monitor"], "mode": cfg["monitor_mode"]}
    )
    early_stopping = EarlyStopping(
        monitor=monitor["metric"],
        mode=monitor["mode"],
        patience=cfg["patience"],
        check_on_train_epoch_end=False,
    )
    callbacks = [early_stopping]
    if checkpoint_callback is not None:
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        logger=logger,
        accelerator="gpu",
        devices=cfg["devices"],
        strategy=(
            "ddp_find_unused_parameters_false" if cfg["devices"] > 1 else "auto"
        ),
        max_epochs=cfg["max_epochs"],
        limit_train_batches=cfg["limit_train_batches"],
        precision=cfg["precision"],
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        enable_checkpointing=save_checkpoint,
        callbacks=callbacks,
        log_every_n_steps=1,
        plugins=[LightningEnvironment()],
        gradient_clip_val=(
            None if cfg["modelname"]["modelname"] == "barybind" else 1.0
        ),
        gradient_clip_algorithm="norm",
        sync_batchnorm=cfg["devices"] > 1,
    )

    trainer.fit(model, datamodule=datamodule)
    if save_checkpoint:
        trainer.test(ckpt_path="best", datamodule=datamodule)
    wandb.finish()


if __name__ == "__main__":
    main()
