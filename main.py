import os 
import wandb
import torch
import hydra

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from hydra import compose, initialize
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins.environments import LightningEnvironment

from helpers import set_all_seeds, build_model, build_datamodule

# Patch TabularPlugin to fix DDP spawn crash where tabular_df is None on subprocesses
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
    set_all_seeds(seed=cfg.seed)
    wandb.finish()
    logger = WandbLogger(project=cfg.wandb.project, dir="wandb/")

    model = build_model(cfg)
    datamodule = build_datamodule(cfg)

    early_stopping = EarlyStopping(
        monitor=cfg.monitor,
        mode=cfg.monitor_mode,
        patience=cfg.patience,
        check_on_train_epoch_end=False,
    )

    trainer = pl.Trainer(
        logger=logger,
        accelerator='gpu',
        devices=cfg.devices,
        strategy="ddp_find_unused_parameters_false" if cfg.devices > 1 else "auto", 
        max_epochs=cfg.max_epochs,
        limit_train_batches=cfg.limit_train_batches,
        precision=cfg.precision,
        check_val_every_n_epoch=1,
        val_check_interval=1.0,
        enable_checkpointing=False,
        callbacks=[early_stopping],
        log_every_n_steps=1,
        plugins=[LightningEnvironment()],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        sync_batchnorm=True if cfg.devices > 1 else False,
        #overfit_batches=1,
    )

    trainer.fit(model, datamodule=datamodule)
    # trainer.test(lightningmodule, datamodule)
    
    #wandb.finish()

if __name__ == "__main__":
    main()
