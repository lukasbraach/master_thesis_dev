from typing import Optional, Tuple, Dict, Any

import hydra
import rootutils
from lightning import LightningModule
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def save(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    if cfg.get("ckpt_path"):
        log.info("Restoring from checkpoint")
        model = model.__class__.load_from_checkpoint(cfg.get("ckpt_path"), net=model.net, optimizer=None)

    if cfg.get("save_path"):
        log.info("Saving model")
        model.get_pretrained_model().save_pretrained(cfg.get("save_path"))

    return {}, {}


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # train the model
    save(cfg)

    return None


if __name__ == "__main__":
    main()
