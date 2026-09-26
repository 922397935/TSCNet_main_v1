from omegaconf import DictConfig, open_dict
from .adhd import load_adhd_data
from .haxby import load_haxby_data
from .abide import load_abide_data
from .dataloader import init_dataloader, init_stratified_dataloader
from typing import List
import torch.utils as utils


def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:

    assert cfg.dataset.name in ['haxby', 'abide', 'adhd']

    datasets = eval(
        f"load_{cfg.dataset.name}_data")(cfg)

    dataloaders = init_stratified_dataloader(cfg, *datasets) \
        if cfg.dataset.stratified \
        else init_dataloader(cfg, *datasets)

    return dataloaders
