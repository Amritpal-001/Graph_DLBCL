
"""Import modules required to run the Jupyter notebook."""
from __future__ import annotations

# Clear logger to use tiatoolbox.logger
import logging

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

logging.basicConfig(
    level=logging.INFO,
)

import copy
import random
import warnings
from pathlib import Path

# Third party imports
import joblib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn import metrics

import ujson as json
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from tiatoolbox import logger

from tiatoolbox.utils.misc import save_as_json


from src.utils import load_json, rm_n_mkdir, mkdir, recur_find_ext
from src.dset import SlideGraphDataset, stratified_split, StratifiedSampler
from src.model import SlideGraphArch
from src.utils import ScalarMovingAverage

warnings.filterwarnings("ignore")
mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook


nodes_preproc_func = None

def run_once(  # noqa: C901, PLR0912, PLR0915
    dataset_dict: dict,
    num_epochs: int,
    save_dir: str | Path,
    pretrained: str | None = None,
    loader_kwargs: dict | None = None,
    arch_kwargs: dict | None = None,
    optim_kwargs: dict | None = None,
    *,
    on_gpu: bool = True,
    GRAPH_DIR = None
) -> list:
    """Running the inference or training loop once.

    The actual running mode is defined via the code name of the dataset
    within `dataset_dict`. Here, `train` is specifically preserved for
    the dataset used for training. `.*infer-valid.*` and `.*infer-train*`
    are reserved for datasets containing the corresponding labels.
    Otherwise, the dataset is assumed to be for the inference run.

    """
    if loader_kwargs is None:
        loader_kwargs = {}

    if arch_kwargs is None:
        arch_kwargs = {}

    if optim_kwargs is None:
        optim_kwargs = {}

    if on_gpu == True:
        device = "cuda"
    else:
        device = "cpu"

    model = SlideGraphArch(**arch_kwargs)
    print(model)
    if pretrained is not None:
        model.load(*pretrained)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), **optim_kwargs)

    # Create the graph dataset holder for each subset info then
    # pipe them through torch/torch geometric specific loader
    # for loading in multi-thread.
    loader_dict = {}
    for subset_name, subset in dataset_dict.items():
        _loader_kwargs = copy.deepcopy(loader_kwargs)
        batch_sampler = None
        if subset_name == "train":
            _loader_kwargs = {}
            batch_sampler = StratifiedSampler(
                labels=[v[1] for v in subset],
                batch_size=loader_kwargs["batch_size"],
            )

        ds = SlideGraphDataset(subset, mode=subset_name, preproc=nodes_preproc_func, graph_dir=GRAPH_DIR)
        loader_dict[subset_name] = DataLoader(
            ds,
            batch_sampler=batch_sampler,
            drop_last=subset_name == "train" and batch_sampler is None,
            shuffle=subset_name == "train" and batch_sampler is None,
            **_loader_kwargs,
        )
    best_score = {}
    for epoch in range(num_epochs):
        logger.info("EPOCH: %03d", epoch)
        for loader_name, loader in loader_dict.items():
            # * EPOCH START
            step_output = []
            ema = ScalarMovingAverage()
            for _step, batch_data in enumerate(tqdm(loader, disable=loader_name!="train")):
                # * STEP COMPLETE CALLBACKS
                if loader_name == "train":
                    outputs = model.train_batch(model, batch_data, optimizer, on_gpu=on_gpu)
                    ema({"loss": outputs[0]})
                else:
                    output = model.infer_batch(model, batch_data, on_gpu=on_gpu)

                    batch_size = batch_data["graph"].num_graphs
                    # Iterate over output head and retrieve
                    # each as N x item, each item may be of
                    # arbitrary dimensions
                    output = [np.split(v, batch_size, axis=0) for v in output]
                    # pairing such that it will be
                    # N batch size x H head list
                    output = list(zip(*output))
                    step_output.extend(output)
                # pbar.update()
            # pbar.close()

            # * EPOCH COMPLETE

            # Callbacks to process output
            logging_dict = {}
            
            if loader_name == "train":
                for val_name, val in ema.tracking_dict.items():
                    logging_dict[f"train-EMA-{val_name}"] = val
            elif "infer" in loader_name and any(v in loader_name for v in ["train", "valid"]):
                # Expand the list of N dataset size x H heads
                # back to a list of H Head each with N samples.
                output = list(zip(*step_output))
                pred, gtruth = output
                pred = np.squeeze(np.array(pred))
                gtruth = np.squeeze(np.array(gtruth))

                # logging_dict[f"{loader_name}-microf1"] = metrics.f1_score(pred, gtruth, average='micro')
                curr_score = metrics.accuracy_score(pred, gtruth)
                logging_dict[f"{loader_name}-accuracy"] = curr_score
                try:
                    if curr_score >= best_score[f"{loader_name}-accuracy"]:
                        best_score[f"{loader_name}-accuracy"] = curr_score
                except:
                    best_score[f"{loader_name}-accuracy"] = 0
                # logging_dict[f"{loader_name}-raw-pred"] = pred
                # logging_dict[f"{loader_name}-raw-gtruth"] = gtruth

            # Callbacks for logging and saving
            for val_name, val in logging_dict.items():
                if "raw" not in val_name:
                    logging.info("%s: %f", val_name, val)
            if "train" not in loader_dict:
                continue

            # Track the statistics
            new_stats = {}
            if (save_dir / "stats.json").exists():
                old_stats = load_json(save_dir/"stats.json")
                # Save a backup first
                save_as_json(old_stats, save_dir/"stats.old.json", exist_ok=True)
                new_stats = copy.deepcopy(old_stats)
                new_stats = {int(k): v for k, v in new_stats.items()}

            old_epoch_stats = {}
            if epoch in new_stats:
                old_epoch_stats = new_stats[epoch]
            old_epoch_stats.update(logging_dict)
            new_stats[epoch] = old_epoch_stats
            save_as_json(new_stats, save_dir/"stats.json", exist_ok=True)

        plt.figure()
        for pkey in new_stats[0].keys():
            vals = [new_stats[eitr][pkey] for eitr in range(epoch+1)]
            plt.plot(np.arange(len(vals)), vals, label=pkey)
        plt.tight_layout()
        plt.legend()
        plt.savefig(save_dir/'progress.png')
        plt.close()

        if epoch % 25 == 0:
            # Save the pytorch model
            model.save(
                f"{save_dir}/epoch={epoch:03d}.weights.pth",
                f"{save_dir}/epoch={epoch:03d}.aux.dat",
            )

        print("best_score" , best_score)
    return step_output

def reset_logging(save_path: str | Path) -> None:
    """Reset logger handler."""
    log_formatter = logging.Formatter(
        "|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d|%H:%M:%S",
    )
    log = logging.getLogger()  # Root logger
    for hdlr in log.handlers[:]:  # Remove all old handlers
        log.removeHandler(hdlr)
    new_hdlr_list = [
        logging.FileHandler(f"{save_path}/debug.log"),
        logging.StreamHandler(),
    ]
    for hdlr in new_hdlr_list:
        hdlr.setFormatter(log_formatter)
        log.addHandler(hdlr)

