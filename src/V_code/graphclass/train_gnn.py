
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
from tiatoolbox.utils.misc import select_device

from utils import load_json, rm_n_mkdir, mkdir, recur_find_ext
from dset import SlideGraphDataset, stratified_split_train_test, StratifiedSampler
from model import SlideGraphArch
from utils import ScalarMovingAverage

warnings.filterwarnings("ignore")
mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook

import argparse, sys
from pprint import pprint
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--on_gpu', type=int, default=0)
parser.add_argument('--mode', type=str, default='resnet')
parser.add_argument('--gembed', type=int, default=0)
parser.add_argument('--conv', type=str, default='EdgeConv')
parser.add_argument('--node_dropout', type=float, default=None)
parser.add_argument('--pred_w_all', type=int, default=1)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
args = parser.parse_args(sys.argv[1:])
pprint(args.__dict__)

ON_GPU = args.on_gpu
device = select_device(on_gpu=ON_GPU)

SEED = args.seed
random.seed(SEED)
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

WORKSPACE_DIR = Path("./workspace/ours")
NUM_EPOCHS = 500
NCLASSES = 3

if args.mode == 'resnet':
    GRAPH_DIR = WORKSPACE_DIR / "graphs_resnet"
    TRAIN_DIR = WORKSPACE_DIR / "training_resnet"
    NUM_NODE_FEATURES = 2048
elif args.mode == 'int':
    GRAPH_DIR = WORKSPACE_DIR / "graphs_int"
    TRAIN_DIR = WORKSPACE_DIR / "training_int"
    NUM_NODE_FEATURES = 128
elif args.mode == 'int_cnt':
    GRAPH_DIR = WORKSPACE_DIR / "graphs_int_cnt"
    TRAIN_DIR = WORKSPACE_DIR / "training_int_cnt"
    NUM_NODE_FEATURES = 128+1
elif args.mode == 'int_cnt_cmp':
    GRAPH_DIR = WORKSPACE_DIR / "graphs_int_cnt_cmp"
    TRAIN_DIR = WORKSPACE_DIR / "training_int_cnt_cmp"
    NUM_NODE_FEATURES = 128+1+5
elif args.mode == 'int_cnt_proj':
    GRAPH_DIR = WORKSPACE_DIR / "graphs_int_cnt_proj"
    TRAIN_DIR = WORKSPACE_DIR / "training_int_cnt_proj"
    NUM_NODE_FEATURES = 128+32
elif args.mode == 'int_cnt_cmp_proj':
    GRAPH_DIR = WORKSPACE_DIR / "graphs_int_cnt_cmp_proj"
    TRAIN_DIR = WORKSPACE_DIR / "training_int_cnt_cmp_proj"
    NUM_NODE_FEATURES = 128+32
else:
    raise NotImplementedError(args.mode)

LABELS_PATH = GRAPH_DIR / "0_labels.txt"
SPLIT_PATH = TRAIN_DIR / "splits.dat"
RUN_OUTPUT_DIR = TRAIN_DIR / f"session_{datetime.now().strftime('%m_%d_%H_%M_%S')}"
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
save_as_json(args.__dict__, f'{RUN_OUTPUT_DIR}/args.json')
MODEL_DIR = RUN_OUTPUT_DIR / "model"

wsi_paths = recur_find_ext(GRAPH_DIR, [".json"])
wsi_names = [Path(v).stem for v in wsi_paths]
assert len(wsi_paths) > 0, "No files found."  # noqa: S101

NUM_FOLDS = 1
TEST_RATIO = 0.2
TRAIN_RATIO = 0.8 * 1.0
VALID_RATIO = 0.8 * 0.0

if SPLIT_PATH and SPLIT_PATH.exists():
    splits = joblib.load(SPLIT_PATH)
else:
    x = np.array(wsi_names)
    with open(LABELS_PATH, 'r') as f:
        labels_dict = json.load(f)
    y = np.array([
        labels_dict[wsi_name+'.json'] for wsi_name in wsi_names
    ])
    # splits = stratified_split(x, y, TRAIN_RATIO, VALID_RATIO, TEST_RATIO, NUM_FOLDS)
    splits = stratified_split_train_test(x, y, TRAIN_RATIO, TEST_RATIO, NUM_FOLDS)
    joblib.dump(splits, SPLIT_PATH)

# # we must define the function after training/loading
# def nodes_preproc_func(node_features: np.ndarray) -> np.ndarray:
#     """Pre-processing function for nodes."""
#     return node_scaler.transform(node_features)
nodes_preproc_func = None

def make_fig_with_new_stats(new_stats, epoch, figpath):
    fig, axes_all = plt.subplots(2, 1, figsize=(7.5, 10))
    # loss
    axes_all[0].set_ylim(0.0001, 1.5)
    axes_all[0].grid(visible=True)
    # accuracy
    axes_all[1].set_ylim(0.2, 1.1)
    axes_all[1].grid(visible=True)
    ax_curr = None
    for pkey in new_stats[0].keys():
        if 'loss' in pkey:
            ax_curr = axes_all[0]
        elif 'accuracy' in pkey:
            ax_curr = axes_all[1]
        else:
            continue
        vals = [new_stats[eitr][pkey] for eitr in range(epoch+1)]
        ax_curr.plot(np.arange(len(vals)), vals, label=pkey)
    fig.tight_layout()
    fig.legend()
    plt.savefig(figpath)
    plt.close()

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

    model = SlideGraphArch(**arch_kwargs)
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
                logging_dict[f"{loader_name}-accuracy"] = metrics.accuracy_score(pred, gtruth)
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

        make_fig_with_new_stats(new_stats, epoch, save_dir/'progress.png')

        if epoch % 50 == 0:
            # Save the pytorch model
            model.save(
                f"{save_dir}/epoch={epoch:03d}.weights.pth",
                f"{save_dir}/epoch={epoch:03d}.aux.dat",
            )
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


splits = joblib.load(SPLIT_PATH)
loader_kwargs = {
    "num_workers": 8,
    "batch_size": 16,
}
arch_kwargs = {
    "dim_features": NUM_NODE_FEATURES,
    "dim_target": NCLASSES,
    "layers": [16, 16, 8],
    "dropout": 0.5,
    "pooling": "mean",
    "aggr": "max",
    # "gembed": False,
    # "gembed": True,
    "gembed": args.gembed,
    # "conv": "EdgeConv",
    # "conv": "GATv2Conv"
    "conv": args.conv,
    # "node_dropout": None,
    # "node_dropout": 0.5,
    "node_dropout": args.node_dropout,
    # "pred_w_all": True,
    # "pred_w_all": False,
    "pred_w_all": args.pred_w_all,
}
optim_kwargs = {
    # "lr": 1.0e-3,
    # "lr": 1.0e-3,
    "lr": args.lr,
    # "weight_decay": 1.0e-4,
    # "weight_decay": 1.0e-3,
    "weight_decay": args.weight_decay,
}

if not MODEL_DIR.exists() or True:
    for split_idx, split in enumerate(splits):
        new_split = {
            "train": split["train"],
            "infer-train": split["train"],
            # "infer-valid-A": split["valid"],
            "infer-valid-B": split["test"],
        }
        split_save_dir = MODEL_DIR/f"{split_idx:02d}/"
        rm_n_mkdir(split_save_dir)
        reset_logging(split_save_dir)
        run_once(
            new_split,
            NUM_EPOCHS,
            save_dir=split_save_dir,
            arch_kwargs=arch_kwargs,
            loader_kwargs=loader_kwargs,
            optim_kwargs=optim_kwargs,
            on_gpu=ON_GPU,
        )
