
"""Import modules required to run the Jupyter notebook."""
from __future__ import annotations

# Clear logger to use tiatoolbox.logger
import logging

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

import copy
import random
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

# Third party imports
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datetime import datetime

import ujson as json
from sklearn.linear_model import LogisticRegression as PlattScaling
from sklearn.metrics import average_precision_score as auprc_scorer
from sklearn.metrics import roc_auc_score as auroc_scorer
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from tiatoolbox import logger

from tiatoolbox.utils.misc import download_data, save_as_json
from tiatoolbox.utils.visualization import plot_graph
from tiatoolbox.wsicore.wsireader import (
    OpenSlideWSIReader,
)
from tiatoolbox.utils.misc import select_device

from utils import load_json, rm_n_mkdir, mkdir, recur_find_ext
from dset import SlideGraphDataset, stratified_split, StratifiedSampler
from model import SlideGraphArch
from utils import create_pbar, ScalarMovingAverage

warnings.filterwarnings("ignore")
mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook

ON_GPU = True

device = select_device(on_gpu=ON_GPU)

SEED = 5
random.seed(SEED)
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

WORKSPACE_DIR = Path("./workspace")

# ROOT_OUTPUT_DIR = WORKSPACE_DIR / "training" / f"session_{datetime.now().strftime('%m_%d_%H_%M_%S')}"
ROOT_OUTPUT_DIR = WORKSPACE_DIR / "training"
ROOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_DIR = WORKSPACE_DIR / "graphs"
CLINICAL_FILE = Path("PATH/TO/DIR/")
SPLIT_PATH = ROOT_OUTPUT_DIR / "splits.dat"
SCALER_PATH = ROOT_OUTPUT_DIR / "node_scaler.dat"
MODEL_DIR = ROOT_OUTPUT_DIR / "model"

NUM_EPOCHS = 100
NUM_NODE_FEATURES = 4

wsi_paths = recur_find_ext(GRAPH_DIR, [".json"])
wsi_names = [Path(v).stem for v in wsi_paths]
assert len(wsi_paths) > 0, "No files found."  # noqa: S101
wsi_labels = np.random.random_integers(0, 4, len(wsi_names))

label_df = list(zip(wsi_names, wsi_labels))
label_df = pd.DataFrame(label_df, columns=["WSI-CODE", "LABEL"])

NUM_FOLDS = 5
TEST_RATIO = 0.2
TRAIN_RATIO = 0.8 * 0.9
VALID_RATIO = 0.8 * 0.1

if SPLIT_PATH and SPLIT_PATH.exists():
    splits = joblib.load(SPLIT_PATH)
else:
    x = np.array(label_df["WSI-CODE"].to_list())
    y = np.array(label_df["LABEL"].to_list())
    splits = stratified_split(x, y, TRAIN_RATIO, VALID_RATIO, TEST_RATIO, NUM_FOLDS)
    joblib.dump(splits, SPLIT_PATH)

if SCALER_PATH and SCALER_PATH.exists():
    node_scaler = joblib.load(SCALER_PATH)
else:
    # ! we need a better way of doing this, will have OOM problem
    loader = SlideGraphDataset(wsi_names, mode="infer", graph_dir=GRAPH_DIR)
    loader = DataLoader(
        loader,
        num_workers=8,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    node_features = [v["graph"].x.numpy() for idx, v in enumerate(tqdm(loader))]
    node_features = np.concatenate(node_features, axis=0)
    node_scaler = StandardScaler(copy=False)
    node_scaler.fit(node_features)
    joblib.dump(node_scaler, SCALER_PATH)

# we must define the function after training/loading
def nodes_preproc_func(node_features: np.ndarray) -> np.ndarray:
    """Pre-processing function for nodes."""
    return node_scaler.transform(node_features)

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
            pbar = create_pbar(loader_name, len(loader))
            for _step, batch_data in enumerate(loader):
                # * STEP COMPLETE CALLBACKS
                if loader_name == "train":
                    output = model.train_batch(model, batch_data, optimizer, on_gpu=on_gpu)
                    # check the output for agreement
                    ema({"loss": output[0]})
                    pbar.postfix[1]["step"] = output[0]
                    pbar.postfix[1]["EMA"] = ema.tracking_dict["loss"]
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
                pbar.update()
            pbar.close()

            # * EPOCH COMPLETE

            # Callbacks to process output
            logging_dict = {}
            if loader_name == "train":
                for val_name, val in ema.tracking_dict.items():
                    logging_dict[f"train-EMA-{val_name}"] = val
            elif "infer" in loader_name and any(
                v in loader_name for v in ["train", "valid"]
            ):
                # Expand the list of N dataset size x H heads
                # back to a list of H Head each with N samples.
                output = list(zip(*step_output))
                logit, true = output
                logit = np.squeeze(np.array(logit))
                true = np.squeeze(np.array(true))

                if "train" in loader_name:
                    scaler = PlattScaling()
                    scaler.fit(np.array(logit, ndmin=2).T, true)
                    model.aux_model["scaler"] = scaler
                scaler = model.aux_model["scaler"]
                prob = scaler.predict_proba(np.array(logit, ndmin=2).T)[:, 0]

                # val = auroc_scorer(true, prob)
                val = 0
                logging_dict[f"{loader_name}-auroc"] = val
                # val = auprc_scorer(true, prob)
                val = 0
                logging_dict[f"{loader_name}-auprc"] = val

                logging_dict[f"{loader_name}-raw-logit"] = logit
                logging_dict[f"{loader_name}-raw-true"] = true

            # Callbacks for logging and saving
            for val_name, val in logging_dict.items():
                if "raw" not in val_name:
                    logging.info("%s: %d:", val_name, val)
            if "train" not in loader_dict:
                continue

            # Track the statistics
            # new_stats = {}
            # if (save_dir / "stats.json").exists():
            #     old_stats = load_json(f"{save_dir}/stats.json")
            #     # Save a backup first
            #     save_as_json(old_stats, f"{save_dir}/stats.old.json", exist_ok=False)
            #     new_stats = copy.deepcopy(old_stats)
            #     new_stats = {int(k): v for k, v in new_stats.items()}

            # old_epoch_stats = {}
            # if epoch in new_stats:
            #     old_epoch_stats = new_stats[epoch]
            # old_epoch_stats.update(logging_dict)
            # new_stats[epoch] = old_epoch_stats
            # save_as_json(new_stats, f"{save_dir}/stats.json", exist_ok=False)

            # Save the pytorch model
            model.save(
                f"{save_dir}/epoch={epoch:03d}.weights.pth",
                f"{save_dir}/epoch={epoch:03d}.aux.dat",
            )
    return step_output

logging.basicConfig(
    level=logging.INFO,
)

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
node_scaler = joblib.load(SCALER_PATH)
loader_kwargs = {
    "num_workers": 8,
    "batch_size": 16,
}
arch_kwargs = {
    "dim_features": NUM_NODE_FEATURES,
    "dim_target": 1,
    "layers": [16, 16, 8],
    "dropout": 0.5,
    "pooling": "mean",
    "conv": "EdgeConv",
    "aggr": "max",
}
optim_kwargs = {
    "lr": 1.0e-3,
    "weight_decay": 1.0e-4,
}

if not MODEL_DIR.exists() or True:
    for split_idx, split in enumerate(splits):
        new_split = {
            "train": split["train"],
            "infer-train": split["train"],
            "infer-valid-A": split["valid"],
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

# def select_checkpoints(
#     stat_file_path: str,
#     top_k: int = 2,
#     metric: str = "infer-valid-auprc",
#     epoch_range: tuple[int] | None = None,
# ) -> tuple[list, list]:
#     """Select checkpoints basing on training statistics.

#     Args:
#         stat_file_path (str): Path pointing to the .json
#             which contains the statistics.
#         top_k (int): Number of top checkpoints to be selected.
#         metric (str): The metric name saved within .json to perform
#             selection.
#         epoch_range (list): The range of epochs for checking, denoted
#             as [start, end] . Epoch x that is `start <= x <= end` is
#             kept for further selection.

#     Returns:
#         paths (list): List of paths or info tuple where each point
#             to the correspond check point saving location.
#         stats (list): List of corresponding statistics.

#     """
#     if epoch_range is None:
#         epoch_range = [0, 1000]
#     stats_dict = load_json(stat_file_path)
#     # k is the epoch counter in this case
#     stats_dict = {
#         k: v
#         for k, v in stats_dict.items()
#         if int(k) >= epoch_range[0] and int(k) <= epoch_range[1]
#     }
#     stats = [[int(k), v[metric], v] for k, v in stats_dict.items()]
#     # sort epoch ranking from largest to smallest
#     stats = sorted(stats, key=lambda v: v[1], reverse=True)
#     chkpt_stats_list = stats[:top_k]  # select top_k

#     model_dir = Path(stat_file_path).parent
#     epochs = [v[0] for v in chkpt_stats_list]
#     paths = [
#         (
#             f"{model_dir}/epoch={epoch:03d}.weights.pth",
#             f"{model_dir}/epoch={epoch:03d}.aux.dat",
#         )
#         for epoch in epochs
#     ]
#     chkpt_stats_list = [[v[0], v[2]] for v in chkpt_stats_list]
#     print(paths)  # noqa: T201
#     return paths, chkpt_stats_list

# # default parameters
# TOP_K = 1
# metric_name = "infer-valid-B-auroc"
# PRETRAINED_DIR = f"{ROOT_OUTPUT_DIR}/model/"
# SCALER_PATH = f"{ROOT_OUTPUT_DIR}/node_scaler.dat"

# # Uncomment and set these variables to run the next cell,
# # either seperately or with customized parameters

# splits = joblib.load(SPLIT_PATH)
# node_scaler = joblib.load(SCALER_PATH)
# loader_kwargs = {
#     "num_workers": 8,
#     "batch_size": 16,
# }
# arch_kwargs = {
#     "dim_features": NUM_NODE_FEATURES,
#     "dim_target": 1,
#     "layers": [16, 16, 8],
#     "dropout": 0.5,
#     "pooling": "mean",
#     "conv": "EdgeConv",
#     "aggr": "max",
# }

# cum_stats = []
# for split_idx, split in enumerate(splits):
#     new_split = {"infer": [v[0] for v in split["test"]]}

#     stat_files = recur_find_ext(f"{PRETRAINED_DIR}/{split_idx:02d}/", [".json"])
#     stat_files = [v for v in stat_files if ".old.json" not in v]
#     assert len(stat_files) == 1  # noqa: S101
#     chkpts, chkpt_stats_list = select_checkpoints(
#         stat_files[0],
#         top_k=TOP_K,
#         metric=metric_name,
#     )

#     # Perform ensembling by averaging probabilities
#     # across checkpoint predictions
#     cum_results = []
#     for chkpt_info in chkpts:
#         chkpt_results = run_once(
#             new_split,
#             num_epochs=1,
#             save_dir=None,
#             pretrained=chkpt_info,
#             arch_kwargs=arch_kwargs,
#             loader_kwargs=loader_kwargs,
#         )
#         # * re-calibrate logit to probabilities
#         model = SlideGraphArch(**arch_kwargs)
#         model.load(*chkpt_info)
#         scaler = model.aux_model["scaler"]
#         chkpt_results = np.array(chkpt_results)
#         chkpt_results = np.squeeze(chkpt_results)
#         chkpt_results = scaler.transform(chkpt_results)

#         cum_results.append(chkpt_results)
#     cum_results = np.array(cum_results)
#     cum_results = np.squeeze(cum_results)

#     prob = cum_results
#     if len(cum_results.shape) == 2:  # noqa: PLR2004
#         prob = np.mean(cum_results, axis=0)

#     # * Calculate split statistics
#     true = [v[1] for v in split["test"]]
#     true = np.array(true)

#     cum_stats.append(
#         {"auroc": auroc_scorer(true, prob), "auprc": auprc_scorer(true, prob)},
#     )

# # Now we print out the results.
# # 
# # 

# stat_df = pd.DataFrame(cum_stats)
# for metric in stat_df.columns:
#     vals = stat_df[metric]
#     mu = np.mean(vals)
#     va = np.std(vals)
#     logger.info(" %s: %0.4fÂ±%0.4f", metric, mu, va)

# # #### Visualizing Node Activation of the Graph Neural Network
# # 
# # 

# # Visualizing the activations of each node within the graph is sometimes necessary to either debug or
# # verify the predictions of the graph neural network. Here, we demonstrate
# # 
# # 1. Loading a pretrained model and running inference on one single sample graph.
# # 1. Retrieving the node activations and plot them on the original WSI.
# # 
# # By default, notice that node activations are output when running the `mode.forward(input)` (Or
# # simply `model(input)` in pytorch).
# # 
# # By default, we download the pretrained model as well as samples from the tiatoolbox server to
# # `DOWNLOAD_DIR`. However, if you want to use your own set of input, you can comment out the next cell
# # and provide your own data.
# # 
# # 

# # ! If you want to run your own set of input, comment out this cell
# # ! and uncomment the next cell
# DOWNLOAD_DIR = "local/dump/"
# WSI_PATH = f"{DOWNLOAD_DIR}/sample.svs"
# GRAPH_PATH = f"{DOWNLOAD_DIR}/graph.json"
# SCALER_PATH = f"{DOWNLOAD_DIR}/node_scaler.dat"
# MODEL_WEIGHTS_PATH = f"{DOWNLOAD_DIR}/model.weigths.pth"
# MODEL_AUX_PATH = f"{DOWNLOAD_DIR}/model.aux.dat"
# mkdir(DOWNLOAD_DIR)

# # Downloading sample image tile
# URL_HOME = "https://tiatoolbox.dcs.warwick.ac.uk/models/slide_graph/cell-composition"
# download_data(
#     f"{URL_HOME}/TCGA-C8-A278-01Z-00-DX1.188B3FE0-7B20-401A-A6B7-8F1798018162.svs",
#     WSI_PATH,
# )
# download_data(
#     f"{URL_HOME}/TCGA-C8-A278-01Z-00-DX1.188B3FE0-7B20-401A-A6B7-8F1798018162.json",
#     GRAPH_PATH,
# )
# download_data(f"{URL_HOME}/node_scaler.dat", SCALER_PATH)
# download_data(f"{URL_HOME}/model.aux.dat", MODEL_AUX_PATH)
# download_data(f"{URL_HOME}/model.weights.pth", MODEL_WEIGHTS_PATH)

# # If you want to run your own set of input,
# # uncomment these lines and then set variables to run next cell

# # Most of the time the nodes within the graph will be at different resolutions
# # from the resolution at which we want to visualize them. Before plotting, we scale their coordinates
# # to the target resolution. We provide `NODE_RESOLUTION` and `PLOT_RESOLUTION` variables
# # respectively as the resolution of the node and the resolution at which to plot the graph.
# # 
# # 

# NODE_SIZE = 25
# NUM_NODE_FEATURES = 4
# NODE_RESOLUTION = {"resolution": 0.25, "units": "mpp"}
# PLOT_RESOLUTION = {"resolution": 4.0, "units": "mpp"}

# node_scaler = joblib.load(SCALER_PATH)
# loader_kwargs = {
#     "num_workers": 8,
#     "batch_size": 16,
# }
# arch_kwargs = {
#     "dim_features": NUM_NODE_FEATURES,
#     "dim_target": 1,
#     "layers": [16, 16, 8],
#     "dropout": 0.5,
#     "pooling": "mean",
#     "conv": "EdgeConv",
#     "aggr": "max",
# }


# with GRAPH_PATH.open() as fptr:
#     graph_dict = json.load(fptr)
# graph_dict = {k: np.array(v) for k, v in graph_dict.items()}
# graph_dict["x"] = node_scaler.transform(graph_dict["x"])
# graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
# graph = Data(**graph_dict)
# batch = Batch.from_data_list([graph])

# model = SlideGraphArch(**arch_kwargs)
# model.load(MODEL_WEIGHTS_PATH, MODEL_AUX_PATH)
# model = model.to("cuda")

# # Data type conversion
# batch = batch.to("cuda")
# batch.x = batch.x.type(torch.float32)
# predictions, node_activations = model(batch)
# node_activations = node_activations.detach().cpu().numpy()

# reader = OpenSlideWSIReader(WSI_PATH)
# node_resolution = reader.slide_dimensions(**NODE_RESOLUTION)
# plot_resolution = reader.slide_dimensions(**PLOT_RESOLUTION)
# fx = np.array(node_resolution) / np.array(plot_resolution)

# cmap = plt.get_cmap("inferno")
# graph = graph.to("cpu")

# node_coordinates = np.array(graph.coordinates) / fx
# node_colors = (cmap(np.squeeze(node_activations))[..., :3] * 255).astype(np.uint8)
# edges = graph.edge_index.T

# thumb = reader.slide_thumbnail(**PLOT_RESOLUTION)
# thumb_overlaid = plot_graph(
#     thumb.copy(),
#     node_coordinates,
#     edges,
#     node_colors=node_colors,
#     node_size=NODE_SIZE,
# )

# ax = plt.subplot(1, 1, 1)
# plt.imshow(thumb_overlaid)
# plt.axis("off")
# # Add minorticks on the colorbar to make it easy to read the
# # values off the colorbar.
# fig = plt.gcf()
# norm = mpl.colors.Normalize(
#     vmin=np.min(node_activations),
#     vmax=np.max(node_activations),
# )
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# cbar = fig.colorbar(sm, ax=ax, extend="both")
# cbar.minorticks_on()
# plt.show()
