"""
Imports all required modules, downloads sample data, reads and saves the constructed graph.
"""

from __future__ import annotations

# Clear logger to use tiatoolbox.logger
import logging

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

# Python standard library imports
import json
import pathlib
import random
import shutil
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable

# Third party imports
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from shapely.geometry import box as shapely_box
from shapely.strtree import STRtree
from skimage.exposure import equalize_hist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import BatchNorm1d, Linear, ReLU
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    EdgeConv,
    GINConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

from tiatoolbox import logger
from tiatoolbox.data import stain_norm_target
from tiatoolbox.models import (
    DeepFeatureExtractor,
    IOSegmentorConfig,
    NucleusInstanceSegmentor,
)
from tiatoolbox.models.architecture.vanilla import CNNBackbone
from tiatoolbox.tools.graph import SlideGraphConstructor
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.tools.stainnorm import get_normalizer
from tiatoolbox.utils.misc import download_data, select_device
from tiatoolbox.utils.visualization import plot_graph
from tiatoolbox.wsicore.wsireader import WSIReader

if TYPE_CHECKING:
    from tiatoolbox.wsicore.wsimeta import Resolution, Units

warnings.filterwarnings("ignore")
mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook
mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode


ON_GPU = False  # Should be changed to False if no cuda-enabled GPU is available


def load_json(path: str) -> dict | list | int | float | str:
    """Load JSON from a file path."""
    with path.open() as fptr:
        return json.load(fptr)


def rmdir(dir_path: str) -> None:
    """Remove a directory."""
    if dir_path.is_dir():
        shutil.rmtree(dir_path)


def rm_n_mkdir(dir_path: str) -> None:
    """Remove then re-create a directory."""
    if dir_path.is_dir():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True)


def mkdir(dir_path: str) -> None:
    """Create a directory if it does not exist."""
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)


SEED = 5
random.seed(SEED)
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


WORKSPACE_DIR = Path("./workspace")
DOWNLOAD_DIR = WORKSPACE_DIR / "download"

WSI_PATH = DOWNLOAD_DIR / "sample.svs"
MSK_PATH = DOWNLOAD_DIR / "sample_mask.png"
PRE_GENERATED_GRAPH_PATH = DOWNLOAD_DIR / "graph.json"
SCALER_PATH = DOWNLOAD_DIR / "node_scaler.dat"
MODEL_WEIGHTS_PATH = DOWNLOAD_DIR / "model.weigths.pth"
MODEL_AUX_PATH = DOWNLOAD_DIR / "model.aux.dat"

# ! Uncomment this to always download data
# rmdir(DOWNLOAD_DIR)

# Downloading sample image tile
if not DOWNLOAD_DIR.exists():
    DOWNLOAD_DIR.mkdir(parents=True)
    URL_HOME = (
        "https://tiatoolbox.dcs.warwick.ac.uk/models/slide_graph/cell-composition"
    )
    SLIDE_CODE = "TCGA-C8-A278-01Z-00-DX1.188B3FE0-7B20-401A-A6B7-8F1798018162"
    download_data(f"{URL_HOME}/{SLIDE_CODE}.svs", WSI_PATH)
    download_data(f"{URL_HOME}/{SLIDE_CODE}.mask.png", MSK_PATH)
    download_data(f"{URL_HOME}/{SLIDE_CODE}.json", PRE_GENERATED_GRAPH_PATH)
    download_data(f"{URL_HOME}/node_scaler.dat", SCALER_PATH)
    download_data(f"{URL_HOME}/model.aux.logistic.dat", MODEL_AUX_PATH)
    download_data(f"{URL_HOME}/model.weights.pth", MODEL_WEIGHTS_PATH)

# ! for using pre-generated graph
NUM_NODE_FEATURES = 4
FEATURE_MODE = "composition" # 10 minutes cpu-local
# FEATURE_MODE = "cnn" # 1.5 hour cpu-local
USE_PRE_GENERATED = True

WSI_FEATURE_DIR = WORKSPACE_DIR / "features/"
GRAPH_PATH = WORKSPACE_DIR / "graph.json"

def construct_graph(wsi_name: str, save_path: str | Path) -> None:
    """Construct graph for one WSI and save to file."""
    positions = np.load(f"{WSI_FEATURE_DIR}/{wsi_name}.position.npy")
    features = np.load(f"{WSI_FEATURE_DIR}/{wsi_name}.features.npy")
    graph_dict = SlideGraphConstructor.build(
        positions[:, :2],
        features,
        feature_range_thresh=None,
    )

    # Write a graph to a JSON file
    with save_path.open("w") as handle:
        graph_dict = {k: v.tolist() for k, v in graph_dict.items()}
        json.dump(graph_dict, handle)


GRAPH_PATH = GRAPH_PATH if not USE_PRE_GENERATED else PRE_GENERATED_GRAPH_PATH


NODE_SIZE = 24
if USE_PRE_GENERATED:
    NODE_RESOLUTION = {"resolution": 0.5, "units": "mpp"}
else:
    NODE_RESOLUTION = {"resolution": 0.25, "units": "mpp"}
PLOT_RESOLUTION = {"resolution": 4.00, "units": "mpp"}

graph_dict = load_json(GRAPH_PATH)
graph_dict = {k: np.array(v) for k, v in graph_dict.items()}
graph = Data(**graph_dict)

# deriving node colors via projecting n-d features down to 3-d
graph.x = StandardScaler().fit_transform(graph.x)
# .c for node colors
node_colors = PCA(n_components=3).fit_transform(graph.x)[:, [1, 0, 2]]
for channel in range(node_colors.shape[-1]):
    node_colors[:, channel] = 1 - equalize_hist(node_colors[:, channel]) ** 2
node_colors = (node_colors * 255).astype(np.uint8)

reader = WSIReader.open(WSI_PATH)
thumb = reader.slide_thumbnail(4.0, "mpp")

node_resolution = reader.slide_dimensions(**NODE_RESOLUTION)
plot_resolution = reader.slide_dimensions(**PLOT_RESOLUTION)
fx = np.array(node_resolution) / np.array(plot_resolution)

if USE_PRE_GENERATED:
    node_coordinates = np.array(graph.coords) / fx
else:
    node_coordinates = np.array(graph.coordinates) / fx
edges = graph.edge_index.T

thumb = reader.slide_thumbnail(**PLOT_RESOLUTION)
thumb_overlaid = plot_graph(
    thumb.copy(),
    node_coordinates,
    edges,
    node_colors=node_colors,
    node_size=NODE_SIZE,
)

plt.subplot(1, 2, 1)
plt.imshow(thumb)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(thumb_overlaid)
plt.axis("off")
plt.tight_layout()
plt.savefig(WORKSPACE_DIR / "download/sample.png")
