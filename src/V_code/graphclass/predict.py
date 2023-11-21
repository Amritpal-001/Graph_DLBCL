
import joblib
import pathlib
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

import torch
from torch_geometric.data import Data, Batch

from tiatoolbox import logger
from tiatoolbox.utils.visualization import plot_graph
from tiatoolbox.wsicore.wsireader import WSIReader
from tiatoolbox.utils.misc import select_device

from model import SlideGraphArch

warnings.filterwarnings("ignore")
mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook
mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode

WORKSPACE_DIR = pathlib.Path("./workspace")
DOWNLOAD_DIR = WORKSPACE_DIR / "download"
SCALER_PATH = DOWNLOAD_DIR / "node_scaler.dat"
WSI_PATH = DOWNLOAD_DIR / "sample.svs"
MODEL_WEIGHTS_PATH = DOWNLOAD_DIR / "model.weigths.pth"
MODEL_AUX_PATH = DOWNLOAD_DIR / "model.aux.dat"

ON_GPU = True
device = select_device(on_gpu=ON_GPU)

NUM_NODE_FEATURES = 4
FEATURE_MODE = "download"
# FEATURE_MODE = "cnn"
# FEATURE_MODE = "composition"
WSI_GRAPH_DIR = WORKSPACE_DIR/FEATURE_MODE

USE_PRE_GENERATED = FEATURE_MODE == "download"

if USE_PRE_GENERATED:
    wsi_name = "graph"
else:
    wsi_name = pathlib.Path(WSI_PATH).stem

GRAPH_PATH = WSI_GRAPH_DIR / f"{wsi_name}.json"

arch_kwargs = {
    "dim_features": NUM_NODE_FEATURES,
    "dim_target": 1,
    "layers": [16, 16, 8],
    "dropout": 0.5,
    "pooling": "mean",
    "conv": "EdgeConv",
    "aggr": "max",
}
node_scaler = joblib.load(SCALER_PATH)

with GRAPH_PATH.open() as fptr:
    graph_dict = json.load(fptr)
graph_dict = {k: np.array(v) for k, v in graph_dict.items()}
graph_dict["x"] = node_scaler.transform(graph_dict["x"])
graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
graph = Data(**graph_dict)
batch = Batch.from_data_list([graph])

# model weights only available for cell-composition?
model = SlideGraphArch(**arch_kwargs)
model.load(MODEL_WEIGHTS_PATH, MODEL_AUX_PATH, map_location=device)
model = model.to(device)

# Data type conversion
with torch.inference_mode():
    batch = batch.to(device)
    batch.x = batch.x.type(torch.float32)
    predictions, node_activations = model(batch)
    predictions = predictions.detach().cpu().numpy()
    probabilities = model.aux_model["scaler"].predict_proba(predictions)[:, 1]
    logger.info("Prediction: %f", predictions[0][0])
    logger.info("Probabilities: %f", probabilities[0])
    node_activations = node_activations.detach().cpu().numpy()

NODE_SIZE = 25
NUM_NODE_FEATURES = 4
NODE_RESOLUTION = {"resolution": 0.5, "units": "mpp"}
PLOT_RESOLUTION = {"resolution": 4.0, "units": "mpp"}

reader = WSIReader.open(WSI_PATH)

if FEATURE_MODE == "composition":
    cmap = plt.get_cmap("inferno")
    graph = graph.to("cpu")
    
    node_resolution = reader.slide_dimensions(**NODE_RESOLUTION)
    plot_resolution = reader.slide_dimensions(**PLOT_RESOLUTION)
    fx = np.array(node_resolution) / np.array(plot_resolution)

    if USE_PRE_GENERATED:
        node_coordinates = np.array(graph.coords) / fx
    else:
        node_coordinates = np.array(graph.coordinates) / fx
    node_colors = (cmap(np.squeeze(node_activations))[..., :3] * 255).astype(np.uint8)
    edges = graph.edge_index.T

    thumb = reader.slide_thumbnail(**PLOT_RESOLUTION)
    thumb_overlaid = plot_graph(
        thumb.copy(),
        node_coordinates,
        edges,
        node_colors=node_colors,
        node_size=NODE_SIZE,
    )

    ax = plt.subplot(1, 1, 1)
    plt.imshow(thumb_overlaid)
    plt.axis("off")
    # Add minorticks on the colorbar to make it easy to read the
    # values off the colorbar.
    fig = plt.gcf()
    norm = mpl.colors.Normalize(
        vmin=np.min(node_activations),
        vmax=np.max(node_activations),
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, extend="both")
    cbar.minorticks_on()
    plt.tight_layout()
    plt.savefig(WSI_GRAPH_DIR / f"{wsi_name}_predicted.png")
