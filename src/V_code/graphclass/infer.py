"""Import modules required to run the Jupyter notebook."""
from __future__ import annotations

# Clear logger to use tiatoolbox.logger
import logging

if logging.getLogger().hasHandlers():
    logging.getLogger().handlers.clear()

# Python standard library imports
import json
import pathlib
import random
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable

# Third party imports
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from shapely.geometry import box as shapely_box
from shapely.strtree import STRtree
from skimage.exposure import equalize_hist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

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
from tiatoolbox.utils.misc import download_data
from tiatoolbox.utils.visualization import plot_graph
from tiatoolbox.wsicore.wsireader import WSIReader

if TYPE_CHECKING:
    from tiatoolbox.wsicore.wsimeta import Resolution, Units

from utils import load_json, rmdir, rm_n_mkdir, mkdir

warnings.filterwarnings("ignore")
mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook
mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode


ON_GPU = True  # Should be changed to False if no cuda-enabled GPU is available

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


def extract_deep_features(
    wsi_paths: list[str],
    msk_paths: list[str],
    save_dir: str,
    preproc_func: Callable | None = None,
) -> list:
    """Helper function to extract deep features."""
    ioconfig = IOSegmentorConfig(
        input_resolutions=[
            {"units": "mpp", "resolution": 0.25},
        ],
        output_resolutions=[
            {"units": "mpp", "resolution": 0.25},
        ],
        patch_input_shape=[512, 512],
        patch_output_shape=[512, 512],
        stride_shape=[512, 512],
        save_resolution={"units": "mpp", "resolution": 8.0},
    )
    model = CNNBackbone("resnet50")
    extractor = DeepFeatureExtractor(batch_size=32, model=model, num_loader_workers=4)
    # Injecting customized preprocessing functions,
    # check the document or sample code below for API.
    extractor.model.preproc_func = preproc_func

    rmdir(save_dir)
    output_map_list = extractor.predict(
        wsi_paths,
        msk_paths,
        mode="wsi",
        ioconfig=ioconfig,
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=save_dir,
    )

    # Rename output files
    for input_path, output_path in output_map_list:
        input_name = Path(input_path).stem

        output_parent_dir = pathlib.Path(output_path).parent

        src_path = pathlib.Path(f"{output_path}.position.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.position.npy")
        src_path.rename(new_path)

        src_path = pathlib.Path(f"{output_path}.features.0.npy")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.features.npy")
        src_path.rename(new_path)

    return output_map_list


def get_cell_compositions(
    wsi_path: str,
    mask_path: str,
    inst_pred_path: str,
    save_dir: str,
    num_types: int = 6,
    patch_input_shape: tuple[int] = (512, 512),
    stride_shape: tuple[int] = (512, 512),
    resolution: Resolution = 0.25,
    units: Units = "mpp",
) -> None:
    """Estimates cellular composition."""
    reader = WSIReader.open(wsi_path)
    inst_pred = joblib.load(inst_pred_path)
    # Convert to {key: int, value: dict}
    inst_pred = {i: v for i, (_, v) in enumerate(inst_pred.items())}

    inst_boxes = [v["box"] for v in inst_pred.values()]
    inst_boxes = np.array(inst_boxes)

    geometries = [shapely_box(*bounds) for bounds in inst_boxes]
    spatial_indexer = STRtree(geometries)

    # * Generate patch coordinates (in xy format)
    wsi_shape = reader.slide_dimensions(resolution=resolution, units=units)

    (patch_inputs, _) = PatchExtractor.get_coordinates(
        image_shape=wsi_shape,
        patch_input_shape=patch_input_shape,
        patch_output_shape=patch_input_shape,
        stride_shape=stride_shape,
    )

    # Filter out coords which dont lie in mask
    selected_coord_indices = PatchExtractor.filter_coordinates(
        WSIReader.open(mask_path),
        patch_inputs,
        wsi_shape=wsi_shape,
        min_mask_ratio=0.5,
    )
    patch_inputs = patch_inputs[selected_coord_indices]

    bounds_compositions = []
    for bounds in patch_inputs:
        bounds_ = shapely_box(*bounds)
        indices = [
            geo
            for geo in spatial_indexer.query(bounds_)
            if bounds_.contains(geometries[geo])
        ]
        insts = [inst_pred[v]["type"] for v in indices]
        uids, freqs = np.unique(insts, return_counts=True)
        # A bound may not contain all types, hence, to sync
        # the array and placement across all types, we create
        # a holder then fill the count within.
        holder = np.zeros(num_types, dtype=np.int16)
        holder[uids.astype(int)] = freqs
        bounds_compositions.append(holder)
    bounds_compositions = np.array(bounds_compositions)

    base_name = pathlib.Path(wsi_path).stem
    # Output in the same saving protocol for construct graph
    np.save(f"{save_dir}/{base_name}.position.npy", patch_inputs)
    np.save(f"{save_dir}/{base_name}.features.npy", bounds_compositions)


def extract_composition_features(
    wsi_paths: list[str],
    msk_paths: list[str],
    save_dir: str,
    preproc_func: Callable,
) -> list:
    """Extract cellular composition features."""
    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        batch_size=16,
        num_postproc_workers=4,
        num_loader_workers=4,
    )
    # bigger tile shape for postprocessing performance
    inst_segmentor.ioconfig.tile_shape = (4000, 4000)
    # Injecting customized preprocessing functions,
    # check the document or sample codes below for API
    inst_segmentor.model.preproc_func = preproc_func

    rmdir(save_dir)
    output_map_list = inst_segmentor.predict(
        wsi_paths,
        msk_paths,
        mode="wsi",
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=save_dir,
    )
    # Rename output files of toolbox
    output_paths = []
    for input_path, output_path in output_map_list:
        input_name = pathlib.Path(input_path).stem

        output_parent_dir = pathlib.Path(output_path).parent

        src_path = pathlib.Path(f"{output_path}.dat")
        new_path = pathlib.Path(f"{output_parent_dir}/{input_name}.dat")
        src_path.rename(new_path)
        output_paths.append(new_path)

    # TODO(TBC): Parallelize this if possible  # noqa: TD003, FIX002
    for idx, path in enumerate(output_paths):
        get_cell_compositions(wsi_paths[idx], msk_paths[idx], path, save_dir)
    return output_paths


target_image = stain_norm_target()
stain_normaliser = get_normalizer("reinhard")
stain_normaliser.fit(target_image)


def stain_norm_func(img: np.ndarray) -> np.ndarray:
    """Helper function to perform stain normalization."""
    return stain_normaliser.transform(img)


USE_PRE_GENERATED = False
# ! for using pre-generated graph
NUM_NODE_FEATURES = 4
FEATURE_MODE = "cnn" # 1.5 min
# FEATURE_MODE = "composition" # ~1 hr
WSI_GRAPH_DIR = WORKSPACE_DIR/FEATURE_MODE

# rmdir(WSI_GRAPH_DIR)
if WSI_GRAPH_DIR.exists():
    raise Exception(f"{WSI_GRAPH_DIR} already exists")
mkdir(WSI_GRAPH_DIR)

def construct_graph(wsi_name: str, save_path: str | Path) -> None:
    """Construct graph for one WSI and save to file."""
    positions = np.load(f"{WSI_GRAPH_DIR}/{wsi_name}.position.npy")
    features = np.load(f"{WSI_GRAPH_DIR}/{wsi_name}.features.npy")
    graph_dict = SlideGraphConstructor.build(
        positions[:, :2],
        features,
        feature_range_thresh=None,
    )

    # Write a graph to a JSON file
    with save_path.open("w") as handle:
        graph_dict = {k: v.tolist() for k, v in graph_dict.items()}
        json.dump(graph_dict, handle)

if FEATURE_MODE == "composition":
    output_list = extract_composition_features(
        [WSI_PATH],
        [MSK_PATH],
        WSI_GRAPH_DIR,
        stain_norm_func,
    )
else:
    output_list = extract_deep_features(
        [WSI_PATH],
        [MSK_PATH],
        WSI_GRAPH_DIR,
        stain_norm_func,
    )

# Build up the graph
wsi_name = pathlib.Path(WSI_PATH).stem
GRAPH_PATH = WSI_GRAPH_DIR / f"{wsi_name}.json"
construct_graph(wsi_name, GRAPH_PATH)

# ### Visualize a Graph
# 
# It is always a good practice to validate data and any results visually.
# Here, we plot the sample graph upon its WSI thumbnail. When plotting,
# the nodes within the graph will often be at different resolution
# from the resolution at which we want to visualize them. We scale their coordinates
# to the target resolution before plotting. We provide `NODE_RESOLUTION` and `PLOT_RESOLUTION` variables
# respectively as the resolution of the node and the resolution at which to plot the graph.
# 
# 

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
plt.savefig(WSI_GRAPH_DIR / f"{wsi_name}.png")
