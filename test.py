import pandas as pd
from os import path as osp

import torch
from tiatoolbox.utils.misc import select_device
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib

# from src.intensity import add_features_and_create_new_dicts
# from src.featureextraction import get_cell_features, add_features_and_create_new_dicts
from src.train import stratified_split, recur_find_ext, run_once, rm_n_mkdir ,reset_logging
# from src.graph_construct import create_graph_with_pooled_patch_nodes, get_pids_labels_for_key

ON_GPU = False
device = select_device(on_gpu=ON_GPU)

SEED = 5
random.seed(SEED)
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


BASEDIR = '/home/amrit/data/proj_data/MLG_project/DLBCL-Morph'


# STAIN = 'MYC'
# STAIN = 'BCL2'
STAIN = 'HE'

FIDIR = f'{BASEDIR}/outputs'
CLINPATH = f'{BASEDIR}/clinical_data_cleaned.csv'
ANNPATH = f'{BASEDIR}/annotations_clean.csv'
FEATSDIR = f'{BASEDIR}/outputs/files/{STAIN}'
FEATSCALERPATH = f"{FEATSDIR}/0_feat_scaler.npz"
PATCH_SIZE = 224
OUTPUT_SIZE = PATCH_SIZE*8

WORKSPACE_DIR = Path(BASEDIR)
# GRAPH_DIR = WORKSPACE_DIR / f"graphs{STAIN}" 
# LABELS_PATH = WORKSPACE_DIR / "graphs/0_labels.txt"


# Graph construction
# PATCH_SIZE = 300
SKEW_NOISE = 0.0001
MIN_CELLS_PER_PATCH = 10
CONNECTIVITY_DISTANCE = 500

LABEL_TYPE = 'multilabel' #'OS' #

GRAPHSDIR = Path(f'{BASEDIR}/graphs/{STAIN}')
LABELSPATH = f'{BASEDIR}/graphs/{STAIN}_labels.json'

NUM_EPOCHS = 100
NUM_NODE_FEATURES = 128
NCLASSES = 3

TRAIN_DIR = WORKSPACE_DIR / "training"
SPLIT_PATH = TRAIN_DIR / f"splits_{STAIN}.dat"
RUN_OUTPUT_DIR = TRAIN_DIR / f"session_{STAIN}_{datetime.now().strftime('%m_%d_%H_%M')}"
RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = RUN_OUTPUT_DIR / "model"


# # we must define the function after training/loading
# def nodes_preproc_func(node_features: np.ndarray) -> np.ndarray:
#     """Pre-processing function for nodes."""
#     return node_scaler.transform(node_features)
nodes_preproc_func = None


splits = joblib.load(SPLIT_PATH)
loader_kwargs = {
    "num_workers": 6,
    "batch_size": 6,
}
# arch_kwargs = {
#     "dim_features": NUM_NODE_FEATURES,
#     "dim_target": NCLASSES,
#     "layers": [32, 32, 16, 8],
#     "dropout": 0.3,
#     "pooling": "mean",
#     "conv": "EdgeConv",
#     "aggr": "max",
# }

if LABEL_TYPE == "OS":
    NCLASSES = 3
else:
    NCLASSES = 6
      
arch_kwargs = {
        "dim_features": NUM_NODE_FEATURES,
        "dim_target": NCLASSES,
        "layers": [64, 32, 32],
        "dropout": 0.1,
        "pooling": "mean",
        "conv": "EdgeConv",
        "aggr": "max",
        "CLASSIFICATION_TYPE" : LABEL_TYPE
}

optim_kwargs = {
    "lr": 5.0e-3,
    "weight_decay": 1.0e-4,
}

NUM_EPOCHS = 100
# if not MODEL_DIR.exists() or True:
for split_idx, split in enumerate(splits):
    new_split = {
        "train": split["train"],
        "infer-train": split["train"],
        "infer-valid-A": split["valid"],
        "infer-valid-B": split["test"],
    }
    MODEL_DIR = Path(MODEL_DIR) 
    split_save_dir = MODEL_DIR / f"{split_idx:02d}/"
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
        GRAPH_DIR=GRAPHSDIR,
        LABEL_TYPE = LABEL_TYPE
    )