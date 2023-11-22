import pandas as pd
import os
import shutil
from os import path as osp

import torch
from tiatoolbox.utils.misc import select_device
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
import json
import glob

# from src.intensity import add_features_and_create_new_dicts

from src.featureextraction import get_cell_features, add_features_and_create_new_dicts
from src.train import stratified_split, recur_find_ext, run_once, rm_n_mkdir ,reset_logging
from src.graph_construct import create_graph_with_pooled_patch_nodes, get_pids_labels_for_key


ON_GPU = False
device = select_device(on_gpu=ON_GPU)

SEED = 5
random.seed(SEED)
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


BASEDIR = '/home/amrit/data/proj_data/MLG_project/DLBCL-Morph'

STAIN = 'MYC'
# STAIN = 'BCL2'

FIDIR = f'{BASEDIR}/outputs'
CLINPATH = f'{BASEDIR}/clinical_data_cleaned.csv'
ANNPATH = f'{BASEDIR}/annotations_clean.csv'
FEATSDIR = f'{BASEDIR}/outputs/files/{STAIN}'
FEATSCALERPATH = f"{FEATSDIR}/0_feat_scaler.npz"
PATCH_SIZE = 224
OUTPUT_SIZE = PATCH_SIZE*8

WORKSPACE_DIR = Path(BASEDIR)

SKEW_NOISE = 0.0001
MIN_CELLS_PER_PATCH = 10
CONNECTIVITY_DISTANCE = 500


GRAPHSDIR = Path(f'{BASEDIR}/graphs/{STAIN}')
LABELSPATH = f'{BASEDIR}/graphs/{STAIN}_labels.json'

NUM_EPOCHS = 100
NUM_NODE_FEATURES = 128
NCLASSES = 3

aggr= "max"

for pooling in ['max', 'mean', 'add']:
    for conv in ['EdgeConv', 'GINConv']:
        for dropout in [0.1,0.2, 0.3, 0.4, 0.5]:
            for layers in [[64, 32, 32]]:
                for lr in [ 5.0e-4]:
                    for batch_size in [32,16]:

                        arch_kwargs = {
                            "dim_features": NUM_NODE_FEATURES,
                            "dim_target": NCLASSES,
                            "layers": layers,
                            "dropout": dropout,
                            "pooling": pooling,
                            "conv": conv,
                            "aggr": aggr,
                        }

                        TRAIN_DIR = WORKSPACE_DIR / "training"
                        SPLIT_PATH = TRAIN_DIR / f"splits_{STAIN}.dat"
                        # RUN_OUTPUT_DIR = TRAIN_DIR / f"session_{STAIN}_{conv}_{pooling}_{aggr}_{str(dropout)}_{str(layers)}_{datetime.now().strftime('%m_%d_%H_%M')}"
                        RUN_OUTPUT_DIR = TRAIN_DIR / f"session_{STAIN}_{conv}_{pooling}_{aggr}_{str(dropout)}_{str(batch_size)}_{str(lr)}_{str(layers)}"
                        print(RUN_OUTPUT_DIR)

                        RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                        MODEL_DIR = RUN_OUTPUT_DIR / "model"

                        wsi_paths = recur_find_ext(GRAPHSDIR, [".json"])
                        wsi_names = [Path(v).stem for v in wsi_paths]
                        assert len(wsi_paths) > 0, "No files found."  # noqa: S101

                        NUM_FOLDS = 1
                        TEST_RATIO = 0.2
                        TRAIN_RATIO = 0.8 * 0.7
                        VALID_RATIO = 0.8 * 0.3

                        if SPLIT_PATH and os.path.exists(SPLIT_PATH):
                            splits = joblib.load(SPLIT_PATH)
                        else:
                            x = np.array(wsi_names)
                            with open(LABELSPATH, 'r') as f:
                                labels_dict = json.load(f)
                            print(labels_dict)
                            y = np.array([labels_dict[wsi_name+'.json'] for wsi_name in wsi_names])
                            y[np.where(y==-1)] = 0
                            splits = stratified_split(x, y, TRAIN_RATIO, VALID_RATIO, TEST_RATIO, NUM_FOLDS)
                            joblib.dump(splits, SPLIT_PATH)

                        nodes_preproc_func = None


                        splits = joblib.load(SPLIT_PATH)
                        loader_kwargs = {
                            "num_workers": 6,
                            "batch_size": batch_size,
                        }

                        optim_kwargs = {
                            "lr": lr,
                            "weight_decay": 1.0e-4,
                        }

                        
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
                                GRAPH_DIR=GRAPHSDIR
                            )