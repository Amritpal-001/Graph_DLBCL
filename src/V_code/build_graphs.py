
import numpy as np
import joblib
import pandas as pd
from tqdm import tqdm
import glob
import os
import shutil
from os import path as osp
import json
import cv2
from scipy.stats import skew
import colorsys
import random
from typing import Dict, List, Tuple, Union

from tiatoolbox.tools.graph import delaunay_adjacency, affinity_to_edge_index
from matplotlib import pyplot as plt

BASEDIR = 'workspace/ours'
STAIN = 'MYC'
FIDIR = f'{BASEDIR}/MYC_upload2'
CLINPATH = f'{BASEDIR}/clinical_data_cleaned.csv'
ANNPATH = f'{BASEDIR}/annotations_clean.csv'
FEATSDIR = f'{BASEDIR}/feats'
FEATSCALERPATH = f"{FEATSDIR}/0_feat_scaler.npz"
GRAPHSDIR = f'{BASEDIR}/graphs'
LABELSPATH = f'{BASEDIR}/graphs/0_labels.txt'
PATCH_SIZE = 300
SKEW_NOISE = 0.0001
MIN_CELLS_PER_PATCH = 2
CONNECTIVITY_DISTANCE = 500

global_patch_stats = []

def get_pids_labels_for_key(df, key='OS', nclasses=3, idkey='patient_id'):
    ckey = key+'_class'
    df = df[[idkey, key]]
    df[ckey] = int(-1)

    separators = np.linspace(0, 1, nclasses+1)[0:-1]
    for isep, sep in enumerate(separators):
        sepval = df[key].quantile(sep)
        sepmask = df[key] > sepval
        df.loc[sepmask, ckey] = isep
    return df

def get_overall_statistics(features):
    overall_mean = np.mean(features, axis=0).tolist()
    overall_std = np.std(features, axis=0).tolist()
    overall_var = np.var(features, axis=0).tolist()
    overall_skewness = skew(features + np.random.randn(*features.shape)*SKEW_NOISE, axis=0).tolist()

    # Calculate quantiles at percentiles 0.1, 0.2, ..., 0.9
    quantiles = []
    for q in [10,25,75,90]: #range(0, 100, 10):
        quantiles += np.percentile(features, q, axis=0).tolist()

    result_list = overall_mean + overall_std + overall_var + overall_skewness + quantiles
    return result_list

def get_patch_pooled_positions_features(celldatadict, patch_size, cell_feat_norm_stats):

    global global_patch_stats

    _cellfeats_gmean, _cellfeats_gvar = cell_feat_norm_stats
    cf_gmean = np.expand_dims(_cellfeats_gmean, axis=0)
    cf_gstd = np.expand_dims(np.sqrt(_cellfeats_gvar), axis=0)

    cellids = [cellid for cellid, _ in celldatadict.items()]
    cellposs = np.array([fdict['centroid'] for _, fdict in celldatadict.items()])
    gridsize = (cellposs.max(axis=0) // patch_size + 1).astype(int).tolist()

    # create list of patches
    patches_grid_cids = [[[] for _ in range(gridsize[1])] for _ in range(gridsize[0])]

    # sort cells into patches
    for cellid, cellpos in zip(cellids, cellposs):
        patch_coor = (cellpos // patch_size).astype(int).tolist()
        patches_grid_cids[patch_coor[0]][patch_coor[1]].append(cellid)

    patches_list_position = []
    patches_list_features = []

    # calc patch wise stats and add global list
    for i, patches_row_cids in enumerate(patches_grid_cids):
        for j, patch_cids in enumerate(patches_row_cids):
            # if patch has less than eq 1 cell, skip creating it 
            if len(patch_cids) <= MIN_CELLS_PER_PATCH:
                continue

            global_patch_stats.append(len(patch_cids))

            patch_cellposs = np.array([celldatadict[cellid]['centroid'] for cellid in patch_cids])
            patch_cellfeats_raw = np.array([celldatadict[cellid]['intensity_feats'] for cellid in patch_cids])
            patch_cellfeats = (patch_cellfeats_raw - cf_gmean) / cf_gstd

            patch_position = np.mean(patch_cellposs, axis=0)
            patch_features = np.array(get_overall_statistics(patch_cellfeats))

            patches_list_position.append(patch_position)
            patches_list_features.append(patch_features)

    return np.array(patches_list_position), np.array(patches_list_features)

def simple_delaunay(point_centroids, feature_centroids, connectivity_distance=4000):
    adjacency_matrix = delaunay_adjacency(
        points=point_centroids,
        dthresh=connectivity_distance,
    )
    edge_index = affinity_to_edge_index(adjacency_matrix)
    return {
        "x": feature_centroids,
        "edge_index": edge_index.astype(np.int64),
        "coordinates": point_centroids,
    }

def create_graph_with_pooled_patch_nodes(featpaths, labels, outgraphpaths, patch_size, cell_feat_norm_stats):

    def process_per_file_group(idx):
        featpath = featpaths[idx]
        label = labels[idx]
        outgraphpath = outgraphpaths[idx]

        celldatadict = joblib.load(featpath)
        positions, features = get_patch_pooled_positions_features(celldatadict, patch_size, cell_feat_norm_stats)

        # graph cannot be constructed with only four patches
        try:
            graph_dict = simple_delaunay(
                positions[:, :2],
                features,
                connectivity_distance=CONNECTIVITY_DISTANCE,
            )
        except Exception as e:
            print('Skipping', featpath, 'due to', e)
        else:
            # Write a graph to a JSON file
            with open(outgraphpath, 'w+') as handle:
                graph_dict = {k: v.tolist() for k, v in graph_dict.items()}
                graph_dict['y'] = label
                json.dump(graph_dict, handle)

    joblib.Parallel(4)(
        joblib.delayed(process_per_file_group)(fidx)
        for fidx in tqdm(range(len(featpaths)), disable=False)
    )
    # [
    #     process_per_file_group(fidx)
    #     for fidx in tqdm(range(len(featpaths)), disable=False)
    # ]

# divide into quantiles
df = pd.read_csv(CLINPATH)
df_labels = get_pids_labels_for_key(df, key='OS', nclasses=3)

# save paths
featpaths = np.sort(glob.glob(f'{FEATSDIR}/*.dat'))
pids = [int(osp.basename(featpath).split('_')[0]) for featpath in featpaths]
df_featpaths = pd.DataFrame(zip(pids, featpaths), columns=['patient_id', 'featpath'])

# merge to find datapoints with graph data and labels
df_data = df_featpaths.merge(df_labels, on='patient_id')
# df_data = df_data[:12]

featpaths_data = df_data['featpath'].to_list()
labels_data = df_data['OS_class'].to_list()
outgraphpaths_data = [f"{GRAPHSDIR}/{osp.basename(featpath).split('.')[0]}.json" for featpath in featpaths_data]

# save labels
labels_dict = {osp.basename(graphpath): label for graphpath, label in zip(outgraphpaths_data, labels_data)}
with open(LABELSPATH, 'w') as f:
    json.dump(labels_dict, f)

# read normalizer stats from file and pass to fn
dd = np.load(FEATSCALERPATH)
cell_feat_norm_stats = (dd['mean'], dd['var'])

# create final graphs data
# shutil.rmtree(GRAPHSDIR)
if not osp.exists(GRAPHSDIR):
    os.makedirs(GRAPHSDIR)
    create_graph_with_pooled_patch_nodes(
        featpaths_data,
        labels_data,
        outgraphpaths_data,
        PATCH_SIZE,
        cell_feat_norm_stats=cell_feat_norm_stats,
    )

    # wont work in parallel mode
    # print(np.mean(global_patch_stats), np.std(global_patch_stats))
