
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


SKEW_NOISE  = 0.0001


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


def get_pids_multilabels_for_key(df, key_list, nclasses=3, idkey='patient_id'):
    df = df[[idkey]+ key_list]

    for key in key_list:
        ckey = key+'_class'
        df[ckey] = int(0)
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

global_patch_stats = []

def get_patch_pooled_positions_features(celldatadict, patch_size, cell_feat_norm_stats, MIN_CELLS_PER_PATCH):

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

def create_graph_with_pooled_patch_nodes(featpaths, labels, outgraphpaths, patch_size, cell_feat_norm_stats , MIN_CELLS_PER_PATCH, CONNECTIVITY_DISTANCE):

    def process_per_file_group(idx):
        featpath = featpaths[idx]
        label = labels[idx]
        outgraphpath = outgraphpaths[idx]

        celldatadict = joblib.load(featpath)
        positions, features = get_patch_pooled_positions_features(celldatadict, patch_size, cell_feat_norm_stats,MIN_CELLS_PER_PATCH)

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
                # print(outgraphpath)
                graph_dict = {k: v.tolist() for k, v in graph_dict.items()}
                graph_dict['y'] = label
                json.dump(graph_dict, handle)

    joblib.Parallel(4)(
        joblib.delayed(process_per_file_group)(fidx)
        for fidx in tqdm(range(len(featpaths)), disable=False)
    )
    # for fidx in tqdm(range(len(featpaths)), disable=False):
    #     process_per_file_group(fidx)


def create_graph_with_pooled_patch_nodes_with_survival_data(featpaths, labels, survival_events, survival_time, outgraphpaths, patch_size, cell_feat_norm_stats , MIN_CELLS_PER_PATCH, CONNECTIVITY_DISTANCE):

    def process_per_file_group(idx):
        featpath = featpaths[idx]
        label = labels[idx]
        surv_event =survival_events[idx]
        surv_time =survival_time[idx]

        outgraphpath = outgraphpaths[idx]

        celldatadict = joblib.load(featpath)
        positions, features = get_patch_pooled_positions_features(celldatadict, patch_size, cell_feat_norm_stats,MIN_CELLS_PER_PATCH)

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
                # print(outgraphpath)
                graph_dict = {k: v.tolist() for k, v in graph_dict.items()}
                graph_dict['y'] = label
                graph_dict['surv_event'] = surv_event
                graph_dict['surv_time'] = surv_time

                json.dump(graph_dict, handle)

    joblib.Parallel(4)(
        joblib.delayed(process_per_file_group)(fidx)
        for fidx in tqdm(range(len(featpaths)), disable=False)
    )
    # for fidx in tqdm(range(len(featpaths)), disable=False):
    #     process_per_file_group(fidx)
