
import numpy as np
import joblib
import pandas as pd
from tqdm import tqdm
import glob
import os
import shutil
from os import path as osp
import json

from build_graphs import simple_delaunay

BASEDIR = 'workspace/ours'
STAIN = 'MYC'
FIDIR = f'{BASEDIR}/MYC_upload2'

CLINPATH = f'{BASEDIR}/clinical_data_cleaned.csv'
ANNPATH = f'{BASEDIR}/annotations_clean.csv'
FEATSDIR = f'{BASEDIR}/feats'
FEATPATHSPATH = f'{BASEDIR}/featpaths.csv'

GRAPHSDIR = f'{BASEDIR}/graphs_resnet'
LABELSPATH = f'{BASEDIR}/graphs_resnet/0_labels.txt'
FEATSRESNETDIR = f'{BASEDIR}/feats_resnet'
FEATSCALERPATH = f"{FEATSRESNETDIR}/0_feat_scaler.npz"
PATCH_SIZE = 300
OUTPUT_SIZE = 224*8
CONNECTIVITY_DISTANCE = 500


def get_patch_resnet_positions_features(featpath, patch_norm_stats):
    # [start_x, start_y, end_x, end_y]
    ends_patches = np.load(f"{featpath}.position.npy")
    mid_positions = np.stack([
        (ends_patches[:, 0] + ends_patches[:, 2]) / 2,
        (ends_patches[:, 1] + ends_patches[:, 3]) / 2,
    ], axis=1)

    features = np.load(f"{featpath}.features.0.npy")

    _patchfeats_gmean, _patchfeats_gvar = patch_norm_stats
    pf_gmean = np.expand_dims(_patchfeats_gmean, axis=0)
    pf_gstd = np.expand_dims(np.sqrt(_patchfeats_gvar), axis=0)
    norm_features = (features - pf_gmean) / (pf_gstd + 1e-6)

    return mid_positions, norm_features

def create_graph_with_patch_nodes_resnet(featdirpaths, labels, outgraphpaths, patch_norm_stats):

    def process_per_file_group(idx):
        featdirpath = featdirpaths[idx]
        label = labels[idx]
        outgraphpath = outgraphpaths[idx]

        positions, features = get_patch_resnet_positions_features(featdirpath, patch_norm_stats)

        # graph cannot be constructed with only four patches
        try:
            graph_dict = simple_delaunay(
                positions[:, :2],
                features,
                connectivity_distance=CONNECTIVITY_DISTANCE,
            )
        except Exception as e:
            print('Skipping', imgpath, 'due to', e)
        else:
            # Write a graph to a JSON file
            with open(outgraphpath, 'w+') as handle:
                graph_dict = {k: v.tolist() for k, v in graph_dict.items()}
                graph_dict['y'] = label
                json.dump(graph_dict, handle)

    joblib.Parallel(4)(
        joblib.delayed(process_per_file_group)(fidx)
        for fidx in tqdm(range(len(featdirpaths)), disable=False)
    )
    # [
    #     process_per_file_group(fidx)
    #     for fidx in tqdm(range(len(featdirpaths)), disable=False)
    # ]

if __name__ == "__main__":

    # load paths
    # for generation of this file check build_feats_resnet.py
    df_data = pd.read_csv(FEATPATHSPATH)
    # df_data = df_data[:2]

    # paths to custom features
    featpaths_data = df_data['featpath'].to_list()
    imgpaths_data = []
    for featpath in featpaths_data:
        fp_pid, fp_stain, fp_dfidx = osp.basename(featpath).split('.')[0].split('_')
        imgpath = f"{FIDIR}/images/{fp_pid}/{fp_pid}_{fp_stain}_*_{OUTPUT_SIZE}_{fp_dfidx}.png"
        ips = glob.glob(imgpath)
        assert len(ips) == 1, imgpath
        imgpaths_data.append(ips[0])
    pids = [int(osp.basename(featpath).split('_')[0]) for featpath in featpaths_data]
    labels_data = df_data['OS_class'].to_list()
    outgraphpaths_data = [f"{GRAPHSDIR}/{osp.basename(featpath).split('.')[0]}.json" for featpath in featpaths_data]

    # list of imagepath, resnetfeatdir
    filemap = joblib.load(f"{FEATSRESNETDIR}/file_map.dat")
    # dict with relative paths
    filemap = {x : y for x, y in filemap}
    featdirpaths_data = [f"workspace/{filemap[x].split('workspace/')[1]}" for x in imgpaths_data]

    # read normalizer stats from file and pass to fn
    dd = np.load(FEATSCALERPATH)
    patch_norm_stats = (dd['mean'], dd['var'])

    # create final graphs data
    # shutil.rmtree(GRAPHSDIR, ignore_errors=True)
    if not osp.exists(GRAPHSDIR):
        os.makedirs(GRAPHSDIR)
        create_graph_with_patch_nodes_resnet(
            featdirpaths_data,
            labels_data,
            outgraphpaths_data,
            patch_norm_stats,
        )

        # save labels
        labels_dict = {osp.basename(graphpath): label for graphpath, label in zip(outgraphpaths_data, labels_data)}
        with open(LABELSPATH, 'w') as f:
            json.dump(labels_dict, f)
