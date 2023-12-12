
import numpy as np
import pandas as pd
import glob
from os import path as osp

from build_graphs import get_pids_labels_for_key
from tiatoolbox.data import stain_norm_target
from tiatoolbox.tools.stainnorm import get_normalizer
from tiatoolbox.models.architecture.vanilla import CNNBackbone
from tiatoolbox.models import DeepFeatureExtractor, IOSegmentorConfig

from sklearn.preprocessing import StandardScaler

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


if __name__ == "__main__":

    if not osp.exists(FEATPATHSPATH):
        # divide into quantiles
        df = pd.read_csv(CLINPATH)
        df_labels = get_pids_labels_for_key(df, key='OS', nclasses=3)

        featpaths = np.sort(glob.glob(f'{FEATSDIR}/*.dat'))
        pids = [int(osp.basename(featpath).split('_')[0]) for featpath in featpaths]
        df_featpaths = pd.DataFrame(zip(pids, featpaths), columns=['patient_id', 'featpath'])
        # merge to find datapoints with graph data and labels
        df_data = df_featpaths.merge(df_labels, on='patient_id')
        # save paths
        df_data.to_csv(FEATPATHSPATH)
    # load paths
    df_data = pd.read_csv(FEATPATHSPATH)
    # df_data = df_data[:2]

    featpaths_data = df_data['featpath'].to_list()
    imgpaths_data = []
    for featpath in featpaths_data:
        fp_pid, fp_stain, fp_dfidx = osp.basename(featpath).split('.')[0].split('_')
        imgpath = f"{FIDIR}/images/{fp_pid}/{fp_pid}_{fp_stain}_*_{OUTPUT_SIZE}_{fp_dfidx}.png"
        ips = glob.glob(imgpath)
        assert len(ips) == 1, imgpath
        imgpaths_data.append(ips[0])

    # shutil.rmtree(FEATSRESNETDIR, ignore_errors=True)
    if not osp.exists(FEATSRESNETDIR):
        target_image = stain_norm_target()
        stain_normaliser = get_normalizer("reinhard")
        stain_normaliser.fit(target_image)
        model = CNNBackbone("resnet50")

        extractor = DeepFeatureExtractor(
            batch_size=8,
            num_loader_workers=4,
            model=model,
        )
        extractor.model.preproc_func = lambda x: stain_normaliser.transform(x)

        # create features
        output_map_list = extractor.predict(
            imgpaths_data,
            None,
            mode="tile",
            ioconfig=IOSegmentorConfig(
                input_resolutions=[{"units": "baseline", "resolution": 1.0},],
                output_resolutions=[{"units": "baseline", "resolution": 1.0},],
                patch_input_shape=[PATCH_SIZE, PATCH_SIZE],
                patch_output_shape=[PATCH_SIZE, PATCH_SIZE],
                stride_shape=[PATCH_SIZE, PATCH_SIZE],
            ),
            on_gpu=False,
            crash_on_exception=True,
            save_dir=FEATSRESNETDIR,
        )
        output_map_dict = {x: y for x, y in output_map_list}
        featpaths = [output_map_dict[imgpath] for imgpath in imgpaths_data]

        print('Calculating global patch normalizer')
        global_patch_norm = StandardScaler()
        for featpath in featpaths:
            _features = np.load(f"{featpath}.features.0.npy")
            global_patch_norm.partial_fit(_features)
        np.savez(FEATSCALERPATH, mean=global_patch_norm.mean_, var=global_patch_norm.var_)
