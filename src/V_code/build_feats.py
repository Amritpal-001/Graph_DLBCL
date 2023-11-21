
import pandas as pd
import os
import shutil
from os import path as osp
from src.intensity import add_features_and_create_new_dicts


BASEDIR = '/home/amrit/data/proj_data/MLG_project/DLBCL-Morph/outputs/files/'


STAIN = 'MYC'
FIDIR = f'{BASEDIR}/MYC/'
CLINPATH = f'{BASEDIR}/clinical_data_cleaned.csv'
ANNPATH = f'{BASEDIR}/annotations_clean.csv'
FEATSDIR = f'{BASEDIR}/feats'
FEATSCALERPATH = f"{FEATSDIR}/0_feat_scaler.npz"
PATCH_SIZE = 224
OUTPUT_SIZE = PATCH_SIZE*8

##########
# read annotation csv, filter, and process intensity features
df = pd.read_csv(ANNPATH)

df = df[df['stain'] == STAIN]
df['area'] = (df['xe'] - df['xs']) *  (df['ye'] - df['ys'])/10000
df = df[df['area'] >= 150]  
df = df[df['xs']  >=0 ]
df = df[df['ys']  >=0 ]
df = df[df['xe']  >=0 ]
df = df[df['ye']  >=0 ]

df = df.reset_index()
##########

###############
# add intensity features
start_index = 0
end_index = len(df.index)

datpaths = []
imgpaths = []
updatpaths = []

# for index in range(start_index, 1):
for index in range(start_index, end_index):
    df_index = df['index'][index]
    patient_id = df['patient_id'][index]
    stain = df['stain'][index]
    tma_id = df['tma_id'][index]
    unique_id = str(patient_id) + '_' + stain + '_' + str(df_index)

    img_file_name = f"{FIDIR}/images/{patient_id}/{patient_id}_{stain}_{tma_id}_{OUTPUT_SIZE}_{df_index}.png"
    dat_file_name = f"{FIDIR}/files/{stain}/{patient_id}/{df_index}/0.dat"
    updat_file_name = f"{FEATSDIR}/{unique_id}.dat"

    datpaths.append(dat_file_name)
    imgpaths.append(img_file_name)
    updatpaths.append(updat_file_name)

# shutil.rmtree(FEATSDIR)
if not osp.exists(FEATSDIR):
    os.makedirs(FEATSDIR)
    add_features_and_create_new_dicts(datpaths, imgpaths, updatpaths)
###############

# get all cell features in all datapoints and get normalization stats
print('Building global cell feature normalizer')
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
import numpy as np

gns = StandardScaler()
for featpath in tqdm(updatpaths):
    celldatadict = joblib.load(featpath)
    cellsfeats = np.array([v['intensity_feats'] for k, v in celldatadict.items()])
    gns.partial_fit(cellsfeats)

np.savez(FEATSCALERPATH, mean=gns.mean_, var=gns.var_)

# dd = np.load(FEATSCALERPATH)
# print(dd['mean'], gns.mean_)
# print(dd['var'], gns.var_)
