
MAINPATTERN = "workspace/ours/training_*/**/stats.json"

import joblib
import glob
import numpy as np
from utils import load_json
from pathlib import Path
from tabulate import tabulate

fpaths = glob.glob(MAINPATTERN, recursive=True)
# print(fpaths)

data = []
for fpath in fpaths:
    toks = fpath.split('/')
    modelname = toks[2] + " " + toks[3]

    stats_all = load_json(Path(fpath))
    
    maxitr = np.sort([int(x) for x in stats_all.keys()])[-2]
    stats = stats_all[str(maxitr)]
    
    # print(modelname, stats)
    # print(modelname, stats['infer-train-accuracy'], stats['infer-valid-B-accuracy'])
    data.append([modelname, maxitr, stats['infer-train-accuracy'], stats['infer-valid-B-accuracy']])

print(tabulate(data, ['Name', 'Iter', 'Train', 'Test']))

