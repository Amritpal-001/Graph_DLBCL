
from intensity import get_cell_features, get_saliency_map
from tiatoolbox.utils.visualization import overlay_prediction_contours, random_colors
from tiatoolbox.utils.visualization import plot_graph

import numpy as np
import joblib
import cv2
from matplotlib import pyplot as plt
from copy import deepcopy
import json

# pid = 13903; did = 560
pid = 13901; did = 532

imgpath = f"workspace/ours/MYC_upload2/images/{pid}/{pid}_MYC_TA232_1792_{did}.png"
# datpath = f"workspace/ours/MYC_upload2/files/MYC/{pid}/{did}/0.dat"
datpath = f"workspace/ours/feats/{pid}_MYC_{did}.dat"
cf_graphpath = f"workspace/ours/graphs/{pid}_MYC_{did}.json"
rn_graphpath = f"workspace/ours/graphs_resnet/{pid}_MYC_{did}.json"

tile_img = cv2.imread(imgpath)
celldata_dict = joblib.load(datpath)
# up_celldata_dict = joblib.load(updatpath)

    # inst_dict=up_celldata_dict,
overlay_image_cells = overlay_prediction_contours(
    canvas=tile_img,
    inst_dict=celldata_dict,
    draw_dot=True,
    type_colours=random_colors(len(celldata_dict), bright=True),
    line_thickness=10,
)

with open(cf_graphpath, 'r') as f:
    cf_graph_dict = json.load(f)

edge_list = [_ for _ in zip(*cf_graph_dict['edge_index'])]
overlay_image_graph = tile_img.copy()
plot_graph(
    canvas=overlay_image_graph,
    nodes=cf_graph_dict['coordinates'],
    edges=edge_list,
    node_size=10,
)

from tiatoolbox.data import stain_norm_target
from tiatoolbox.tools.stainnorm import get_normalizer

target_image = stain_norm_target()
stain_normaliser = get_normalizer("reinhard")
stain_normaliser.fit(target_image)
def stain_norm_func(img: np.ndarray) -> np.ndarray:
    """Helper function to perform stain normalization."""
    return stain_normaliser.transform(img)

# stain normalized image
proc_img = stain_norm_func(tile_img)

with open(rn_graphpath, 'r') as f:
    rn_graph_dict = json.load(f)

edge_list = [_ for _ in zip(*rn_graph_dict['edge_index'])]
rn_overlay_image_graph = tile_img.copy()
plot_graph(
    canvas=rn_overlay_image_graph,
    nodes=rn_graph_dict['coordinates'],
    edges=edge_list,
    node_size=10,
)

# tile_img = np.ones((256, 100, 3), dtype=np.uint8) * 128
# tile_img[:, :25, :] = 0
# celldata_dict = {
#     'xyz': {
#         'contour': np.array([[0, 0],[0, 50],[25, 75],[50, 25],[25, 0],]) + np.array([40, 0]),
#         'centroid': [15, 15],
#     },
# }

# sal_img = get_saliency_map(tile_img)
# updict = get_cell_features(celldata_dict['xyz'], tile_img, sal_img, original=False)
# print(updict)

# # count num valid cells
# celldata_dict_new = deepcopy(celldata_dict)
# for k, v in celldata_dict.items():
#     keep = False
#     bmin = np.min(v['contour'], axis=0)[::-1]
#     bmax = np.max(v['contour'], axis=0)[::-1]
#     if (tile_img.shape[:2] > bmax).all():
#         keep = True
#     if np.array(v['contour']).shape[0] > 5:
#         keep = True

#     if not keep:
#         del celldata_dict_new[k]
# print(len(celldata_dict_new), len(celldata_dict))

# overlay_image_graph = overlay_prediction_contours(
#     canvas=tile_img,
#     inst_dict=celldata_dict_new,
#     draw_dot=True,
#     type_colours=random_colors(len(celldata_dict), bright=True),
#     line_thickness=10,
# )

fig = plt.figure(figsize=(10,10))
plt.subplot(3, 2, 1), plt.imshow(tile_img), plt.axis("off")
plt.subplot(3, 2, 3), plt.imshow(overlay_image_cells), plt.axis("off")
plt.subplot(3, 2, 5), plt.imshow(overlay_image_graph), plt.axis("off")

plt.subplot(3, 2, 4), plt.imshow(proc_img), plt.axis("off")
plt.subplot(3, 2, 6), plt.imshow(rn_overlay_image_graph), plt.axis("off")

plt.tight_layout()
plt.savefig('workspace/ours/sample.png')
