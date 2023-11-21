
from src.intensity import get_cell_features, get_saliency_map
from tiatoolbox.utils.visualization import overlay_prediction_contours, random_colors
from tiatoolbox.utils.visualization import plot_graph

import numpy as np
import joblib
import cv2
from matplotlib import pyplot as plt
from copy import deepcopy
import json

pid = 13903
did = 560
imgpath = f"workspace/ours/MYC_upload2/images/{pid}/{pid}_MYC_TA232_1792_{did}.png"
datpath = f"workspace/ours/MYC_upload2/files/MYC/{pid}/{did}/0.dat"
updatpath = f"workspace/ours/feats/{pid}_MYC_{did}.dat"
graphpath = f"workspace/ours/graphs/{pid}_MYC_{did}.json"

tile_img = cv2.imread(imgpath)
celldata_dict = joblib.load(datpath)
up_celldata_dict = joblib.load(updatpath)
with open(graphpath, 'r') as f:
    graph_dict = json.load(f)

    # inst_dict=celldata_dict,
overlay_image_1 = overlay_prediction_contours(
    canvas=tile_img,
    inst_dict=up_celldata_dict,
    draw_dot=True,
    type_colours=random_colors(len(celldata_dict), bright=True),
    line_thickness=10,
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

# overlay_image_2 = overlay_prediction_contours(
#     canvas=tile_img,
#     inst_dict=celldata_dict_new,
#     draw_dot=True,
#     type_colours=random_colors(len(celldata_dict), bright=True),
#     line_thickness=10,
# )

edge_list = [_ for _ in zip(*graph_dict['edge_index'])]
overlay_image_2 = tile_img.copy()
plot_graph(
    canvas=overlay_image_2,
    nodes=graph_dict['coordinates'],
    edges=edge_list,
    node_size=10,
)

fig = plt.figure(figsize=(15,15))
ax1 = plt.subplot(2, 2, 1), plt.imshow(tile_img), plt.axis("off")
ax2 = plt.subplot(2, 2, 2), plt.imshow(overlay_image_1), plt.axis("off")
ax2 = plt.subplot(2, 2, 4), plt.imshow(overlay_image_2), plt.axis("off")
plt.tight_layout()
plt.savefig('workspace/ours/sample.png')
