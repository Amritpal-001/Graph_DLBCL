
from __future__ import annotations

import torch
print('torch.__file__', torch.__file__)
print('torch.__version__', torch.__version__)
print('torch.cuda.is_available()', torch.cuda.is_available())
print('torch.version.cuda', torch.version.cuda)

import torch.nn.functional as F  # noqa: N812
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import BatchNorm1d, Linear, ReLU

import torch_geometric
print('torch_geometric.__version__', torch_geometric.__version__)

from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    EdgeConv,
    GINConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

import tiatoolbox
print('tiatoolbox.__version__', tiatoolbox.__version__)

from tiatoolbox import logger
from tiatoolbox.data import stain_norm_target
from tiatoolbox.models import (
    DeepFeatureExtractor,
    IOSegmentorConfig,
    NucleusInstanceSegmentor,
)
from tiatoolbox.models.architecture.vanilla import CNNBackbone
from tiatoolbox.tools.graph import SlideGraphConstructor
from tiatoolbox.tools.patchextraction import PatchExtractor
from tiatoolbox.tools.stainnorm import get_normalizer
from tiatoolbox.utils.misc import download_data, select_device
from tiatoolbox.utils.visualization import plot_graph
from tiatoolbox.wsicore.wsireader import WSIReader
