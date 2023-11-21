
from typing import Union
from pathlib import Path
import joblib
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from torch.nn import BatchNorm1d, Linear, ReLU

from tiatoolbox.utils.misc import select_device

from torch_geometric.nn import (
    EdgeConv,
    GINConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

class SlideGraphArch(nn.Module):
    """Define SlideGraph architecture."""

    def __init__(
        self: nn.Module,
        dim_features: int,
        dim_target: int,
        layers: "tuple[int, int]",
        pooling: str = "max",
        dropout: int = 0.0,
        conv: str = "GINConv",
        *,
        gembed: bool = False,
        **kwargs: dict,
    ) -> None:
        """Initialize SlideGraphArch."""
        super().__init__()
        if layers is None:
            layers = [6, 6]
        self.dropout = dropout
        self.embeddings_dim = layers
        self.num_layers = len(self.embeddings_dim)
        self.nns = []
        self.convs = []
        self.linears = []
        self.pooling = {
            "max": global_max_pool,
            "mean": global_mean_pool,
            "add": global_add_pool,
        }[pooling]
        # If True then learn a graph embedding for final classification
        # (classify pooled node features), otherwise pool node decision scores.
        self.gembed = gembed

        conv_dict = {"GINConv": [GINConv, 1], "EdgeConv": [EdgeConv, 2]}
        if conv not in conv_dict:
            msg = f'Not support `conv="{conv}".'
            raise ValueError(msg)

        def create_linear(in_dims: int, out_dims: int) -> Linear:
            return nn.Sequential(
                Linear(in_dims, out_dims),
                BatchNorm1d(out_dims),
                ReLU(),
            )

        input_emb_dim = dim_features
        out_emb_dim = self.embeddings_dim[0]
        self.first_h = create_linear(input_emb_dim, out_emb_dim)
        self.linears.append(Linear(out_emb_dim, dim_target))

        input_emb_dim = out_emb_dim
        for out_emb_dim in self.embeddings_dim[1:]:
            conv_class, alpha = conv_dict[conv]
            subnet = create_linear(alpha * input_emb_dim, out_emb_dim)
            # ! this variable should be removed after training integrity checking
            self.nns.append(subnet)  # <--| as it already within ConvClass
            self.convs.append(conv_class(self.nns[-1], **kwargs))
            self.linears.append(Linear(out_emb_dim, dim_target))
            input_emb_dim = out_emb_dim

        self.nns = torch.nn.ModuleList(self.nns)
        self.convs = torch.nn.ModuleList(self.convs)
        # Has got one more for initial input, what does this mean
        self.linears = torch.nn.ModuleList(self.linears)

        # Auxilary holder for external model, these are saved separately from torch.save
        # as they can be sklearn model etc.
        self.aux_model = {}

    def save(self: nn.Module, path: Union[str, Path], aux_path: Union[str, Path]) -> None:
        """Save torch model."""
        state_dict = self.state_dict()
        torch.save(state_dict, path)
        joblib.dump(self.aux_model, aux_path)

    def load(self: nn.Module, path: Union[str, Path], aux_path: Union[str, Path], map_location=None) -> None:
        """Load torch model."""
        state_dict = torch.load(path, map_location=map_location)
        self.load_state_dict(state_dict)
        self.aux_model = joblib.load(aux_path)

    def forward(
        self: nn.Module,
        data: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Torch model forward function."""
        feature, edge_index, batch = (
            data.x,
            data.edge_index.type(torch.int64),
            data.batch,
        )

        wsi_prediction = 0
        pooling = self.pooling
        node_prediction = 0

        feature = self.first_h(feature)
        for layer in range(self.num_layers):
            if layer == 0:
                node_prediction_sub = self.linears[layer](feature)
                node_prediction += node_prediction_sub
                node_pooled = pooling(node_prediction_sub, batch)
                wsi_prediction_sub = F.dropout(
                    node_pooled,
                    p=self.dropout,
                    training=self.training,
                )
                wsi_prediction += wsi_prediction_sub
            else:
                feature = self.convs[layer - 1](feature, edge_index)
                if not self.gembed:
                    node_prediction_sub = self.linears[layer](feature)
                    node_prediction += node_prediction_sub
                    node_pooled = pooling(node_prediction_sub, batch)
                    wsi_prediction_sub = F.dropout(
                        node_pooled,
                        p=self.dropout,
                        training=self.training,
                    )
                else:
                    node_pooled = pooling(feature, batch)
                    node_prediction_sub = self.linears[layer](node_pooled)
                    wsi_prediction_sub = F.dropout(
                        node_prediction_sub,
                        p=self.dropout,
                        training=self.training,
                    )
                wsi_prediction += wsi_prediction_sub
        return wsi_prediction, node_prediction

    # Run one single step
    @staticmethod
    def train_batch(
        model: nn.Module,
        batch_data: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        on_gpu: bool = False,
    ) -> list:
        """Helper function for model training."""
        device = select_device(on_gpu=on_gpu)
        wsi_graphs = batch_data["graph"].to(device)
        wsi_labels = batch_data["label"].to(device)
        model = model.to(device)

        # Data type conversion
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        # Not an RNN so does not accumulate
        model.train()
        optimizer.zero_grad()

        wsi_output, _ = model(wsi_graphs)
        loss = torch.nn.functional.cross_entropy(wsi_output, wsi_labels)

        # Backprop and update
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().numpy()
        assert not np.isnan(loss)  # noqa: S101
        wsi_labels = wsi_labels.cpu().numpy()
        return [loss, wsi_output, wsi_labels]

    # Run one inference step
    @staticmethod
    def infer_batch(
        model: nn.Module,
        batch_data: torch.Tensor,
        on_gpu: bool = False,
    ) -> list:
        """Model inference."""
        device = select_device(on_gpu=on_gpu)
        wsi_graphs = batch_data["graph"].to(device)
        model = model.to(device)

        # Data type conversion
        wsi_graphs.x = wsi_graphs.x.type(torch.float32)

        # Inference mode
        model.eval()
        # Do not compute the gradient (not training)
        with torch.inference_mode():
            wsi_output, _ = model(wsi_graphs)

        wsi_output = wsi_output.cpu().numpy()
        # predict label
        wsi_output = np.argmax(wsi_output, axis=1)

        # Output should be a single tensor or scalar
        if "label" in batch_data:
            wsi_labels = batch_data["label"]
            wsi_labels = wsi_labels.cpu().numpy()
            return wsi_output, wsi_labels
        return [wsi_output]

