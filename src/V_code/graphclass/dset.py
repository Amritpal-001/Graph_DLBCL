
from __future__ import annotations

from typing import Callable
import ujson as json
import numpy as np
import pathlib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import Sampler
from torch_geometric.data import Data, Dataset


class SlideGraphDataset(Dataset):
    """Handling loading graph data from disk.

    Args:
        info_list (list): In case of `train` or `valid` is in `mode`,
            this is expected to be a list of `[uid, label]` . Otherwise,
            it is a list of `uid`. Here, `uid` is used to construct
            `f"{GRAPH_DIR}/{wsi_code}.json"` which is a path points to
            a `.json` file containing the graph structure. By `label`, we mean
            the label of the graph. The format within the `.json` file comes
            from `tiatoolbox.tools.graph`.
        mode (str): This denotes which data mode the `info_list` is in.
        preproc (callable): The prerocessing function for each node
            within the graph.

    """

    def __init__(
        self: Dataset,
        info_list: list,
        mode: str = "train",
        graph_dir: pathlib.Path = None,
        preproc: Callable | None = None,
    ) -> None:
        """Initialize SlideGraphDataset."""
        self.info_list = info_list
        self.mode = mode
        self.graph_dir = graph_dir
        self.preproc = preproc

    def __getitem__(self: Dataset, idx: int) -> Dataset:
        """Get an element from SlideGraphDataset."""
        info = self.info_list[idx]
        if any(v in self.mode for v in ["train", "valid"]):
            wsi_code, label = info
            # torch.Tensor will create 1-d vector not scalar
            label = torch.tensor(label)
        else:
            wsi_code = info

        with (self.graph_dir / f"{wsi_code}.json").open() as fptr:
            graph_dict = json.load(fptr)
        graph_dict = {k: np.array(v) for k, v in graph_dict.items()}

        if self.preproc is not None:
            graph_dict["x"] = self.preproc(graph_dict["x"])

        graph_dict = {k: torch.tensor(v) for k, v in graph_dict.items()}
        graph = Data(**graph_dict)

        if any(v in self.mode for v in ["train", "valid"]):
            return {"graph": graph, "label": label}
        return {"graph": graph}

    def __len__(self: Dataset) -> int:
        """Length of SlideGraphDataset."""
        return len(self.info_list)

    def len(self):
        return self.__len__()

    def get(self, idx):
        return self.__getitem__(idx)


def stratified_split(
    x: list,
    y: list,
    train: float,
    valid: float,
    test: float,
    num_folds: int,
    seed: int = 5,
) -> list:
    """Helper to generate stratified splits.

    Split `x` and `y` in to N number of `num_folds` sets
    of `train`, `valid`, and `test` set in stratified manner.
    `train`, `valid`, and `test` are guaranteed to be mutually
    exclusive.

    Args:
        x (list, np.ndarray):
            List of samples.
        y (list, np.ndarray):
            List of labels, each value is the value
            of the sample at the same index in `x`.
        train (float):
            Percentage to be used for training set.
        valid (float):
            Percentage to be used for validation set.
        test (float):
            Percentage to be used for testing set.
        num_folds (int):
            Number of split generated.
        seed (int):
            Random seed. Default=5.

    Returns:
        A list of splits where each is a dictionary of
        {
            'train': [(sample_A, label_A), (sample_B, label_B), ...],
            'valid': [(sample_C, label_C), (sample_D, label_D), ...],
            'test' : [(sample_E, label_E), (sample_E, label_E), ...],
        }

    """
    assert (  # noqa: S101
        train + valid + test - 1.0 < 1.0e-10  # noqa: PLR2004
    ), "Ratios must sum to 1.0 ."

    outer_splitter = StratifiedShuffleSplit(
        n_splits=num_folds,
        train_size=train + valid,
        random_state=seed,
    )
    inner_splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train / (train + valid),
        random_state=seed,
    )

    x = np.array(x)
    y = np.array(y)
    splits = []
    for train_valid_idx, test_idx in outer_splitter.split(x, y):
        test_x = x[test_idx]
        test_y = y[test_idx]

        # Holder for train_valid set
        x_ = x[train_valid_idx]
        y_ = y[train_valid_idx]

        # Split train_valid into train and valid set
        train_idx, valid_idx = next(iter(inner_splitter.split(x_, y_)))
        valid_x = x_[valid_idx]
        valid_y = y_[valid_idx]

        train_x = x_[train_idx]
        train_y = y_[train_idx]

        # Integrity check
        assert len(set(train_x).intersection(set(valid_x))) == 0  # noqa: S101
        assert len(set(valid_x).intersection(set(test_x))) == 0  # noqa: S101
        assert len(set(train_x).intersection(set(test_x))) == 0  # noqa: S101

        splits.append(
            {
                "train": list(zip(train_x, train_y)),
                "valid": list(zip(valid_x, valid_y)),
                "test": list(zip(test_x, test_y)),
            },
        )
    return splits


def stratified_split_train_test(
    x: list,
    y: list,
    train: float,
    test: float,
    num_folds: int,
    seed: int = 5,
) -> list:
    assert (  # noqa: S101
        train + test - 1.0 < 1.0e-10  # noqa: PLR2004
    ), "Ratios must sum to 1.0 ."

    outer_splitter = StratifiedShuffleSplit(
        n_splits=num_folds,
        train_size=train,
        random_state=seed,
    )

    x = np.array(x)
    y = np.array(y)
    splits = []
    for train_idx, test_idx in outer_splitter.split(x, y):
        test_x = x[test_idx]
        test_y = y[test_idx]

        # Holder for train_valid set
        train_x = x[train_idx]
        train_y = y[train_idx]

        splits.append(
            {
                "train": list(zip(train_x, train_y)),
                "test": list(zip(test_x, test_y)),
            },
        )
    return splits


class StratifiedSampler(Sampler):
    """Sampling the dataset such that the batch contains stratified samples.

    Args:
        labels (list): List of labels, must be in the same ordering as input
            samples provided to the `SlideGraphDataset` object.
        batch_size (int): Size of the batch.

    Returns:
        List of indices to query from the `SlideGraphDataset` object.

    """

    def __init__(self: Sampler, labels: list, batch_size: int = 10) -> None:
        """Initialize StratifiedSampler."""
        self.batch_size = batch_size
        self.num_splits = int(len(labels) / self.batch_size)
        self.labels = labels
        self.num_steps = self.num_splits

    def _sampling(self: Sampler) -> list:
        """Do we want to control randomness here."""
        skf = StratifiedKFold(n_splits=self.num_splits, shuffle=True)
        indices = np.arange(len(self.labels))  # idx holder
        # return array of arrays of indices in each batch
        return [tidx for _, tidx in skf.split(indices, self.labels)]

    def __iter__(self: Sampler) -> Iterator:
        """Define Iterator."""
        return iter(self._sampling())

    def __len__(self: Sampler) -> int:
        """The length of the sampler.

        This value actually corresponds to the number of steps to query
        sampled batch indices. Thus, to maintain epoch and steps hierarchy,
        this should be equal to the number of expected steps as in usual
        sampling: `steps=dataset_size / batch_size`.

        """
        return self.num_steps
