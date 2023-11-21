
from __future__ import annotations

import os
import json
import shutil
import pathlib
from tqdm import tqdm


def load_json(path: str) -> dict | list | int | float | str:
    """Load JSON from a file path."""
    with path.open() as fptr:
        return json.load(fptr)


def rmdir(dir_path: str) -> None:
    """Remove a directory."""
    if dir_path.is_dir():
        shutil.rmtree(dir_path)


def rm_n_mkdir(dir_path: str) -> None:
    """Remove then re-create a directory."""
    if dir_path.is_dir():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True)


def mkdir(dir_path: str) -> None:
    """Create a directory if it does not exist."""
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)


def recur_find_ext(root_dir: pathlib.Path, exts: list[str]) -> list[str]:
    """Recursively find files with an extension in `exts`.

    This is much faster than glob if the folder
    hierachy is complicated and contain > 1000 files.

    Args:
        root_dir (pathlib.Path):
            Root directory for searching.
        exts (list):
            List of extensions to match.

    Returns:
        List of full paths with matched extension in sorted order.

    """
    assert isinstance(exts, list)  # noqa: S101
    file_path_list = []
    for cur_path, _dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            file_ext = pathlib.Path(file_name).suffix
            if file_ext in exts:
                full_path = pathlib.Path(cur_path) / file_name
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list


def create_pbar(subset_name: str, num_steps: int) -> tqdm:
    """Create a nice progress bar."""
    pbar_format = (
        "Processing: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
    )
    pbar = tqdm(total=num_steps, leave=True, bar_format=pbar_format, ascii=True)
    if subset_name == "train":
        pbar_format += "step={postfix[1][step]:0.5f}|EMA={postfix[1][EMA]:0.5f}"
        # * Changing print char may break the bar so avoid it
        pbar = tqdm(
            total=num_steps,
            leave=True,
            initial=0,
            bar_format=pbar_format,
            ascii=True,
            postfix=["", {"step": float("NaN"), "EMA": float("NaN")}],
        )
    return pbar


class ScalarMovingAverage:
    """Class to calculate running average."""

    def __init__(self: ScalarMovingAverage, alpha: float = 0.95) -> None:
        """Initialize ScalarMovingAverage."""
        super().__init__()
        self.alpha = alpha
        self.tracking_dict = {}

    def __call__(self: ScalarMovingAverage, step_output: dict) -> None:
        """ScalarMovingAverage instances behave and can be called like a function."""
        for key, current_value in step_output.items():
            if key in self.tracking_dict:
                old_ema_value = self.tracking_dict[key]
                # Calculate the exponential moving average
                new_ema_value = (
                    old_ema_value * self.alpha + (1.0 - self.alpha) * current_value
                )
                self.tracking_dict[key] = new_ema_value
            else:  # Init for variable which appear for the first time
                new_ema_value = current_value
                self.tracking_dict[key] = new_ema_value

