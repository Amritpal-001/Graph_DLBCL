def load_json(path: Path) -> dict | list | int | float | str:
    """Load JSON from a file path."""
    with path.open() as fptr:
        return json.load(fptr)


def rmdir(dir_path: Path) -> None:
    """Remove a directory."""
    if dir_path.is_dir():
        shutil.rmtree(dir_path)


def rm_n_mkdir(dir_path: Path) -> None:
    """Remove then re-create a directory."""
    if dir_path.is_dir():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True)


def mkdir(dir_path: Path) -> None:
    """Create a directory if it does not exist."""
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True)


def recur_find_ext(root_dir: Path, exts: list[str]) -> list[str]:
    """Recursively find files with an extension in `exts`.

    This is much faster than glob if the folder
    hierachy is complicated and contain > 1000 files.

    Args:
        root_dir (Path):
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
            file_ext = Path(file_name).suffix
            if file_ext in exts:
                full_path = cur_path / file_name
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list