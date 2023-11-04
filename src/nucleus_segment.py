




def extract_composition_features(
    wsi_paths: list[str],
    msk_paths: list[str],
    save_dir: str,
    preproc_func: Callable,
) -> list:
    """Extract cellular composition features."""
    inst_segmentor = NucleusInstanceSegmentor(
        pretrained_model="hovernet_fast-pannuke",
        batch_size=16,
        num_postproc_workers=4,
        num_loader_workers=4,
    )
    # bigger tile shape for postprocessing performance
    inst_segmentor.ioconfig.tile_shape = (4000, 4000)
    # Injecting customized preprocessing functions,
    # check the document or sample codes below for API
    inst_segmentor.model.preproc_func = preproc_func

    rmdir(save_dir)
    output_map_list = inst_segmentor.predict(
        wsi_paths,
        msk_paths,
        mode="wsi",
        on_gpu=ON_GPU,
        crash_on_exception=True,
        save_dir=save_dir,
    )
    # Rename output files of toolbox
    output_paths = []
    for input_path, output_path in output_map_list:
        input_name = Path(input_path).stem

        output_parent_dir = Path(output_path).parent

        src_path = Path(f"{output_path}.dat")
        new_path = Path(f"{output_parent_dir}/{input_name}.dat")
        src_path.rename(new_path)
        output_paths.append(new_path)

    # TODO(TBC): Parallelize this if possible  # noqa: TD003, FIX002
    for idx, path in enumerate(output_paths):
        get_cell_compositions(wsi_paths[idx], msk_paths[idx], path, save_dir)
    return output_paths