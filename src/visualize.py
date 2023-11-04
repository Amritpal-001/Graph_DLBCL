import colorsys
import random
from typing import Dict, List, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import numpy as np
import cv2

warnings.filterwarnings("ignore")
mpl.rcParams["figure.dpi"] = 300  # for high resolution figure in notebook
mpl.rcParams["figure.facecolor"] = "white"  # To make sure text is visible in dark mode
plt.rcParams.update({"font.size": 5})


def random_colors(num_colors, bright=True):
    """Generate a number of random colors.

    To get visually distinct colors, generate them in HSV space then
    convert to RGB.

    Args:
        num_colors(int):
            Number of perceptively different colors to generate.
        bright(bool):
            To use bright color or not.

    Returns:
        list:
            List of (r, g, b) colors.

    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / num_colors, 1, brightness) for i in range(num_colors)]
    colors = [colorsys.hsv_to_rgb(*c) for c in hsv]
    random.shuffle(colors)
    return colors

def overlay_prediction_contours(
    canvas: np.ndarray,
    inst_dict: dict,
    draw_dot: bool = False,
    type_colours: dict = None,
    inst_colours: Union[np.ndarray, Tuple[int]] = (255, 255, 0),
    line_thickness: int = 2,
):
    """Overlaying instance contours on image.

    Internally, colours from `type_colours` are prioritized over
    `inst_colours`. However, if `inst_colours` is `None` and
    `type_colours` is not provided, random colour is generated for each
    instance.

    Args:
        canvas (:class:`numpy.ndarray`):
            Image to draw predictions on.
        inst_dict (dict):
            Dictionary of instances. It is expected to be in the
            following format: `{instance_id: {type: int, contour:
            List[List[int]], centroid:List[float]}`.
        draw_dot (bool):
            To draw a dot for each centroid or not.
        type_colours (dict):
            A dict of {type_id : (type_name, colour)}, `type_id` is from
            0 to N and `colour` is a tuple of `(r, g, b)`.
        inst_colours (tuple, np.ndarray):
            A colour to assign for all instances, or a list of colours
            to assigned for each instance in `inst_dict`. By default,
            all instances will have RGB colour `(255, 255, 0)`.
        line_thickness:
            Line thickness of contours.

    Returns:
        :class:`numpy.ndarray`:
            The overlaid image.

    """

    overlay_mask = np.zeros(canvas.shape)

    overlay_image = np.copy(canvas)

    if inst_colours is None:
        inst_colours = random_colors(len(inst_dict))
        inst_colours = np.array(inst_colours) * 255
        inst_colours = inst_colours.astype(np.uint8)
    elif isinstance(inst_colours, tuple):
        inst_colours = np.array([inst_colours] * len(inst_dict))
    elif not isinstance(inst_colours, np.ndarray):
        raise ValueError(
            f"`inst_colours` must be np.ndarray or tuple: {type(inst_colours)}"
        )
    inst_colours = inst_colours.astype(np.uint8)
    # print(inst_colours)

    for idx, [_, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        if "type" in inst_info and type_colours is not None:
            inst_colour = type_colours[inst_info["type"]][1]
        else:
            inst_colour = (inst_colours[idx]).tolist()
        cv2.drawContours(overlay_mask, [np.array(inst_contour)], -1, inst_colour, line_thickness)
        cv2.fillPoly(overlay_image, [np.array(inst_contour)], inst_colour)

        if draw_dot:
            inst_centroid = inst_info["centroid"]
            inst_centroid = tuple(int(v) for v in inst_centroid)
            overlay_mask = cv2.circle(overlay_mask, inst_centroid, 3, (255, 0, 0), -1)
    return overlay_mask, overlay_image
