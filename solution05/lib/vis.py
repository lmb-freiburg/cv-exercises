import functools
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_DEFAULT_FONT_SIZE = 10
_DEFAULT_FONT_PATH = "OpenSans-Regular.ttf"
_DEFAULT_FONTS = {
    _DEFAULT_FONT_SIZE: (
        ImageFont.truetype(_DEFAULT_FONT_PATH, _DEFAULT_FONT_SIZE)
        if os.path.isfile(_DEFAULT_FONT_PATH)
        else None
    )
}
_DEFAULT_BBOX_COLOR = (238, 232, 213)
_DEFAULT_BBOX_STROKE = None
_DEFAULT_TEXT_COLOR = (0, 43, 54)
_DEFAULT_CMAP = "turbo"
_DEFAULT_MARKER_COLOR = (255, 0, 0)


def _get_default_font(size=None):
    if size is None:
        if _DEFAULT_FONT_SIZE not in _DEFAULT_FONTS:
            _DEFAULT_FONTS[_DEFAULT_FONT_SIZE] = (
                ImageFont.truetype(_DEFAULT_FONT_PATH, _DEFAULT_FONT_SIZE)
                if os.path.isfile(_DEFAULT_FONT_PATH)
                else None
            )
        return _DEFAULT_FONTS[_DEFAULT_FONT_SIZE]
    else:
        return (
            ImageFont.truetype(_DEFAULT_FONT_PATH, size)
            if os.path.isfile(_DEFAULT_FONT_PATH)
            else None
        )


def _get_cmap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    return cmap


def _cmap_min_str(cmap_name):
    if cmap_name == "plasma":
        return "blue"
    elif cmap_name == "jet":
        return "blue"
    elif cmap_name == "turbo":
        return "purple"
    elif cmap_name == "gray":
        return "black"
    elif cmap_name == "autumn":
        return "red"
    elif cmap_name == "cool":
        return "blue"
    else:
        """"""


def _cmap_max_str(cmap_name):
    if cmap_name == "plasma":
        return "yellow"
    elif cmap_name == "jet":
        return "red"
    elif cmap_name == "turbo":
        return "red"
    elif cmap_name == "gray":
        return "white"
    elif cmap_name == "autumn":
        return "yellow"
    elif cmap_name == "cool":
        return "pink"
    else:
        """"""


def _get_marker_range_text(markers, marker_cmap):
    marker_range_text = None
    if markers:  # not None and not empty
        marker_cmap = _DEFAULT_CMAP if marker_cmap is None else marker_cmap
        use_marker_range = all(["score" in marker for marker in markers])
        with np.errstate(divide="ignore", invalid="ignore"):
            min_marker_score_text = (
                np.nanmin([marker["score"] for marker in markers])
                if use_marker_range
                else None
            )
            max_marker_score_text = (
                np.nanmax([marker["score"] for marker in markers])
                if use_marker_range
                else None
            )
        is_marker_score_constant = (
            min_marker_score_text == max_marker_score_text if use_marker_range else None
        )

        min_marker_color = _cmap_min_str(marker_cmap)
        max_marker_color = _cmap_max_str(marker_cmap)

        if is_marker_score_constant:
            marker_range_text = f"Markers: Constant: {min_marker_score_text:0.3f}"
        elif use_marker_range:
            marker_range_text = (
                f"Markers: Min ({min_marker_color}): {min_marker_score_text:0.3f} "
                f"Max ({max_marker_color}): {max_marker_score_text:0.3f}"
            )

    return marker_range_text


def _get_draw_text(
    text,
    text_off,
    image_range_text,
    image_range_text_off,
    marker_range_text,
    marker_range_text_off,
):
    draw_text = ""
    if text is not None and not text_off:
        draw_text += text

        if (
            marker_range_text is not None and not marker_range_text_off
        ) or not image_range_text_off:
            draw_text += "\n"

    if marker_range_text is not None and not marker_range_text_off:
        draw_text += marker_range_text

        if not image_range_text_off:
            draw_text += "\n"

    if not image_range_text_off:
        draw_text += image_range_text

    return draw_text


def _to_img(arr, mode):
    if mode == "BGR" and arr.ndim == 3:  # convert('BGR') somehow does not work..
        arr = arr[:, :, ::-1]
        mode = "RGB"

    img = Image.fromarray(arr).convert(mode)

    return img


def _convert_to_out_format(img, out_format):
    if out_format["type"] == "PIL":
        out = img
    elif out_format["type"] == "np":
        out = np.array(
            img, dtype=out_format["dtype"] if "dtype" in out_format else None
        )
        out = (
            np.transpose(out, [2, 0, 1])
            if "channels" in out_format and out_format["channels"] == "CHW"
            else out
        )
    else:
        raise ValueError(f"Unknown out_format type: {out_format['type']}")
    return out


def _apply_out_action(out, out_action, out_format):
    if out_action is None:
        return

    elif isinstance(out_action, dict):
        if out_action["type"] == "save":
            if out_format["type"] == "PIL":
                out.save(out_action["path"])
            elif out_format["type"] == "np":
                np.save(out_action["path"], out)

    elif isinstance(out_action, str):
        if out_action == "show":
            out.show()


def np2d(
    arr,
    colorize=True,
    clipping=False,
    upper_clipping_thresh=None,
    lower_clipping_thresh=None,
    mark_clipping=False,
    clipping_color=None,
    invalid_values=None,
    mark_invalid=False,
    invalid_color=None,
    text=None,
    cmap=_DEFAULT_CMAP,
    markers=None,
    marker_radius=None,
    marker_text_off=False,
    marker_cmap=_DEFAULT_CMAP,
    ignore_marker_scores=False,
    min_marker_score=None,
    max_marker_score=None,
    image_range_text_off=False,
    image_range_colors_off=False,
    marker_range_text_off=False,
    text_off=False,
    out_format=None,
    out_action=None,
):
    """
    Creates a visualization of a 2d numpy array and returns it as a PIL image.

    Args:
        arr: 2D numpy array.
        colorize: If set to true, the values will be visualized by a colormap, otherwise as gray-values.
        clipping: If true, values above a certain threshold will be clipped before the visualization.
        upper_clipping_thresh: Threshold that is used for clipping the values. If set to False,
        the value mean + 2*std_deviation of the array will be used as threshold. The thresholds are also used
        as limits of the color range.
        lower_clipping_thresh: Threshold that is used for clipping the values. If set to False,
        the value mean - 2*std_deviation of the array will be used as threshold. The thresholds are also used
        as limits of the color range.
        mark_clipping: Mark clipped values with specific colors in the visualization.
        clipping_color: Color for marking clipped values.
        invalid_values: list of values that are invalid (e.g. [0]). If no such values exist, just pass None.
        mark_invalid: Mark invalid (NaN/Inf and all values in the invalid_values list) with
        specific colors in the visualization.
        invalid_color: Color for marking invalid values.
        text: Additional text that is printed on the visualization.
        cmap: Colormap to use for the visualization.
        markers: List of markers, where one marker is a dict with keys xy_pos=(x,y)-coordinate (x is dist to left border, y to top),
        desc=description string, colors=dict with keys marker_color, text_color, bbox_color, bbox_stroke, score).
        Everything except for coordinates can be None.
        marker_radius: Radius for markers.
        marker_text_off: If True, no text information is added to the markers.
        marker_cmap: Colormap to use for the visualization of the markers.
        ignore_marker_scores: If True, scores are never used for coloring the markers.
        min_marker_score: Minimum marker score, used for coloring the markers according to their scores.
        max_marker_score: Maximum marker score, used for coloring the markers according to their scores.
        image_range_text_off: If True, no text information about the range of the image values is added.
        image_range_colors_off:
        marker_range_text_off: If True, no text information about the range of the marker scores is added.
        text_off: If True, the provided text is not added to the image.
        out_format: Dict that describes the format of the output. All such dicts must have 'type' and 'mode' key.
        Currently supported are:
        {'type': 'PIL', 'mode': 'RGB' (see PIL docs for supported modes)} (this is the default format),
        {'type': 'np', 'mode': 'RGB' (see PIL docs for supported modes), 'channels': 'CHW' (or 'HWC'), 'dtype': 'uint8'}.
        out_action: Dict that describes an action on the output. All such dicts must have 'type' key.
        Note that some actions require a specific out_format.
        Currently supported are:
        None,
        {'type': 'show'}.
    """
    arr = arr.astype(np.float32, copy=True)
    cmap_name = _DEFAULT_CMAP if cmap is None else cmap
    out_format = {"type": "PIL", "mode": "RGB"} if out_format is None else out_format
    out_format["mode"] = "RGB" if "mode" not in out_format else out_format["mode"]

    # Remove additional dimensions of size 1 if required:
    if arr.ndim > 2:
        channel_dim = [i for i in range(arr.ndim) if np.size(arr, i) == 1]
        # if arr.ndim - len(channel_dim) < 2:
        #     print("WARNING: Channel dimensions were ambiguous!")
        arr = arr.squeeze(axis=tuple(channel_dim[: arr.ndim - 2]))

    # Filter out all values that are somehow invalid and set them to 0:
    (
        arr,
        invalid_mask,
        invalid_values_mask,
        clipping_mask,
        upper_clipping_mask,
        lower_clipping_mask,
        upper_clipping_thresh,
        lower_clipping_thresh,
    ) = invalidate_np_array(
        arr, clipping, upper_clipping_thresh, lower_clipping_thresh, invalid_values
    )

    # Now work only with valid values of the array and make them visualizable (range 0, 256):
    arr_valid_only = np.ma.masked_array(arr, invalid_mask)

    if not clipping:
        min_value = arr_min = float(np.ma.min(arr_valid_only))
        max_value = arr_max = float(np.ma.max(arr_valid_only))
    else:
        min_value = float(lower_clipping_thresh)
        max_value = float(upper_clipping_thresh)
        arr_min = float(np.ma.min(arr_valid_only))
        arr_max = float(np.ma.max(arr_valid_only))

    min_max_diff = max_value - min_value
    is_constant = max_value == min_value

    if is_constant:
        if min_value == 0:  # array is constant 0
            arr_valid_only *= 0
        else:
            arr_valid_only /= min_value
            arr_valid_only *= 255.0
    else:
        arr_valid_only -= min_value
        arr_valid_only /= min_max_diff
        arr_valid_only *= 255.0

    arr = arr.astype(np.uint8)

    # Now make some (r,g,b) values out of the (0, 255) values:
    if colorize:
        cmap = _get_cmap(cmap_name)
        arr = np.uint8(cmap(arr) * 255)[:, :, 0:3]

        if mark_invalid:
            invalid_color = (
                np.array([0, 0, 0]) if invalid_color is None else invalid_color
            )
            arr[invalid_values_mask] = invalid_color

        if clipping:
            if mark_clipping:
                clipping_color = (
                    np.array([255, 255, 255])
                    if clipping_color is None
                    else clipping_color
                )
                arr[clipping_mask] = clipping_color
            else:
                min_color = np.uint8(cmap([0.0]) * 255)[:, 0:3]
                max_color = np.uint8(cmap([1.0]) * 255)[:, 0:3]
                arr[upper_clipping_mask] = max_color
                arr[lower_clipping_mask] = min_color

    else:
        arr = np.stack([arr, arr, arr], axis=-1)

        if mark_invalid:
            invalid_color = (
                np.array([2, 10, 30]) if invalid_color is None else invalid_color
            )
            arr[invalid_values_mask] = invalid_color

        if clipping:
            if mark_clipping:
                clipping_color = (
                    np.array([67, 50, 54]) if clipping_color is None else clipping_color
                )
                arr[clipping_mask] = clipping_color
            else:
                min_color = np.array([0, 0, 0])
                max_color = np.array([255, 255, 255])
                arr[upper_clipping_mask] = max_color
                arr[lower_clipping_mask] = min_color

    img = _to_img(arr=arr, mode=out_format["mode"])

    marker_range_text = _get_marker_range_text(markers=markers, marker_cmap=marker_cmap)
    min_color = "black" if not colorize else _cmap_min_str(cmap_name)
    max_color = "white" if not colorize else _cmap_max_str(cmap_name)
    if image_range_colors_off:
        image_range_text = (
            "Image: Constant: %0.3f" % min_value
            if is_constant
            else "Min: %0.3f Max: %0.3f" % (arr_min, arr_max)
        )
    else:
        image_range_text = (
            "Image: Constant: %0.3f" % min_value
            if is_constant
            else "Min (%s): %0.3f Max (%s): %0.3f"
            % (min_color, arr_min, max_color, arr_max)
        )

    draw_text = _get_draw_text(
        text,
        text_off,
        image_range_text,
        image_range_text_off,
        marker_range_text,
        marker_range_text_off,
    )

    orig_height = img.height
    img = add_text_to_img(img=img, text=draw_text, xy_leftbottom=(5, 5))
    scale = img.height / orig_height

    if markers:  # not None and not empty
        if scale != 1.0:
            markers = [marker.copy() for marker in markers]
            for marker in markers:
                marker["xy_pos"] = (
                    marker["xy_pos"][0] * scale,
                    marker["xy_pos"][1] * scale,
                )
        add_markers_to_img(
            img=img,
            markers=markers,
            marker_radius=marker_radius,
            marker_text_off=marker_text_off,
            cmap=marker_cmap,
            ignore_scores=ignore_marker_scores,
            min_score=min_marker_score,
            max_score=max_marker_score,
        )

    out = _convert_to_out_format(img, out_format)
    _apply_out_action(out=out, out_action=out_action, out_format=out_format)

    return out


def np3d(
    arr,
    channels="RGB",
    text=None,
    gray=False,
    clipping=False,
    upper_clipping_thresh=None,
    lower_clipping_thresh=None,
    mark_clipping=False,
    clipping_color=None,
    invalid_values=None,
    mark_invalid=False,
    invalid_color=None,
    markers=None,
    marker_radius=None,
    marker_text_off=False,
    marker_cmap=_DEFAULT_CMAP,
    ignore_marker_scores=False,
    min_marker_score=None,
    max_marker_score=None,
    image_range_text_off=False,
    marker_range_text_off=False,
    text_off=False,
    out_format=None,
    out_action=None,
):
    """
    Create a visualization of a 3d numpy array and returns it as PIL image.

    Args:
        arr: 3D numpy array.
        channels: Encoding of the channels. Currently supported are 'RGB', 'BGR', 'FLOW'.
        text: Additional text that is printed on the visualization.
        gray: If true, the 3D array is visualizes in grayscale. Ignored for 'FLOW' encoding.
        clipping: If true, values above a certain threshold will be clipped before the visualization.
        upper_clipping_thresh: Threshold that is used for clipping the values. If set to False,
        the value mean + 2*std_deviation of the array will be used as threshold. The thresholds are also used
        as limits of the color range.
        lower_clipping_thresh: Threshold that is used for clipping the values. If set to False,
        the value mean - 2*std_deviation of the array will be used as threshold. The thresholds are also used
        as limits of the color range.
        mark_clipping: Mark clipped values with specific colors in the visualization.
        clipping_color: Color for marking clipped values.
        invalid_values: list of values that are invalid (e.g. [0]). If no such values exist, just pass None.
        mark_invalid: Mark invalid (NaN/Inf and all values in the invalid_values list) and clipped values with
        specific colors in the visualization.
        invalid_color: Color for marking invalid values.
        markers: List of markers, where one marker is a dict with keys xy_pos=(x,y)-coordinate (x is dist to left border, y to top),
        desc=description string, colors=dict with keys marker_color, text_color, bbox_color, bbox_stroke, score).
        Everything except for coordinates can be None.
        marker_radius: Radius for markers.
        marker_text_off: If True, no text information is added to the markers.
        marker_cmap: Colormap to use for the visualization of the marker scores.
        ignore_marker_scores: If True, scores are never used for coloring the markers.
        min_marker_score: Minimum marker score, used for coloring the markers according to their scores.
        max_marker_score: Maximum marker score, used for coloring the markers according to their scores.
        image_range_text_off: If True, no text information about the range of the image values is added.
        marker_range_text_off: If True, no text information about the range of the marker scores is added.
        text_off: If True, the provided text is not added to the image.
        out_format: Dict that describes the format of the output. All such dicts must have 'type' and 'mode' key.
        Currently supported are:
        {'type': 'PIL', 'mode': 'RGB' (see PIL docs for supported modes)} (this is the default format),
        {'type': 'np', 'mode': 'RGB' (see PIL docs for supported modes), 'channels': 'CHW' (or 'HWC'), 'dtype': 'uint8'}.
        out_action: Dict that describes an action on the output. All such dicts must have 'type' key.
        Note that some actions require a specific out_format.
        Currently supported are:
        None,
        {'type': 'show'}.
    """

    # Remove additional dimensions of size 1 if required:
    if arr.ndim > 3:
        channel_dim = [i for i in range(arr.ndim) if np.size(arr, i) == 1]
        # if arr.ndim - len(channel_dim) < 3:
        #     print("WARNING: Channel dimensions were ambiguous!")
        arr = arr.squeeze(axis=tuple(channel_dim[: arr.ndim - 3]))

    channels = channels.upper()

    if channels == "FLOW":
        vis_arr = _visualize_np_flow_array(
            arr=arr,
            text=text,
            mark_invalid=mark_invalid,
            invalid_color=invalid_color,
            markers=markers,
            marker_radius=marker_radius,
            marker_text_off=marker_text_off,
            marker_cmap=marker_cmap,
            ignore_marker_scores=ignore_marker_scores,
            min_marker_score=min_marker_score,
            max_marker_score=max_marker_score,
            image_range_text_off=image_range_text_off,
            marker_range_text_off=marker_range_text_off,
            text_off=text_off,
            out_format=out_format,
            out_action=out_action,
        )

    elif channels == "RGB" or channels == "BGR":
        # find channels dimension of the input array by looking for a dimension with size 3:
        channel_dim = [i for i in range(arr.ndim) if np.size(arr, i) == 3]
        channel_dim = channel_dim[0]
        if channel_dim != 2:
            transpose_order = [i for i in range(3) if i != channel_dim] + [channel_dim]
            arr = np.transpose(arr, transpose_order)

        if channels == "BGR":
            arr = np.flip(arr, 2)

        vis_arr = _visualize_np_rgb_array(
            arr=arr,
            text=text,
            gray=gray,
            clipping=clipping,
            upper_clipping_thresh=upper_clipping_thresh,
            lower_clipping_thresh=lower_clipping_thresh,
            mark_clipping=mark_clipping,
            clipping_color=clipping_color,
            invalid_values=invalid_values,
            mark_invalid=mark_invalid,
            invalid_color=invalid_color,
            markers=markers,
            marker_radius=marker_radius,
            marker_text_off=marker_text_off,
            marker_cmap=marker_cmap,
            ignore_marker_scores=ignore_marker_scores,
            min_marker_score=min_marker_score,
            max_marker_score=max_marker_score,
            image_range_text_off=image_range_text_off,
            marker_range_text_off=marker_range_text_off,
            text_off=text_off,
            out_format=out_format,
            out_action=out_action,
        )
    else:
        raise ValueError(f"Unknown channels: {channels}")
    return vis_arr


def get_cmap_color(
    value, lower_thresh, upper_thresh, cmap=_DEFAULT_CMAP, channels="rgb"
):
    cmap = _DEFAULT_CMAP if cmap is None else cmap
    cmap = _get_cmap(cmap)

    if lower_thresh == upper_thresh:
        lower_thresh = upper_thresh - 1.0

    norm_value = max(
        min(((value - lower_thresh) / (upper_thresh - lower_thresh)), 1.0), 0.0
    )
    if channels == "rgb":
        color = (np.array(cmap(norm_value)) * 255)[0:3].astype(np.uint8)
    elif channels == "rgba":
        color = (np.array(cmap(norm_value)) * 255).astype(np.uint8)
    else:
        raise ValueError(f"Unknown channels: {channels}")
    return color


def add_text_to_img(
    img,
    text,
    xy_lefttop=None,
    xy_leftbottom=None,
    x_abs_shift=None,
    y_abs_shift=None,
    x_rel_shift=None,
    y_rel_shift=None,
    do_resize=True,
    resize_xy=False,
    max_resize_factor=None,
    text_color=None,
    font=None,
    font_size=None,
    bbox_color=_DEFAULT_BBOX_COLOR,
    bbox_stroke=_DEFAULT_BBOX_STROKE,
):
    """
    Add a text, optionally in a bounding box, to a PIL image.

    Upscales the image to fit the text if necessary. Note: in this case, a copy of the image is returned!

    Args:
        img: Image.
        text: Text. Can span multiple lines via '\n'.
        xy_lefttop: (x,y)-coordinate of the top-left-corner of the text. x is distance to left border, y to top.
        Either xy_lefttop or xy_leftbottom must not be None.
        xy_leftbottom: (x,y)-coordinate of the bottom-left-corner of the text. x is distance to left border, y to bottom.
        Either xy_lefttop or xy_leftbottom must not be None.
        x_abs_shift: Absolute offset in x direction to shift the text.
        y_abs_shift: Absolute offset in y direction to shift the text.
        x_rel_shift: Relative offset in x direction to shift the text.
        y_rel_shift: Relative offset in y direction to shift the text.
        do_resize: Specifies whether the image should be resized to fit the text.
        resize_xy: Specifies whether the (x,y)-position should be adjusted to the resize.
        max_resize_factor: Specifies the maximum factor for resizing the image.
        font: Font to be used for the text. If None, the default font will be used.
        font_size: Font size. If font is supplied, this parameter is ignored.
        text_color: (r,g,b) color for the text. If not set, default will be used.
        bbox_color: (r,g,b) color for a bbox to be drawn around the text. If not set, default will be used.
        bbox_stroke: (r,g,b) color for a bbox to be drawn around the text. If not set, default will be used.
    """

    if text == "":
        return img

    text_color = _DEFAULT_TEXT_COLOR if text_color is None else text_color
    font = _get_default_font(size=font_size) if font is None else font
    draw = ImageDraw.Draw(img)
    text_size = draw.multiline_textbbox(xy=[0, 0], text=text, font=font)[
        -2:
    ]  # (width, height)

    # shift xy pos according to xy_abs/rel_shifts:
    x_shift = (x_rel_shift * text_size[0] if x_rel_shift is not None else 0) + (
        x_abs_shift if x_abs_shift is not None else 0
    )
    y_shift = (y_rel_shift * text_size[1] if y_rel_shift is not None else 0) + (
        y_abs_shift if y_abs_shift is not None else 0
    )

    if xy_lefttop is not None:
        xy_lefttop = (xy_lefttop[0] + x_shift, xy_lefttop[1] + y_shift)

    if xy_leftbottom is not None:
        xy_leftbottom = (xy_leftbottom[0] + x_shift, xy_leftbottom[1] + y_shift)

    resized = False
    if do_resize:
        resize_factor = 1.0
        if xy_lefttop is not None:
            while (
                img.width < text_size[0] + xy_lefttop[0]
                or img.height < text_size[1] + xy_lefttop[1]
            ):
                if (
                    max_resize_factor is not None
                    and resize_factor * 2 > max_resize_factor
                ):
                    break

                img = img.resize(
                    size=(img.width * 2, img.height * 2), resample=Image.NEAREST
                )

                xy_lefttop = (
                    (xy_lefttop[0] * 2, xy_lefttop[1] * 2) if resize_xy else xy_lefttop
                )

                resize_factor *= 2
                resized = True
        else:
            while (
                img.width < text_size[0] + xy_leftbottom[0]
                or img.height < text_size[1] + xy_leftbottom[1]
            ):
                if (
                    max_resize_factor is not None
                    and resize_factor * 2 > max_resize_factor
                ):
                    break

                img = img.resize(
                    size=(img.width * 2, img.height * 2), resample=Image.NEAREST
                )

                xy_leftbottom = (
                    (xy_leftbottom[0] * 2, xy_leftbottom[1] * 2)
                    if resize_xy
                    else xy_leftbottom
                )

                resize_factor *= 2
                resized = True

    if xy_lefttop is None:
        xy_lefttop = (xy_leftbottom[0], img.height - xy_leftbottom[1] - text_size[1])

    draw = ImageDraw.Draw(img) if resized else draw

    if bbox_color is not None or bbox_stroke is not None:
        bbox_space = text_size[1] * 0.1
        # bbox = ([(xy_lefttop[0] - bbox_space, xy_lefttop[1] - bbox_space), (text_size[0] + xy_lefttop[0] + bbox_space + 1, text_size[1] + xy_lefttop[1] + bbox_space + 1)])
        # removed bbox space from top because somehow the text size estimates seemed to be slightly off anyways
        bbox = [
            (xy_lefttop[0] - bbox_space, xy_lefttop[1]),
            (
                text_size[0] + xy_lefttop[0] + bbox_space + 1,
                text_size[1] + xy_lefttop[1] + bbox_space + 1,
            ),
        ]
        draw.rectangle(bbox, bbox_color, bbox_stroke)

    draw.multiline_text(xy=xy_lefttop, text=text, fill=text_color, font=font)

    return img


def add_markers_to_img(
    img,
    markers,
    marker_radius=None,
    marker_text_off=False,
    cmap=_DEFAULT_CMAP,
    ignore_scores=False,
    min_score=None,
    max_score=None,
):
    """
    Add markers to an image.

    Draws markers, optionally with a description, onto an image.
    Operates in-place and hence changes the input image.

    Args:
        img: Image.
        markers: List of markers, where one marker is a dict with
        keys 'xy_pos' ((x,y)-coordinate with x as dist to left border, y to top),
        'desc' (description string), 'marker_color', 'text_color', 'bbox_color', 'bbox_stroke',
        'score', 'marker_radius'.
        Everything except for coordinates can be left out.
        marker_radius: Radius for markers.
        marker_text_off: If True, no marker descriptions and/or scores are added to the image.
        cmap: Colormap to use in case of coloring the markers according to their scores.
        ignore_scores: If True, scores are never used for coloring the markers.
        min_score: Minimum marker score, used for coloring the markers according to their scores.
        max_score: Maximum marker score, used for coloring the markers according to their scores.
    """

    marker_radius = (
        min(img.height / 70, img.width / 70) if marker_radius is None else marker_radius
    )
    marker_radius = int(marker_radius)

    if not ignore_scores:
        use_scores = True

        do_update_min_score = min_score is None
        do_update_max_score = max_score is None

        min_score = np.inf if do_update_min_score else min_score
        max_score = -np.inf if do_update_max_score else max_score

        for marker in markers:
            xy_pos = marker["xy_pos"]

            if not (0 <= xy_pos[0] < img.width and 0 <= xy_pos[1] < img.height):
                continue

            if "score" in marker:
                min_score = (
                    min(min_score, marker["score"])
                    if do_update_min_score
                    else min_score
                )
                max_score = (
                    max(max_score, marker["score"])
                    if do_update_max_score
                    else max_score
                )
            else:
                use_scores = False
                break

    else:
        use_scores = False

    draw = ImageDraw.Draw(img)

    for marker in markers:
        xy_pos = marker["xy_pos"]

        if not (0 <= xy_pos[0] < img.width and 0 <= xy_pos[1] < img.height):
            continue

        if use_scores:
            if np.isfinite(marker["score"]):
                marker_color = get_cmap_color(
                    value=marker["score"],
                    lower_thresh=min_score,
                    upper_thresh=max_score,
                    cmap=cmap,
                )
                marker_color = (marker_color[0], marker_color[1], marker_color[2])
            elif "marker_color" not in marker:
                marker_color = (192, 192, 192)
            else:
                marker_color = marker["marker_color"]
        elif "marker_color" in marker:
            marker_color = marker["marker_color"]
        elif "text_color" in marker:
            marker_color = marker["text_color"]
        else:
            marker_color = _DEFAULT_MARKER_COLOR

        text_color = marker["text_color"] if "text_color" in marker else None
        bbox_color = (
            marker["bbox_color"] if "bbox_color" in marker else _DEFAULT_BBOX_COLOR
        )
        bbox_stroke = (
            marker["bbox_stroke"] if "bbox_stroke" in marker else _DEFAULT_BBOX_STROKE
        )

        r = marker_radius if "marker_radius" not in marker else marker["marker_radius"]
        draw.ellipse(
            (
                max(0, xy_pos[0] - r),
                max(0, xy_pos[1] - r),
                xy_pos[0] + r,
                xy_pos[1] + r,
            ),
            fill=marker_color,
        )

        desc = marker["desc"] if "desc" in marker else None
        desc_text_off = marker["desc_text_off"] if "desc_text_off" in marker else False
        score_text_off = (
            marker["score_text_off"] if "score_text_off" in marker else False
        )

        show_desc_text = desc is not None and not desc_text_off
        show_score_text = (
            "score" in marker and np.isfinite(marker["score"]) and not score_text_off
        )

        if not marker_text_off and (show_desc_text or show_score_text):
            text = (
                (desc if show_desc_text else "")
                + ("\n" if (show_desc_text and show_score_text) else "")
                + ("s=" + "%0.2f" % (marker["score"]) if show_score_text else "")
            )
            add_text_to_img(
                img=img,
                text=text,
                xy_leftbottom=(xy_pos[0], img.height - xy_pos[1]),
                x_rel_shift=-0.5,
                y_rel_shift=-1.0,
                y_abs_shift=-r - 2,
                do_resize=False,
                text_color=text_color,
                bbox_color=bbox_color,
                bbox_stroke=bbox_stroke,
            )

    return img


def _make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Output: numpy array of shape (55,3)
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def _visualize_np_flow_array(
    arr,
    text=None,
    mark_invalid=False,
    invalid_color=None,
    markers=None,
    marker_radius=None,
    marker_text_off=False,
    marker_cmap=_DEFAULT_CMAP,
    ignore_marker_scores=False,
    min_marker_score=None,
    max_marker_score=None,
    image_range_text_off=False,
    marker_range_text_off=False,
    text_off=False,
    out_format=None,
    out_action=None,
):
    """
    Creates a visualization of a 3d numpy array of shape (H, W, 2) (or a permutation of it)
    where each pixel (h,w) contains a 2d vector(flow_x, flow_y) that indicates the optical flow.

    Note that flow_x is the 'horizontal flow' and flow_y is the 'vertical flow'.
    """
    arr = arr.astype(np.float64, copy=True)  # implicitely copies the input array
    out_format = {"type": "PIL", "mode": "RGB"} if out_format is None else out_format
    out_format["mode"] = "RGB" if "mode" not in out_format else out_format["mode"]

    # find channels dimension of the input array by looking for a dimension with size 2:
    channel_dim = [i for i in range(arr.ndim) if np.size(arr, i) == 2]
    # if len(channel_dim) > 1:
    #     print("WARNING: Channel dimension was ambiguous!")
    channel_dim = channel_dim[0]
    if channel_dim != 2:
        transpose_order = [i for i in range(3) if i != channel_dim] + [channel_dim]
        arr = np.transpose(arr, transpose_order)

    # Filter out all values that are somehow invalid and set them to 0:
    arr, invalid_mask, invalid_values_mask, _, _, _, _, _ = invalidate_np_array(
        arr
    )  # add clipping here if you want
    invalid_flow_values_mask = np.logical_and(
        invalid_values_mask[:, :, 0], invalid_values_mask[:, :, 1]
    )

    # Get min and max values of the flow in x and y direction:
    flow_arr_valid_only = np.ma.masked_array(arr, invalid_mask)
    flow_x_arr_valid_only = flow_arr_valid_only[:, :, 0]  # horizontal flow
    flow_y_arr_valid_only = flow_arr_valid_only[:, :, 1]  # vertical flow

    flow_x_min = float(np.ma.min(flow_x_arr_valid_only))
    flow_x_max = float(np.ma.max(flow_x_arr_valid_only))

    flow_y_min = float(np.ma.min(flow_y_arr_valid_only))
    flow_y_max = float(np.ma.max(flow_y_arr_valid_only))

    # Convert flow to colors (taken from github.com/tomrunia/OpticalFlow_Visualization ):
    vis_arr = np.zeros((arr.shape[0], arr.shape[1], 3), np.uint8)

    colorwheel = _make_colorwheel()  # shape (55,3)
    colorwheel_cols = colorwheel.shape[0]

    flow_x_arr = arr[:, :, 0]  # horizontal flow
    flow_y_arr = arr[:, :, 1]  # vertical flow

    rad = np.sqrt(np.square(flow_x_arr) + np.square(flow_y_arr))
    rad_max = np.max(rad)

    epsilon = 1e-5
    u = flow_x_arr / (rad_max + epsilon)
    v = flow_y_arr / (rad_max + epsilon)

    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (colorwheel_cols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == colorwheel_cols] = 0
    f = fk - k0

    for color_channel_idx in range(
        colorwheel.shape[1]
    ):  # color_channel_idx in [0,1,2] -> r, g, b
        tmp = colorwheel[:, color_channel_idx]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range?

        vis_arr[:, :, color_channel_idx] = np.floor(255 * col)

    invalid_color = np.array([0, 0, 0]) if invalid_color is None else invalid_color

    if mark_invalid:
        vis_arr[invalid_flow_values_mask] = invalid_color

    # if clipping:
    #     if mark_clipping:
    #         arr[clipping_mask] = clipping_color
    #     else:
    #         arr[upper_clipping_mask] = max_color
    #         arr[lower_clipping_mask] = min_color

    img = _to_img(arr=vis_arr, mode=out_format["mode"])

    marker_range_text = _get_marker_range_text(markers=markers, marker_cmap=marker_cmap)
    image_range_text = "Flow: X-Range: (%0.1f, %0.1f) Y-Range: (%0.1f, %0.1f)" % (
        flow_x_min,
        flow_x_max,
        flow_y_min,
        flow_y_max,
    )
    draw_text = _get_draw_text(
        text,
        text_off,
        image_range_text,
        image_range_text_off,
        marker_range_text,
        marker_range_text_off,
    )

    orig_height = img.height
    img = add_text_to_img(img=img, text=draw_text, xy_leftbottom=(5, 5))
    scale = img.height / orig_height

    if markers:  # not None and not empty
        if scale != 1.0:
            markers = [marker.copy() for marker in markers]
            for marker in markers:
                marker["xy_pos"] = (
                    marker["xy_pos"][0] * scale,
                    marker["xy_pos"][1] * scale,
                )
        add_markers_to_img(
            img=img,
            markers=markers,
            marker_radius=marker_radius,
            marker_text_off=marker_text_off,
            cmap=marker_cmap,
            ignore_scores=ignore_marker_scores,
            min_score=min_marker_score,
            max_score=max_marker_score,
        )

    out = _convert_to_out_format(img, out_format)
    _apply_out_action(out=out, out_action=out_action, out_format=out_format)

    return out


def _visualize_np_rgb_array(
    arr,
    text,
    gray=False,
    clipping=False,
    upper_clipping_thresh=None,
    lower_clipping_thresh=None,
    mark_clipping=False,
    clipping_color=None,
    invalid_values=None,
    mark_invalid=False,
    invalid_color=None,
    markers=None,
    marker_radius=None,
    marker_text_off=False,
    marker_cmap=_DEFAULT_CMAP,
    ignore_marker_scores=False,
    min_marker_score=None,
    max_marker_score=None,
    image_range_text_off=False,
    marker_range_text_off=False,
    text_off=False,
    out_format=None,
    out_action=None,
):
    """
    Visualizes a rgb image, which is encoded as numpy array of shape (H, W, 3).
    """

    arr = arr.astype(np.float32, copy=True)
    marker_cmap = _DEFAULT_CMAP if marker_cmap is None else marker_cmap
    out_format = {"type": "PIL", "mode": "RGB"} if out_format is None else out_format
    out_format["mode"] = "RGB" if "mode" not in out_format else out_format["mode"]

    # Filter out all values that are somehow invalid and set them to 0:
    (
        arr,
        invalid_mask,
        invalid_values_mask,
        clipping_mask,
        upper_clipping_mask,
        lower_clipping_mask,
        upper_clipping_thresh,
        lower_clipping_thresh,
    ) = invalidate_np_array(
        arr, clipping, upper_clipping_thresh, lower_clipping_thresh, invalid_values
    )

    # Now work only with valid values of the array and make them visualizable (range 0, 256):
    arr_valid_only = np.ma.masked_array(arr, invalid_mask)

    if not clipping:
        min_value = float(np.ma.min(arr_valid_only))
        max_value = float(np.ma.max(arr_valid_only))
    else:
        min_value = float(lower_clipping_thresh)
        max_value = float(upper_clipping_thresh)

    min_max_diff = max_value - min_value
    is_constant = max_value == min_value

    if is_constant:
        if min_value == 0:  # array is constant 0
            arr_valid_only *= 0
        else:
            arr_valid_only /= min_value
            arr_valid_only *= 255.0
    else:
        arr_valid_only -= min_value
        arr_valid_only /= min_max_diff
        arr_valid_only *= 255.0

    if gray:
        arr = np.mean(arr_valid_only, axis=2)
        arr[np.any(invalid_mask, axis=2)] = 0
        arr = np.stack([arr, arr, arr], axis=-1)

    arr = arr.astype(np.uint8)

    if mark_invalid:
        invalid_color = np.array([0, 0, 0]) if invalid_color is None else invalid_color
        arr[np.any(invalid_values_mask, axis=2)] = invalid_color

    if clipping:
        if mark_clipping:
            clipping_color = (
                np.array([255, 255, 255]) if clipping_color is None else clipping_color
            )
            arr[np.any(clipping_mask, axis=2)] = clipping_color
        else:
            min_color = np.array([min_value] * 3)
            max_color = np.array([max_value] * 3)
            arr[np.any(upper_clipping_mask, axis=2)] = max_color
            arr[np.any(lower_clipping_mask, axis=2)] = min_color

    img = _to_img(arr=arr, mode=out_format["mode"])

    marker_range_text = _get_marker_range_text(markers=markers, marker_cmap=marker_cmap)
    image_range_text = (
        "Image: Constant: %0.3f" % min_value
        if is_constant
        else "Min: %0.3f Max: %0.3f" % (min_value, max_value)
    )
    draw_text = _get_draw_text(
        text,
        text_off,
        image_range_text,
        image_range_text_off,
        marker_range_text,
        marker_range_text_off,
    )

    orig_height = img.height
    img = add_text_to_img(img=img, text=draw_text, xy_leftbottom=(5, 5))
    scale = img.height / orig_height

    if markers:  # not None and not empty
        if scale != 1.0:
            markers = [marker.copy() for marker in markers]
            for marker in markers:
                marker["xy_pos"] = (
                    marker["xy_pos"][0] * scale,
                    marker["xy_pos"][1] * scale,
                )
        add_markers_to_img(
            img=img,
            markers=markers,
            marker_radius=marker_radius,
            marker_text_off=marker_text_off,
            cmap=marker_cmap,
            ignore_scores=ignore_marker_scores,
            min_score=min_marker_score,
            max_score=max_marker_score,
        )

    out = _convert_to_out_format(img, out_format)
    _apply_out_action(out=out, out_action=out_action, out_format=out_format)

    return out


def invalidate_np_array(
    arr,
    clipping=False,
    upper_clipping_thresh=None,
    lower_clipping_thresh=None,
    invalid_values=None,
):
    """
    Sets non-finite values (inf / nan), values that should be clipped (above / below some threshold), and specific values to 0.

    Can be used with arrays of arbitrary shapes. However, all filtering performs on single values only. So, for filtering
    values across multiple channels you have to split the array and filter each channel separately.
    """
    invalid_values_mask = np.isinf(arr) | np.isnan(arr)
    if invalid_values is not None:
        invalid_values_mask = invalid_values_mask | np.isin(arr, invalid_values)

    if clipping:
        if upper_clipping_thresh is None or lower_clipping_thresh is None:
            mean = np.nanmean(arr[~invalid_values_mask])
            std = np.nanstd(arr[~invalid_values_mask])
            all_values_invalid = np.all(invalid_values_mask)

            if upper_clipping_thresh is None:
                upper_clipping_thresh = (
                    min(np.nanmax(arr[~invalid_values_mask]), mean + 2 * std)
                    if not all_values_invalid
                    else np.nan
                )
            if lower_clipping_thresh is None:
                lower_clipping_thresh = (
                    max(np.nanmin(arr[~invalid_values_mask]), mean - 2 * std)
                    if not all_values_invalid
                    else np.nan
                )

        with np.errstate(invalid="ignore"):
            upper_clipping_mask = np.logical_and(
                (arr > upper_clipping_thresh), ~invalid_values_mask
            )
            lower_clipping_mask = np.logical_and(
                (arr < lower_clipping_thresh), ~invalid_values_mask
            )
        clipping_mask = (
            upper_clipping_mask | lower_clipping_mask
        )  # True = value should be clipped
    else:
        clipping_mask = np.zeros_like(
            arr, dtype="bool"
        )  # All False because no values should be clipped
        upper_clipping_mask = clipping_mask
        lower_clipping_mask = clipping_mask

    invalid_mask = invalid_values_mask | clipping_mask
    arr[invalid_mask] = 0

    return (
        arr,
        invalid_mask,
        invalid_values_mask,
        clipping_mask,
        upper_clipping_mask,
        lower_clipping_mask,
        upper_clipping_thresh,
        lower_clipping_thresh,
    )


# Simple aliases:
np2ds = functools.partial(np2d, out_action="show")
np3ds = functools.partial(np3d, out_action="show")
