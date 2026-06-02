from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import torch
from cellpose import models
from matplotlib.colors import to_rgb
from skimage.measure import regionprops
from skimage.transform import resize

from measure_mask_regionprops import _measure, _plot_histogram


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Cellpose mask generation, then export mask-area CSV and area histogram."
    )
    parser.add_argument("input", help="Input TIFF file or directory.")
    parser.add_argument("--pattern", default="*.tif", help="Used only when input is a directory.")
    parser.add_argument("--output-dir", default="outputs/cellpose_area")
    parser.add_argument(
        "--axes",
        default=None,
        help="Optional TIFF axes override, e.g. YX, CYX, ZYX, CZYX, ZCYX, TCZYX.",
    )
    parser.add_argument(
        "--seg-channel",
        type=int,
        default=None,
        help="0-based channel for segmentation. Example: C3 DAPI is --seg-channel 2.",
    )
    parser.add_argument("--max-yx", type=int, default=0, help="Resize longest Y/X side to this size. 0 keeps original pixels.")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--do-3d", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--pretrained-model", default="cpsam")
    parser.add_argument("--model-type", default=None)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--z-mode", choices=["auto", "all", "per-slice", "max-area-slice", "volume"], default="all")
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--hist-y-scale", choices=["linear", "log"], default="log")
    parser.add_argument("--hist-min-area", type=float, default=0.0)
    parser.add_argument(
        "--label-color-mode",
        choices=["label", "area-threshold"],
        default="label",
        help="Color mode for figures/*_mask_labels.png.",
    )
    parser.add_argument(
        "--label-area-threshold",
        type=float,
        default=None,
        help="Area threshold for --label-color-mode area-threshold.",
    )
    parser.add_argument("--label-color-above", default="red")
    parser.add_argument("--label-color-below", default="yellow")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    for directory in (mask_dir, table_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    model = models.CellposeModel(
        gpu=device.type == "cuda",
        device=device,
        pretrained_model=args.pretrained_model,
        model_type=args.model_type,
    )
    print(f"Cellpose device: {device}", flush=True)
    print(f"Cellpose pretrained_model: {args.pretrained_model}", flush=True)
    print(f"Cellpose model_type: {args.model_type}", flush=True)

    input_files = _iter_inputs(input_path, args.pattern)
    print(f"Input files: {len(input_files)}", flush=True)

    rows = []
    for path in input_files:
        axes, image = _read_tiff(path, args.axes)
        prepared, prepared_axes = _prepare_for_cellpose(image, axes, args.seg_channel)
        prepared = _resize_yx(prepared, args.max_yx)
        do_3d = _resolve_do_3d(args.do_3d, prepared_axes)

        masks, flows, styles = model.eval(
            prepared,
            channel_axis=None,
            z_axis=_axis_index(prepared_axes, "Z"),
            do_3D=do_3d,
            diameter=args.diameter,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            min_size=args.min_size,
        )

        mask_path = mask_dir / f"{path.stem}_cellpose_masks.tif"
        tifffile.imwrite(mask_path, masks.astype("uint32"), imagej=False)

        table = _measure(masks, args.z_mode)
        csv_path = table_dir / f"{path.stem}_mask_area.csv"
        hist_path = figure_dir / f"{path.stem}_area_histogram.png"
        table.to_csv(csv_path, index=False)
        _plot_histogram(table, hist_path, args.bins, args.hist_y_scale, args.hist_min_area)
        label_guide_path = figure_dir / f"{path.stem}_mask_labels.png"
        _save_label_guide(
            masks,
            label_guide_path,
            args.label_color_mode,
            args.label_area_threshold,
            args.label_color_above,
            args.label_color_below,
        )

        rows.append(
            {
                "file": path.name,
                "axes": axes,
                "prepared_axes": prepared_axes,
                "original_shape": "x".join(str(value) for value in image.shape),
                "prepared_shape": "x".join(str(value) for value in prepared.shape),
                "seg_channel": args.seg_channel,
                "do_3d": do_3d,
                "pretrained_model": args.pretrained_model,
                "model_type": args.model_type,
                "labels": int(masks.max()) if masks.size else 0,
                "mask_tif": str(mask_path),
                "area_csv": str(csv_path),
                "area_histogram": str(hist_path),
                "label_guide": str(label_guide_path),
            }
        )
        print(f"OK {path.name}: labels={rows[-1]['labels']}", flush=True)

    pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False)
    print(output_dir / "summary.csv")
    return 0


def _iter_inputs(input_path: Path, pattern: str):
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"Input is neither a file nor a directory: {input_path}")

    files = sorted(input_path.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matched pattern {pattern!r} in {input_path}. "
            "If your file extension is .tiff, use --pattern '*.tiff'. "
            "If you want one image, pass the full .tif/.tiff file path."
        )
    return files


def _read_tiff(path: Path, axes_override: str | None):
    with tifffile.TiffFile(path) as tif:
        axes = axes_override or tif.series[0].axes
        image = tif.asarray()
    if len(axes) != image.ndim:
        raise ValueError(f"Axes length does not match image ndim: axes={axes}, shape={image.shape}")
    return axes, image


def _prepare_for_cellpose(image, axes: str, seg_channel: int | None):
    channel_axis = _channel_axis(axes)
    if channel_axis is None:
        return image, axes
    if seg_channel is None:
        raise ValueError(
            f"Input has channel axis {axes}. Specify --seg-channel. "
            "Example: C3 DAPI is --seg-channel 2."
        )
    image = image.take(indices=seg_channel, axis=channel_axis)
    axes = axes[:channel_axis] + axes[channel_axis + 1 :]
    return image, axes


def _resize_yx(image, max_yx: int):
    if max_yx <= 0:
        return image
    y_size, x_size = image.shape[-2], image.shape[-1]
    if max(y_size, x_size) <= max_yx:
        return image
    scale = max_yx / max(y_size, x_size)
    target_shape = list(image.shape)
    target_shape[-2] = max(1, int(round(y_size * scale)))
    target_shape[-1] = max(1, int(round(x_size * scale)))
    resized = resize(image, target_shape, preserve_range=True, anti_aliasing=True)
    return resized.astype(image.dtype, copy=False)


def _resolve_do_3d(requested: str, axes: str) -> bool:
    if requested == "true":
        return True
    if requested == "false":
        return False
    return "Z" in axes


def _channel_axis(axes: str):
    for name in ("C", "S"):
        if name in axes:
            return axes.index(name)
    return None


def _axis_index(axes: str, name: str):
    return axes.index(name) if name in axes else None


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
    return torch.device(requested)


def _save_label_guide(
    labels,
    output_path: Path,
    color_mode: str = "label",
    area_threshold: float | None = None,
    color_above: str = "red",
    color_below: str = "yellow",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if labels.ndim == 2:
        display = labels
        text_items = [
            (prop.label, prop.centroid[1], prop.centroid[0], prop.area)
            for prop in regionprops(labels)
        ]
        title = "Mask labels"
    elif labels.ndim == 3:
        display, text_items = _project_labels_with_centroids(labels)
        title = "Mask labels, max-area Z per object"
    else:
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    if color_mode == "area-threshold":
        if area_threshold is None:
            raise ValueError("--label-area-threshold is required when --label-color-mode area-threshold is used.")
        ax.imshow(
            _area_threshold_rgb(display, text_items, area_threshold, color_above, color_below),
            interpolation="nearest",
        )
        title = f"{title}, area >= {area_threshold:g}: {color_above}, area < {area_threshold:g}: {color_below}"
    else:
        masked = np.ma.masked_where(display == 0, display)
        ax.imshow(masked, cmap="nipy_spectral", interpolation="nearest")
    ax.set_title(title)
    ax.set_axis_off()
    for label, x_pos, y_pos, area in text_items:
        ax.text(
            x_pos,
            y_pos,
            str(label),
            color="white",
            fontsize=6,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _project_labels_with_centroids(labels):
    display = np.zeros(labels.shape[-2:], dtype=labels.dtype)
    text_items = []
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        object_mask = labels == label_id
        area_by_z = object_mask.reshape(object_mask.shape[0], -1).sum(axis=1)
        z_index = int(np.argmax(area_by_z))
        plane = object_mask[z_index]
        display[plane] = label_id
        coords = np.argwhere(plane)
        if coords.size:
            y_pos, x_pos = coords.mean(axis=0)
            text_items.append((int(label_id), float(x_pos), float(y_pos), float(area_by_z[z_index])))
    return display, text_items


def _area_threshold_rgb(display, text_items, area_threshold: float, color_above: str, color_below: str):
    rgb = np.ones((*display.shape, 3), dtype=float)
    above = np.array(to_rgb(color_above), dtype=float)
    below = np.array(to_rgb(color_below), dtype=float)
    areas_by_label = {label: area for label, x_pos, y_pos, area in text_items}
    for label, area in areas_by_label.items():
        color = above if area >= area_threshold else below
        rgb[display == label] = color
    return rgb


if __name__ == "__main__":
    raise SystemExit(main())
