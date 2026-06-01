from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tifffile
import torch
from cellpose import models
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
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--do-3d", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--z-mode", choices=["auto", "all", "per-slice", "max-area-slice", "volume"], default="all")
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    for directory in (mask_dir, table_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    model = models.CellposeModel(gpu=device.type == "cuda", device=device)
    print(f"Cellpose device: {device}", flush=True)

    rows = []
    for path in _iter_inputs(input_path, args.pattern):
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
        _plot_histogram(table, hist_path, args.bins)

        rows.append(
            {
                "file": path.name,
                "axes": axes,
                "prepared_axes": prepared_axes,
                "original_shape": "x".join(str(value) for value in image.shape),
                "prepared_shape": "x".join(str(value) for value in prepared.shape),
                "seg_channel": args.seg_channel,
                "do_3d": do_3d,
                "labels": int(masks.max()) if masks.size else 0,
                "mask_tif": str(mask_path),
                "area_csv": str(csv_path),
                "area_histogram": str(hist_path),
            }
        )
        print(f"OK {path.name}: labels={rows[-1]['labels']}", flush=True)

    pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False)
    print(output_dir / "summary.csv")
    return 0


def _iter_inputs(input_path: Path, pattern: str):
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.glob(pattern))


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
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(requested)


if __name__ == "__main__":
    raise SystemExit(main())
