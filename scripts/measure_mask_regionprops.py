from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table


PROPERTIES = ["label", "area"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure label-mask areas and plot area histograms as a standalone utility.")
    parser.add_argument("mask_tif", help="Label mask TIFF. Background should be 0; objects should be integer labels.")
    parser.add_argument("-o", "--output-csv", default=None)
    parser.add_argument("--histogram-png", default=None)
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument(
        "--z-mode",
        choices=["auto", "all", "per-slice", "max-area-slice", "volume"],
        default="auto",
        help="How to summarize Z-stack masks. auto/all output all 3D modes.",
    )
    args = parser.parse_args()

    mask_path = Path(args.mask_tif)
    output_csv = Path(args.output_csv) if args.output_csv else mask_path.with_name(f"{mask_path.stem}_regionprops.csv")
    histogram_png = (
        Path(args.histogram_png)
        if args.histogram_png
        else mask_path.with_name(f"{mask_path.stem}_area_histogram.png")
    )

    labels = tifffile.imread(mask_path)
    table = _measure(labels, args.z_mode)
    table.to_csv(output_csv, index=False)
    _plot_histogram(table, histogram_png, args.bins)
    print(output_csv)
    print(histogram_png)
    return 0


def _measure(labels, z_mode: str) -> pd.DataFrame:
    if labels.ndim == 4:
        rows = []
        for frame, frame_labels in enumerate(labels):
            table = _measure_one(frame_labels, z_mode)
            table.insert(0, "frame", frame)
            rows.append(table)
        return pd.concat(rows, ignore_index=True) if rows else _empty_result()

    table = _measure_one(labels, z_mode)
    table.insert(0, "frame", 0)
    return table


def _measure_one(labels, z_mode: str) -> pd.DataFrame:
    if labels.max() == 0:
        return _empty_result(drop_frame=True)
    if labels.ndim == 3:
        mode = "all" if z_mode == "auto" else z_mode
        if mode == "all":
            return pd.concat(
                [
                    _measure_max_area_slice(labels),
                    _measure_volume(labels),
                    _measure_per_slice(labels),
                ],
                ignore_index=True,
            )
        if mode == "per-slice":
            return _measure_per_slice(labels)
        if mode == "max-area-slice":
            return _measure_max_area_slice(labels)
        if mode == "volume":
            return _measure_volume(labels)
        raise ValueError(f"Unsupported z-mode: {z_mode}")
    table = pd.DataFrame(regionprops_table(labels, properties=PROPERTIES))
    table["measurement_mode"] = "2d"
    return table


def _measure_per_slice(labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for z_index, plane in enumerate(labels):
        if plane.max() == 0:
            continue
        table = pd.DataFrame(regionprops_table(plane, properties=PROPERTIES))
        table.insert(0, "z", z_index)
        table["measurement_mode"] = "per-slice"
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else _empty_result(drop_frame=True)


def _measure_max_area_slice(labels: np.ndarray) -> pd.DataFrame:
    rows = []
    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids > 0]
    for label_id in label_ids:
        object_mask = labels == label_id
        area_by_z = object_mask.reshape(object_mask.shape[0], -1).sum(axis=1)
        z_index = int(np.argmax(area_by_z))
        plane = np.zeros_like(labels[z_index], dtype=np.int32)
        plane[object_mask[z_index]] = int(label_id)
        table = pd.DataFrame(regionprops_table(plane, properties=PROPERTIES))
        if table.empty:
            continue
        table.insert(0, "z", z_index)
        table["z_area_max"] = int(area_by_z[z_index])
        table["z_span_slices"] = int(np.count_nonzero(area_by_z))
        table["volume_voxels"] = int(area_by_z.sum())
        table["measurement_mode"] = "max-area-slice"
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else _empty_result(drop_frame=True)


def _measure_volume(labels: np.ndarray) -> pd.DataFrame:
    table = pd.DataFrame(regionprops_table(labels, properties=PROPERTIES))
    table["volume_voxels"] = table["area"]
    table["measurement_mode"] = "volume"
    return table


def _plot_histogram(table: pd.DataFrame, output_path: Path, bins: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if "area" not in table or table.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_title("No masks measured")
        ax.set_xlabel("area")
        ax.set_ylabel("mask count")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return

    modes = list(table["measurement_mode"].dropna().unique()) if "measurement_mode" in table else ["all"]
    fig, axes = plt.subplots(len(modes), 1, figsize=(7, max(3, 3 * len(modes))), squeeze=False)
    for ax, mode in zip(axes[:, 0], modes):
        subset = table if mode == "all" else table[table["measurement_mode"] == mode]
        values = subset["area"].dropna()
        if values.empty:
            ax.set_title(f"{mode}: no masks measured")
        else:
            ax.hist(values, bins=bins, color="#4c78a8", edgecolor="white", linewidth=0.6)
            ax.set_title(f"{mode} area distribution")
        ax.set_xlabel("area")
        ax.set_ylabel("mask count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _empty_result(drop_frame: bool = False) -> pd.DataFrame:
    columns = PROPERTIES if drop_frame else ["frame", *PROPERTIES]
    return pd.DataFrame(columns=columns)


if __name__ == "__main__":
    raise SystemExit(main())
