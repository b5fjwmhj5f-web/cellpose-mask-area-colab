# Cellpose Mask Area Colab

Small Colab-oriented utility for:

1. Running Cellpose on TIFF images.
2. Saving label mask TIFFs.
3. Measuring mask area.
4. Exporting area CSV files and area histograms.

## Colab Setup

```python
!git clone https://github.com/b5fjwmhj5f-web/cellpose-mask-area-colab.git
%cd cellpose-mask-area-colab
!pip install -r requirements.txt
```

If Colab warns that `numpy` was already imported, restart the runtime once.

## Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Example: 3-channel TIFF, C3 DAPI Segmentation

This uses zero-based channel indexing:

- C1 = `--seg-channel 0`
- C2 = `--seg-channel 1`
- C3 = `--seg-channel 2`

```python
!python scripts/run_cellpose_area_pipeline.py \
  "/content/drive/MyDrive/YOUR_FOLDER/YOUR_IMAGE.tif" \
  --output-dir "/content/drive/MyDrive/cellpose_area_outputs" \
  --seg-channel 2 \
  --max-yx 0 \
  --device cuda \
  --z-mode all
```

Outputs:

```text
cellpose_area_outputs/
├── masks/
├── tables/
├── figures/
└── summary.csv
```

## Z-stack Area Modes

- `max-area-slice`: one row per 3D mask, using the largest cross-sectional area.
- `volume`: one row per 3D mask, using total mask voxel count.
- `per-slice`: one row per Z-slice cross-section.
- `all`: outputs all three modes.

