# RGB-D Pothole Fusion (YOLOv8n-seg) — Sensors paper artifacts

Public code and reproducibility artifacts for:

**Does Depth Improve Pothole Instance Segmentation? A Controlled Comparison of Early and Late RGB-D Fusion in YOLOv8n-seg**

Authors: Randy Davoh & Prince Boakye Sekyerehene (University of Ghana)

## What is in this repository

| Path | Contents |
|------|----------|
| `scripts/` | Training, data prep, late-fusion evaluation |
| `configs/` | Default training config |
| `models/` | Stem adaptation helpers (`copy_rgb`, `average_rgb`) |
| `splits/` | Train / val / test image **IDs** (seed 42; 794 / 99 / 100) |
| `results/metrics/` | Test-set metric JSON (RGB, early, depth, late), GFLOPs, stem L1 |
| `results/curves/` | Ultralytics `results.csv` for RGB / early / depth (100 epochs) |
| `dataset_configs/` | YOLO `data.yaml` stubs (paths are local; update after download) |

**Not included:** PothRGBD RGB/depth images or trained `.pt` weights (large; dataset is third-party).

## Dataset (obtain separately)

PothRGBD by Yurdakul & Taşdemir:

- IEEE DataPort: https://doi.org/10.21227/z8eq-sf60  
- Upstream GitHub: https://github.com/ymyurdakul/PData  

Place paired RGB and depth files locally, then align folders to the IDs in `splits/*.txt` (same stem for RGB and depth).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# After dataset is arranged locally:
python scripts/prepare_data.py
python scripts/train.py --modality rgb
python scripts/train.py --modality early
python scripts/train.py --modality depth
# Late fusion (paths to your best.pt weights):
python scripts/late_fusion_eval.py --rgb-weights ... --depth-weights ...
```

## License

Code and split lists in this repository: MIT (see `LICENSE`).  
PothRGBD images remain under the original dataset license / terms.

## Citation

Davoh, R., & Sekyerehene, P. B. (2026). Does Depth Improve Pothole Instance Segmentation? A Controlled Comparison of Early and Late RGB-D Fusion in YOLOv8n-seg. Zenodo. https://doi.org/10.5281/zenodo.23202453

Also cite the PothRGBD dataset (Yurdakul & Taşdemir) via the DOI/links in the Dataset section above.
