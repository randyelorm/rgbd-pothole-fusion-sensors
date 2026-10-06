#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate

LOGDIR="$ROOT/pothole_fusion/results/logs"
mkdir -p "$LOGDIR"

echo "[$(date)] Preparing data..."
python pothole_fusion/scripts/prepare_data.py | tee "$LOGDIR/prepare.log"

for m in rgb early depth; do
  echo "[$(date)] Training $m (100 epochs, MPS)..."
  python pothole_fusion/scripts/train.py --modality "$m" --name "${m}_yolov8n_seg" \
    2>&1 | tee "$LOGDIR/train_${m}.log"
done

RGB_W="$ROOT/pothole_fusion/runs/rgb_yolov8n_seg/weights/best.pt"
DEPTH_W="$ROOT/pothole_fusion/runs/depth_yolov8n_seg/weights/best.pt"
echo "[$(date)] Late fusion eval..."
python pothole_fusion/scripts/late_fusion_eval.py \
  --rgb-weights "$RGB_W" \
  --depth-weights "$DEPTH_W" \
  2>&1 | tee "$LOGDIR/late_fusion.log"

echo "[$(date)] Done."
