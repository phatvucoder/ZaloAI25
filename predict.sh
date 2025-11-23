#!/bin/bash

echo "[INFO] Running predict.sh ..."


echo "[INFO] Loading model..."
start_load=$(date +%s%3N)
python3 - << 'EOF'
from predict import Model
model = Model()
EOF

end_load=$(date +%s%3N)
load_time=$((end_load - start_load))
echo "[TIME] Model load time (ms): $load_time"



echo "[INFO] Starting prediction..."
start_pred=$(date +%s%3N)

python3 /code/predict.py  # tự động đọc từ /data và ghi ra /result/submission.json

end_pred=$(date +%s%3N)
predict_time=$((end_pred - start_pred))

echo "[TIME] Total prediction time (ms): $predict_time"
echo "[INFO] Prediction done. Output stored at /result/submission.json"
