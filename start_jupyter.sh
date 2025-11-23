#!/bin/bash

echo "[INFO] Starting Jupyter Lab on port 9777..."

jupyter lab \
    --port 9777 \
    --ip 0.0.0.0 \
    --allow-root \
    --no-browser \
    --NotebookApp.password='zac2025' \
    --NotebookApp.token='zac2025'
