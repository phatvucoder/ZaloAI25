import os
import json
import numpy as np
from ultralytics import YOLO


class Model:
    def __init__(
        self,
        weight_path: str = "./saved_models/yolo11n.pt",
        imgsz: int = 640,
        conf: float = 0.35,
    ):
        print("[INIT] Loading YOLO model...")
        self.model = YOLO(weight_path)
        self.imgsz = imgsz
        self.conf = conf

        # Lưu kết quả per-frame để ghi submission.json
        self.results = {}

        print("[INIT] Done.")

    def predict_streaming(self, frame_rgb_np, frame_idx):
        # YOLO inference
        out = self.model(
            source=frame_rgb_np, imgsz=self.imgsz, conf=self.conf, verbose=False
        )[0]

        dets = out.boxes.xyxy.cpu().numpy()
        confs = out.boxes.conf.cpu().numpy()

        # Không có object nào
        if len(dets) == 0:
            bbox = None
        else:
            # Lấy object có confidence cao nhất
            idx = np.argmax(confs)
            x1, y1, x2, y2 = map(int, dets[idx])
            bbox = [x1, y1, x2, y2]

        # lưu kết quả cho submission.json
        self.results[frame_idx] = bbox

        return bbox

    def write_submission(self, out_path="/result/submission.json"):
        result_dir = os.path.dirname(out_path)
        os.makedirs(result_dir, exist_ok=True)

        output = []
        for fid in sorted(self.results.keys()):
            output.append({"id": fid, "bbox": self.results[fid]})

        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[SAVE] submission.json → {out_path}")
