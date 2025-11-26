from ultralytics import YOLO
import cv2
import numpy as np
import os

# 모델 경로 (AI repo 기준)
MODEL_PATH = os.path.join("models", "ui_detector_sketch2aia.pt")


class UIDetector:
    def __init__(self):
        print("[INFO] Loading YOLO model...")
        self.model = YOLO(MODEL_PATH)

    def run(self, image_bytes):
        """
        image_bytes: 프론트/파일에서 읽어온 raw bytes
        return: [
          { "id": 0, "class": "Button", "bbox": [x, y, w, h] },
          ...
        ]
        """
        # byte → numpy
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # YOLO inference
        results = self.model(img, verbose=False)

        detections = []
        det_id = 0

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                xc, yc, w, h = box.xywh[0].tolist()  # center-based
                label = self.model.names[cls_id]

                # center → top-left 변환
                x1 = xc - w / 2
                y1 = yc - h / 2

                detections.append({
                    "id": det_id,                      # 🔥 고유 ID
                    "class": label,
                    "bbox": [float(x1), float(y1), float(w), float(h)]
                })
                det_id += 1

        return detections