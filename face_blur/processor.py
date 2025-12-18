import cv2
import torch
from ultralytics import YOLO


class FaceBlurYOLO:
    """
    Face blurring using YOLOv8 face detection.
    """

    def __init__(
        self,
        model_path: str,
        analyze_width: int = 640,
        detect_every_n: int = 1,
        blur_kernel: tuple = (51, 51),
        conf: float = 0.05,
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        print(f"[INFO] Using device: {self.device}")

        self.model = YOLO(model_path).to(self.device)

        self.analyze_width = analyze_width
        self.detect_every_n = detect_every_n
        self.blur_kernel = blur_kernel
        self.conf = conf

        self.frame_id = 0
        self.last_boxes: list[tuple[int, int, int, int]] = []

    def _downscale(self, frame):
        h, w = frame.shape[:2]
        scale = self.analyze_width / w
        resized = cv2.resize(frame, (self.analyze_width, int(h * scale)))
        return resized, scale

    def _detect_faces(self, frame_small):
        result = self.model(
            frame_small,
            conf=self.conf,
            iou=0.5,
            verbose=False,
        )[0]

        boxes = []
        if result.boxes is not None:
            for x1, y1, x2, y2 in result.boxes.xyxy.cpu().numpy():
                boxes.append(
                    (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                )
        return boxes

    def _scale_boxes(self, boxes, scale):
        return [
            (
                int(x / scale),
                int(y / scale),
                int(w / scale),
                int(h / scale),
            )
            for x, y, w, h in boxes
        ]

    def _blur_faces(self, frame, boxes):
        for x, y, w, h in boxes:
            roi = frame[y:y + h, x:x + w]
            if roi.size:
                frame[y:y + h, x:x + w] = cv2.GaussianBlur(
                    roi, self.blur_kernel, 0
                )
        return frame

    def process_frame(self, frame):
        self.frame_id += 1

        if self.frame_id % self.detect_every_n == 0 or not self.last_boxes:
            small, scale = self._downscale(frame)
            boxes_small = self._detect_faces(small)
            self.last_boxes = self._scale_boxes(boxes_small, scale)

        return self._blur_faces(frame, self.last_boxes)
