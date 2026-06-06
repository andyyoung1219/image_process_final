import glob
import os

import cv2


class YoloCharPredictor:
    def __init__(self, model_path="best.pt", conf=0.25, imgsz=640, iou=0.45):
        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz
        self.iou = iou
        self.model = self._load_model(model_path)

    def predict(self, image, offset=(0, 0), conf=None, imgsz=None, iou=None):
        """Predict character boxes from a BGR image.

        Args:
            image: OpenCV BGR image.
            offset: (x, y) added to every bbox, useful when image is a plate ROI
                cropped from the original image.
            conf: optional confidence threshold.
            imgsz: optional YOLO input image size.

        Returns:
            A list of dictionaries sorted left-to-right. Each item contains:
            index, label, confidence, bbox=(x, y, w, h), xyxy=(x1, y1, x2, y2).
        """

        if image is None:
            raise ValueError("image cannot be None")

        results = self.model.predict(
            source=image,
            conf=self.conf if conf is None else conf,
            imgsz=self.imgsz if imgsz is None else imgsz,
            iou=self.iou if iou is None else iou,
            agnostic_nms=True,
            verbose=False,
            device=0,
        )
        
        if not results:
            return []

        ox, oy = offset
        image_h, image_w = image.shape[:2]
        detections = []
        result = results[0]
        names = result.names

        for box in result.boxes:
            x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
            local_bbox = (x1, y1, x2 - x1, y2 - y1)
            if not self._is_valid_char_box(local_bbox, image_w, image_h):
                continue

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = str(names.get(class_id, class_id)) if isinstance(names, dict) else str(class_id)

            x1 += ox
            x2 += ox
            y1 += oy
            y2 += oy

            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                    "xyxy": (x1, y1, x2, y2),
                }
            )

        detections = self._dedupe_detections(detections)
        detections = sorted(detections, key=lambda item: (item["bbox"][0], item["bbox"][1]))
        for index, detection in enumerate(detections, start=1):
            detection["index"] = index

        return detections

    def predict_file(self, image_path, offset=(0, 0), conf=None, imgsz=None, iou=None):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        return self.predict(image, offset=offset, conf=conf, imgsz=imgsz, iou=iou), image

    def _dedupe_detections(self, detections, overlap_threshold=0.80):
        deduped = []
        for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
            if all(self._iou(detection["bbox"], old["bbox"]) < overlap_threshold for old in deduped):
                deduped.append(detection)

        return deduped

    def save_debug(self, image, detections, save_dir, basename):
        os.makedirs(save_dir, exist_ok=True)

        debug_img = image.copy()
        csv_path = os.path.join(save_dir, f"{basename}_char_boxes.csv")

        with open(csv_path, "w", encoding="utf-8") as file:
            file.write("index,label,confidence,x,y,w,h\n")
            for detection in detections:
                index = detection["index"]
                label = detection["label"]
                confidence = detection["confidence"]
                x, y, w, h = detection["bbox"]

                file.write(f"{index},{label},{confidence:.6f},{x},{y},{w},{h}\n")
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(
                    debug_img,
                    f"{index}:{label}",
                    (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imwrite(os.path.join(save_dir, f"{basename}_pred.jpg"), debug_img)

    def _load_model(self, model_path):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required. Install it with: uv sync"
            ) from exc

        return YOLO(model_path)

    @staticmethod
    def _is_valid_char_box(bbox, image_w, image_h):
        _, _, w, h = bbox

        if w < image_w * 0.025:
            return False
        if h < image_h * 0.22:
            return False
        if w > image_w * 0.35:
            return False
        if h > image_h * 0.85:
            return False

        return True

    @staticmethod
    def _iou(box_a, box_b):
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b

        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = aw * ah + bw * bh - inter
        return inter / union if union else 0.0


def get_image_paths(input_path):
    if os.path.isfile(input_path):
        return [input_path]

    return sorted(glob.glob(os.path.join(input_path, "*.jpg")))


def format_char_boxes(detections):
    return [detection["bbox"] for detection in detections]

