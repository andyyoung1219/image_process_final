import glob
import os

import cv2


class YoloPlatePredictor:
    def __init__(
        self,
        model_path="best_lin.pt",
        conf=0.8,
        imgsz=640,
        iou=0.45,
        device=0,
        agnostic_nms=True,
    ):
        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz
        self.iou = iou
        self.device = device
        self.agnostic_nms = agnostic_nms
        self.model = self._load_model(model_path)

    def predict(self, image, conf=None, imgsz=None, iou=None, device=None):
        """Predict license plate boxes from a BGR image.

        Args:
            image: OpenCV BGR image.
            conf: optional confidence threshold.
            imgsz: optional YOLO input image size.
            iou: optional NMS IoU threshold.
            device: optional inference device. Use None to let ultralytics decide.

        Returns:
            A list of dictionaries sorted by confidence from high to low.
            Each item contains index, label, confidence, bbox=(x, y, w, h),
            xyxy=(x1, y1, x2, y2), area, and center=(cx, cy).
        """

        if image is None:
            raise ValueError("image cannot be None")

        predict_kwargs = {
            "source": image,
            "conf": self.conf if conf is None else conf,
            "imgsz": self.imgsz if imgsz is None else imgsz,
            "iou": self.iou if iou is None else iou,
            "agnostic_nms": self.agnostic_nms,
            "verbose": False,
        }

        infer_device = self.device if device is None else device
        if infer_device is not None:
            predict_kwargs["device"] = infer_device

        results = self.model.predict(**predict_kwargs)
        if not results:
            return []

        image_h, image_w = image.shape[:2]
        result = results[0]
        names = result.names
        detections = []

        for box in result.boxes:
            x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = self._clip_xyxy((x1, y1, x2, y2), image_w, image_h)
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue

            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = str(names.get(class_id, class_id)) if isinstance(names, dict) else str(class_id)

            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox": (x1, y1, w, h),
                    "xyxy": (x1, y1, x2, y2),
                    "area": w * h,
                    "center": (x1 + w * 0.5, y1 + h * 0.5),
                }
            )

        detections = sorted(detections, key=lambda item: item["confidence"], reverse=True)
        for index, detection in enumerate(detections, start=1):
            detection["index"] = index

        return detections

    def detect(
        self,
        image,
        conf=None,
        imgsz=None,
        iou=None,
        device=None,
        pad_ratio=(0.07, 0.18),
    ):
        """Detect the best license plate ROI from a BGR image.

        Returns:
            plate_roi: cropped BGR image, or None when no plate is found.
            plate_box: (x, y, w, h) in the original image, including padding.
            detections: all YOLO plate detections sorted by confidence.
        """

        detections = self.predict(
            image,
            conf=conf,
            imgsz=imgsz,
            iou=iou,
            device=device,
        )
        if not detections:
            return None, None, []

        plate_box = self.padded_box(detections[0]["bbox"], image.shape, pad_ratio=pad_ratio)
        x, y, w, h = plate_box
        plate_roi = image[y : y + h, x : x + w]

        return plate_roi, plate_box, detections

    def predict_file(self, image_path, conf=None, imgsz=None, iou=None, device=None):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        detections = self.predict(
            image,
            conf=conf,
            imgsz=imgsz,
            iou=iou,
            device=device,
        )
        return detections, image

    def detect_file(
        self,
        image_path,
        conf=None,
        imgsz=None,
        iou=None,
        device=None,
        pad_ratio=(0.07, 0.18),
    ):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        plate_roi, plate_box, detections = self.detect(
            image,
            conf=conf,
            imgsz=imgsz,
            iou=iou,
            device=device,
            pad_ratio=pad_ratio,
        )
        return plate_roi, plate_box, detections, image

    def save_debug(self, image, detections, save_dir, basename):
        os.makedirs(save_dir, exist_ok=True)

        debug_img = image.copy()
        for detection in detections:
            index = detection["index"]
            label = detection["label"]
            confidence = detection["confidence"]
            x, y, w, h = detection["bbox"]

            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(
                debug_img,
                f"{index}:{label} {confidence:.2f}",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imwrite(os.path.join(save_dir, f"{basename}_plate_pred.jpg"), debug_img)

    def padded_box(self, bbox, image_shape, pad_ratio=(0.07, 0.18)):
        image_h, image_w = image_shape[:2]
        x, y, w, h = bbox
        pad_x = int(w * pad_ratio[0])
        pad_y = int(h * pad_ratio[1])

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image_w, x + w + pad_x)
        y2 = min(image_h, y + h + pad_y)

        return (x1, y1, x2 - x1, y2 - y1)

    def _load_model(self, model_path):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics is required. Install it with: uv sync") from exc

        return YOLO(model_path)

    @staticmethod
    def _clip_xyxy(xyxy, image_w, image_h):
        x1, y1, x2, y2 = xyxy

        x1 = max(0, min(image_w, x1))
        y1 = max(0, min(image_h, y1))
        x2 = max(0, min(image_w, x2))
        y2 = max(0, min(image_h, y2))

        return x1, y1, x2, y2


def get_image_paths(input_path):
    if os.path.isfile(input_path):
        return [input_path]

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(input_path, ext)))

    return sorted(image_paths)


def format_plate_boxes(detections):
    return [detection["bbox"] for detection in detections]


INPUT_PATH = "img"
DEBUG_DIR = "debug_plate_yolo"
MODEL_PATH = "best_lin.pt"
CONF = 0.25
IMGSZ = 640
IOU = 0.45
DEVICE = 0


def main():
    image_paths = get_image_paths(INPUT_PATH)
    if not image_paths:
        print(f"No images found in {INPUT_PATH}")
        return

    predictor = YoloPlatePredictor(
        model_path=MODEL_PATH,
        conf=CONF,
        imgsz=IMGSZ,
        iou=IOU,
        device=DEVICE,
    )

    os.makedirs(DEBUG_DIR, exist_ok=True)

    for image_path in image_paths:
        basename = os.path.splitext(os.path.basename(image_path))[0]
        plate_roi, plate_box, detections, image = predictor.detect_file(image_path)

        predictor.save_debug(image, detections, DEBUG_DIR, basename)

        if plate_roi is None:
            print(f"{basename}: no plate detected")
            continue

        roi_path = os.path.join(DEBUG_DIR, f"{basename}_plate_roi.jpg")
        cv2.imwrite(roi_path, plate_roi)

        best = detections[0]
        print(
            f"{basename}: box={plate_box}, confidence={best['confidence']:.4f}, "
            f"detections={len(detections)}, roi={roi_path}"
        )


if __name__ == "__main__":
    main()
