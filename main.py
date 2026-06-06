import glob
import os

import cv2

from detect_plate import PlateDetector
from model_pred import YoloCharPredictor, format_char_boxes
from ROI import ROI


INPUT_PATH = "img"
DEBUG_DIR = "debug_plate"
CHAR_DIR = "debug_chars"
YOLO_OUTPUT_DIR = "debug_yolo"
USE_MODEL = True
MODEL_PATH = "best.pt"
CONF = 0.25
IMGSZ = 640
IOU = 0.45
SHOW_IMAGE = False

ANS_PATH = "411285041.txt"


def get_image_paths(input_path):
    if os.path.isfile(input_path):
        return [input_path]

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(glob.glob(os.path.join(input_path, ext)))

    return sorted(image_paths)


def process_image(
    image_path,
    detector,
    roi_processor,
    yolo_predictor,
    debug_dir,
    char_dir,
    yolo_output_dir,
    use_model=True,
    show_image=False,
):
    filename = os.path.splitext(os.path.basename(image_path))[0]
    input_img = cv2.imread(image_path)
    if input_img is None:
        print(f"{filename}: cannot read image")
        return

    plate_roi, plate_box, candidates = detector.detect(
        image_path,
        debug=True,
        save_dir=debug_dir,
    )

    if plate_roi is None:
        print(f"{filename}: no plate ROI found, use original img")
        plate_roi = input_img
        plate_box = (0, 0, input_img.shape[1], input_img.shape[0])

    if use_model:
        detections = yolo_predictor.predict(
            plate_roi,
            offset=(plate_box[0], plate_box[1]),
        )
        yolo_predictor.save_debug(
            input_img,
            detections,
            yolo_output_dir,
            filename,
        )
        char_count = len(detections)
        char_boxes_original = format_char_boxes(detections)
        method = "model"
    else:
        chars, char_boxes_roi, _ = roi_processor.segment(
            plate_roi,
            debug=True,
            save_dir=char_dir,
            basename=filename,
        )
        char_boxes_original = roi_processor.boxes_to_original(char_boxes_roi, plate_box)
        roi_processor.save_original_debug(
            input_img,
            char_boxes_original,
            char_dir,
            filename,
        )
        char_count = len(chars)
        method = "ROI"

    if candidates:
        best = candidates[0]
        print(
            f"{filename}: box={plate_box}, candidates={len(candidates)}, "
            f"score={best['score']:.2f}, source={best['source']}, method={method}, chars={char_count}, "
            f"char_boxes={char_boxes_original}"
        )
    else:
        print(
            f"{filename}: box={plate_box}, candidates=0, method={method}, chars={char_count}, "
            f"char_boxes={char_boxes_original}"
        )
    with open(ANS_PATH, 'a') as f:
        f.write(f"{filename}\n")
        f.write(f"{char_count}\n")
        for box in char_boxes_original:
            f.write(f"{list(box)}\n")

    if show_image:
        cv2.imshow("plate roi", plate_roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    input_path = INPUT_PATH
    debug_dir = DEBUG_DIR
    char_dir = CHAR_DIR
    yolo_output_dir = YOLO_OUTPUT_DIR
    use_model = USE_MODEL
    show_image = SHOW_IMAGE
    
    with open(ANS_PATH, 'w') as f :
        pass

    image_paths = get_image_paths(input_path)
    if not image_paths:
        print(f"No JPG images found in {input_path}")
        return

    detector = PlateDetector()
    roi_processor = ROI()
    yolo_predictor = None
    if use_model:
        yolo_predictor = YoloCharPredictor(
            model_path=MODEL_PATH,
            conf=CONF,
            imgsz=IMGSZ,
            iou=IOU,
        )

    for image_path in image_paths:
        process_image(
            image_path,
            detector,
            roi_processor,
            yolo_predictor,
            debug_dir,
            char_dir,
            yolo_output_dir,
            use_model=use_model,
            show_image=show_image,
        )


if __name__ == "__main__":
    main()
