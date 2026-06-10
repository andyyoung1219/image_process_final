import glob
import os

import cv2

from detect_plate import PlateDetector
from model_pred import YoloCharPredictor, format_char_boxes
from model_pred_plate import YoloPlatePredictor
from ROI import ROI


INPUT_PATH = "img"
DEBUG_DIR = "debug_plate"
CHAR_DIR = "debug_chars"
YOLO_OUTPUT_DIR = "debug_yolo"
USE_MODEL_PLATE = True
USE_MODEL = True
DEBUG = False
MODEL_PATH = "best.pt"
MODEL_PLATE_PATH = "best_lin.pt"
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
    use_model_plate=True,
    use_model=True,
    debug=False,
    show_image=False,
):
    filename = os.path.splitext(os.path.basename(image_path))[0]
    input_img = cv2.imread(image_path)
    if input_img is None:
        print(f"{filename}: cannot read image")
        return []

    if use_model_plate:
        plate_roi, plate_box, candidates = detector.detect(input_img)
        if debug:
            detector.save_debug(input_img, candidates, debug_dir, filename)
    else:
        plate_roi, plate_box, candidates = detector.detect(
            image_path,
            debug=debug,
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
        if debug:
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
            debug=debug,
            save_dir=char_dir,
            basename=filename,
        )
        char_boxes_original = roi_processor.boxes_to_original(char_boxes_roi, plate_box)
        if debug:
            roi_processor.save_original_debug(
                input_img,
                char_boxes_original,
                char_dir,
                filename,
            )
        char_count = len(chars)
        method = "ROI"

    if debug:
        if candidates:
            best = candidates[0]
            if use_model_plate:
                print(
                    f"{filename}: box={plate_box}, candidates={len(candidates)}, "
                    f"confidence={best['confidence']:.4f}, plate_method=model, method={method}, "
                    f"chars={char_count}, char_boxes={char_boxes_original}"
                )
            else:
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
    if debug and show_image:
        cv2.imshow("plate roi", plate_roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    output_lines = [filename, str(char_count)]
    output_lines.extend(" ".join(str(value) for value in box) for box in char_boxes_original)
    return output_lines


def main():
    input_path = INPUT_PATH
    debug_dir = DEBUG_DIR
    char_dir = CHAR_DIR
    yolo_output_dir = YOLO_OUTPUT_DIR
    use_model_plate = USE_MODEL_PLATE
    use_model = USE_MODEL
    debug = DEBUG
    show_image = SHOW_IMAGE

    image_paths = get_image_paths(input_path)
    if not image_paths:
        print(f"No JPG images found in {input_path}")
        return

    if use_model_plate:
        detector = YoloPlatePredictor(
            model_path=MODEL_PLATE_PATH,
            conf=CONF,
            imgsz=IMGSZ,
            iou=IOU,
        )
    else:
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

    output_lines = []
    for image_path in image_paths:
        output_lines.extend(process_image(
            image_path,
            detector,
            roi_processor,
            yolo_predictor,
            debug_dir,
            char_dir,
            yolo_output_dir,
            use_model_plate=use_model_plate,
            use_model=use_model,
            debug=debug,
            show_image=show_image,
        ))

    with open(ANS_PATH, "w", encoding="utf-8") as file:
        if output_lines:
            file.write("\n".join(output_lines) + "\n")


if __name__ == "__main__":
    main()
