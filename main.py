import argparse
import glob
import os

import cv2
import numpy as np


def _score_similarity(value, target, tolerance):
    return max(0.0, 1.0 - abs(value - target) / tolerance)


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


def _collect_candidates(mask, source_name, min_area=250):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area:
            continue

        candidates.append(
            {
                "box": (x, y, w, h),
                "contour_area": cv2.contourArea(contour),
                "source": source_name,
            }
        )

    return candidates


def _dedupe_candidates(candidates):
    candidates = sorted(
        candidates,
        key=lambda item: item["box"][2] * item["box"][3],
        reverse=True,
    )

    deduped = []
    for candidate in candidates:
        if all(_iou(candidate["box"], old["box"]) < 0.55 for old in deduped):
            deduped.append(candidate)

    return deduped


def _crop_plate_surface(plate_roi):
    height, width = plate_roi.shape[:2]
    hsv = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2HSV)
    plate_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 65]),
        np.array([180, 155, 255]),
    )
    plate_mask = cv2.morphologyEx(
        plate_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)),
        iterations=2,
    )

    contours, _ = cv2.findContours(plate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_box = None
    best_score = 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect = w / float(h + 1e-6)
        if area < width * height * 0.12:
            continue
        if not (1.4 <= aspect <= 7.0):
            continue

        score = area * _score_similarity(aspect, 3.2, 2.6)
        if score > best_score:
            best_score = score
            best_box = (x, y, w, h)

    if best_box is None:
        return plate_roi, (0, 0, width, height)

    x, y, w, h = best_box
    pad_x = max(1, int(w * 0.02))
    pad_y = max(1, int(h * 0.04))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)

    return plate_roi[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def _split_wide_character_box(binary, box):
    x, y, w, h = box
    if w / float(h + 1e-6) < 0.82:
        return [box]

    roi = binary[y : y + h, x : x + w]
    projection = np.sum(roi > 0, axis=0)
    threshold = max(1, int(h * 0.08))

    gaps = []
    start = None
    for idx, value in enumerate(projection):
        if value <= threshold and start is None:
            start = idx
        elif value > threshold and start is not None:
            if idx - start >= max(2, int(w * 0.05)):
                gaps.append((start, idx))
            start = None
    if start is not None and w - start >= max(2, int(w * 0.05)):
        gaps.append((start, w))

    center_gaps = [
        (start, end)
        for start, end in gaps
        if w * 0.20 <= (start + end) * 0.5 <= w * 0.80
    ]
    if not center_gaps:
        return [box]

    split_at = max(center_gaps, key=lambda gap: gap[1] - gap[0])
    mid = (split_at[0] + split_at[1]) // 2
    left_w = mid
    right_w = w - mid
    if left_w < h * 0.12 or right_w < h * 0.12:
        return [box]

    return [(x, y, left_w, h), (x + mid, y, right_w, h)]


def _filter_character_artifacts(boxes, image_w, image_h):
    if not boxes:
        return []

    median_w = float(np.median([box[2] for box in boxes]))
    median_h = float(np.median([box[3] for box in boxes]))

    filtered = []
    for x, y, w, h in boxes:
        near_edge = x < image_w * 0.06 or x + w > image_w * 0.94

        if near_edge and h > median_h * 1.22 and w < median_w * 0.80:
            continue
        if near_edge and w > median_w * 1.55 and h > median_h * 1.15:
            continue
        if w > median_w * 1.90 and h > median_h * 1.15:
            continue
        if h > median_h * 1.45 and w < median_w:
            continue

        filtered.append((x, y, w, h))

    return filtered


def _merge_character_fragments(boxes):
    if len(boxes) < 2:
        return boxes

    boxes = sorted(boxes, key=lambda item: item[0])
    median_w = float(np.median([box[2] for box in boxes]))
    merged = []
    idx = 0

    while idx < len(boxes):
        x, y, w, h = boxes[idx]

        if idx + 1 < len(boxes):
            nx, ny, nw, nh = boxes[idx + 1]
            gap = nx - (x + w)
            merged_x1 = min(x, nx)
            merged_y1 = min(y, ny)
            merged_x2 = max(x + w, nx + nw)
            merged_y2 = max(y + h, ny + nh)
            merged_w = merged_x2 - merged_x1

            if (
                w < median_w * 0.72
                and gap <= median_w * 0.10
                and nw <= median_w * 1.30
                and merged_w <= median_w * 1.70
            ):
                merged.append(
                    (
                        merged_x1,
                        merged_y1,
                        merged_w,
                        merged_y2 - merged_y1,
                    )
                )
                idx += 2
                continue

        merged.append((x, y, w, h))
        idx += 1

    return merged


def _boxes_to_original(char_boxes, plate_box):
    plate_x, plate_y, _, _ = plate_box
    return [
        (plate_x + x, plate_y + y, w, h)
        for x, y, w, h in char_boxes
    ]


def _save_original_char_debug(image, char_boxes, save_dir, basename):
    output_dir = os.path.join(save_dir, basename)
    os.makedirs(output_dir, exist_ok=True)

    debug_img = image.copy()
    csv_path = os.path.join(output_dir, "char_boxes.csv")

    with open(csv_path, "w", encoding="utf-8") as file:
        file.write("index,x,y,w,h\n")
        for idx, (x, y, w, h) in enumerate(char_boxes, start=1):
            file.write(f"{idx},{x},{y},{w},{h}\n")
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(
                debug_img,
                str(idx),
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.imwrite(os.path.join(output_dir, "original_char_boxes.jpg"), debug_img)


def segment_plate_characters(plate_roi, debug=False, save_dir="debug_chars", basename="plate"):
    """Split a cropped license plate ROI into character images."""

    plate_surface, surface_box = _crop_plate_surface(plate_roi)
    height, width = plate_surface.shape[:2]
    if height == 0 or width == 0:
        return [], [], None

    scale = 120 / float(height)
    resized = cv2.resize(
        plate_surface,
        (max(1, int(width * scale)), 120),
        interpolation=cv2.INTER_CUBIC,
    )
    resized_h, resized_w = resized.shape[:2]

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 35, 35)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        11,
    )
    binary = cv2.bitwise_or(otsu, adaptive)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1,
    )

    search_x = int(resized_w * 0.04)
    search_y = int(resized_h * 0.18)
    search_w = max(1, int(resized_w * 0.92))
    search_h = max(1, int(resized_h * 0.68))
    search_binary = binary[search_y : search_y + search_h, search_x : search_x + search_w]

    contours, _ = cv2.findContours(search_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x += search_x
        y += search_y
        area = w * h
        aspect = w / float(h + 1e-6)
        center_y = (y + h * 0.5) / resized_h
        fill_ratio = cv2.contourArea(contour) / float(area + 1e-6)

        if not (resized_h * 0.28 <= h <= resized_h * 0.74):
            continue
        if not (resized_w * 0.012 <= w <= resized_w * 0.18):
            continue
        if not (0.06 <= aspect <= 1.10):
            continue
        if not (0.34 <= center_y <= 0.76):
            continue
        if fill_ratio < 0.12:
            continue

        boxes.append((x, y, w, h))

    split_boxes = []
    for box in sorted(boxes, key=lambda item: item[0]):
        split_boxes.extend(_split_wide_character_box(binary, box))

    boxes = _filter_character_artifacts(split_boxes, resized_w, resized_h)
    boxes = _merge_character_fragments(boxes)
    boxes = sorted(boxes, key=lambda item: item[0])

    if len(boxes) > 8:
        median_h = float(np.median([box[3] for box in boxes]))
        boxes = [
            box
            for box in boxes
            if box[3] >= median_h * 0.72 and box[2] * box[3] >= resized_h * resized_w * 0.006
        ]
        boxes = sorted(boxes, key=lambda item: item[0])[:8]

    surface_x, surface_y, _, _ = surface_box
    char_images = []
    resized_boxes = []
    roi_boxes = []
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        pad_x = max(2, int(w * 0.12))
        pad_y = max(2, int(h * 0.08))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(resized_w, x + w + pad_x)
        y2 = min(resized_h, y + h + pad_y)

        char = binary[y1:y2, x1:x2]
        char = cv2.copyMakeBorder(char, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=0)
        char = cv2.resize(char, (40, 80), interpolation=cv2.INTER_AREA)
        char_images.append(char)
        resized_boxes.append((x1, y1, x2 - x1, y2 - y1))

        roi_x1 = surface_x + int(round(x1 / scale))
        roi_y1 = surface_y + int(round(y1 / scale))
        roi_x2 = surface_x + int(round(x2 / scale))
        roi_y2 = surface_y + int(round(y2 / scale))
        roi_boxes.append((roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1))

    if debug:
        output_dir = os.path.join(save_dir, basename)
        os.makedirs(output_dir, exist_ok=True)

        debug_img = resized.copy()
        for idx, (x, y, w, h) in enumerate(resized_boxes, start=1):
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(
                debug_img,
                str(idx),
                (x, max(15, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imwrite(os.path.join(output_dir, "plate_surface.jpg"), plate_surface)
        cv2.imwrite(os.path.join(output_dir, "binary.jpg"), binary)
        cv2.imwrite(os.path.join(output_dir, "segmented.jpg"), debug_img)
        with open(os.path.join(output_dir, "char_boxes_roi.csv"), "w", encoding="utf-8") as file:
            file.write("index,x,y,w,h\n")
            for idx, (x, y, w, h) in enumerate(roi_boxes, start=1):
                file.write(f"{idx},{x},{y},{w},{h}\n")
        for idx, char in enumerate(char_images, start=1):
            cv2.imwrite(os.path.join(output_dir, f"char_{idx:02d}.jpg"), char)

    return char_images, roi_boxes, surface_box


def detect_plate_roi(image_path, debug=False, save_dir="debug_roi"):
    """Detect a license plate ROI from one image.

    Returns:
        plate_roi: cropped BGR image, or None when no candidate is found.
        plate_box: (x, y, w, h) in the original image.
        candidates: scored candidate boxes, sorted from best to worst.
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    height, width = img.shape[:2]
    image_area = width * height

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 85]),
        np.array([180, 115, 255]),
    )

    sobel_x = cv2.Sobel(gray_eq, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x = np.absolute(sobel_x)
    sobel_x = np.uint8(255 * sobel_x / (sobel_x.max() + 1e-6))
    _, edge_mask = cv2.threshold(
        sobel_x,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    combined = cv2.bitwise_and(white_mask, edge_mask)

    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5))
    edge_morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, edge_kernel, iterations=2)
    edge_morph = cv2.morphologyEx(
        edge_morph,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    plate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 7))
    white_morph = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, plate_kernel, iterations=2)
    white_morph = cv2.morphologyEx(
        white_morph,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        iterations=1,
    )

    raw_candidates = []
    raw_candidates.extend(_collect_candidates(edge_morph, "edge"))
    raw_candidates.extend(_collect_candidates(white_morph, "white"))

    candidates = []
    for candidate in _dedupe_candidates(raw_candidates):
        x, y, w, h = candidate["box"]
        area = w * h
        aspect = w / float(h + 1e-6)
        area_ratio = area / image_area
        center_y = (y + h * 0.5) / height

        if not (1.60 <= aspect <= 5.8):
            continue
        if not (width * 0.04 <= w <= width * 0.45):
            continue
        if not (height * 0.035 <= h <= height * 0.22):
            continue
        if not (0.001 <= area_ratio <= 0.06):
            continue
        if y < height * 0.04:
            continue

        roi_gray = gray[y : y + h, x : x + w]
        roi_white = white_mask[y : y + h, x : x + w]
        roi_edges = edge_mask[y : y + h, x : x + w]

        white_ratio = float(np.mean(roi_white > 0))
        dark_ratio = float(np.mean(roi_gray < 105))
        edge_density = float(np.mean(roi_edges > 0))
        fill_ratio = candidate["contour_area"] / float(area + 1e-6)

        if white_ratio < 0.14:
            continue
        if not (0.04 <= dark_ratio <= 0.70):
            continue
        if edge_density < 0.035:
            continue

        aspect_score = _score_similarity(aspect, 2.35, 1.85)
        white_score = _score_similarity(white_ratio, 0.62, 0.45)
        dark_score = _score_similarity(dark_ratio, 0.30, 0.34)
        edge_score = _score_similarity(edge_density, 0.24, 0.22)
        fill_score = _score_similarity(fill_ratio, 0.80, 0.40)
        size_score = _score_similarity(area_ratio, 0.018, 0.050)

        road_penalty = 0.40 if center_y > 0.72 else 1.0
        tiny_penalty = 0.45 if area_ratio < 0.006 else 1.0

        score = (
            (2.4 * white_score)
            + (2.6 * dark_score)
            + (3.0 * aspect_score)
            + (2.4 * edge_score)
            + (2.0 * fill_score)
            + (1.8 * size_score)
        ) * road_penalty * tiny_penalty

        candidates.append(
            {
                **candidate,
                "area": area,
                "aspect": aspect,
                "area_ratio": area_ratio,
                "white_ratio": white_ratio,
                "dark_ratio": dark_ratio,
                "edge_density": edge_density,
                "fill_ratio": fill_ratio,
                "score": score,
            }
        )

    if not candidates:
        return None, None, []

    candidates = sorted(candidates, key=lambda item: item["score"], reverse=True)
    x, y, w, h = candidates[0]["box"]

    pad_x = int(w * 0.07)
    pad_y = int(h * 0.18)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)

    plate_roi = img[y1:y2, x1:x2]
    plate_box = (x1, y1, x2 - x1, y2 - y1)

    if debug:
        os.makedirs(save_dir, exist_ok=True)

        debug_img = img.copy()
        for candidate in candidates:
            bx, by, bw, bh = candidate["box"]
            cv2.rectangle(debug_img, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)

        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

        basename = os.path.splitext(os.path.basename(image_path))[0]
        cv2.imwrite(os.path.join(save_dir, f"{basename}_white_mask.jpg"), white_mask)
        cv2.imwrite(os.path.join(save_dir, f"{basename}_edge_mask.jpg"), edge_mask)
        cv2.imwrite(os.path.join(save_dir, f"{basename}_combined.jpg"), combined)
        cv2.imwrite(os.path.join(save_dir, f"{basename}_morph.jpg"), edge_morph)
        cv2.imwrite(os.path.join(save_dir, f"{basename}_white_morph.jpg"), white_morph)
        cv2.imwrite(os.path.join(save_dir, f"{basename}_detected.jpg"), debug_img)
        cv2.imwrite(os.path.join(save_dir, f"{basename}_roi.jpg"), plate_roi)

    return plate_roi, plate_box, candidates


def main():
    parser = argparse.ArgumentParser(description="Detect license plate ROIs and split characters")
    parser.add_argument("--input", default="img", help="Input image folder")
    parser.add_argument("--debug-dir", default="debug_roi", help="Folder for debug images")
    parser.add_argument("--char-dir", default="debug_chars", help="Folder for character crops")
    parser.add_argument(
        "--skip-detect",
        action="store_true",
        help="Treat input images as already-cropped plate ROIs",
    )
    parser.add_argument("--show", action="store_true", help="Show each detected ROI")
    args = parser.parse_args()

    if os.path.isfile(args.input):
        image_paths = [args.input]
    elif args.skip_detect:
        roi_paths = sorted(glob.glob(os.path.join(args.input, "*_roi.jpg")))
        image_paths = roi_paths or sorted(glob.glob(os.path.join(args.input, "*.jpg")))
    else:
        image_paths = sorted(glob.glob(os.path.join(args.input, "*.jpg")))
    if not image_paths:
        print(f"No JPG images found in {args.input}")
        return

    for image_path in image_paths:
        filename = os.path.splitext(os.path.basename(image_path))[0]
        input_img = cv2.imread(image_path)
        if input_img is None:
            print(f"{filename}: cannot read image")
            continue

        if args.skip_detect or filename.endswith("_roi"):
            roi = input_img
            box = (0, 0, input_img.shape[1], input_img.shape[0])
            candidates = []
        else:
            roi, box, candidates = detect_plate_roi(
                image_path,
                debug=True,
                save_dir=args.debug_dir,
            )

        if roi is None:
            print(f"{filename}: no plate ROI found")
            continue

        chars, char_boxes_roi, _ = segment_plate_characters(
            roi,
            debug=True,
            save_dir=args.char_dir,
            basename=filename,
        )
        char_boxes_original = _boxes_to_original(char_boxes_roi, box)
        _save_original_char_debug(input_img, char_boxes_original, args.char_dir, filename)

        if candidates:
            best = candidates[0]
            print(
                f"{filename}: box={box}, candidates={len(candidates)}, "
                f"score={best['score']:.2f}, source={best['source']}, chars={len(chars)}, "
                f"char_boxes={char_boxes_original}"
            )
        else:
            print(
                f"{filename}: treated as ROI, chars={len(chars)}, "
                f"char_boxes={char_boxes_original}"
            )

        if args.show:
            cv2.imshow("plate roi", roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
