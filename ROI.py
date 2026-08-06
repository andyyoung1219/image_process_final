import os

import cv2
import numpy as np


class ROI:
    def segment(self, plate_roi, debug=False, save_dir="debug_chars", basename="plate"):
        """Split a cropped license plate ROI into character images.

        Returns:
            char_images: normalized 40x80 binary character images.
            roi_boxes: character boxes in the input ROI coordinate system.
            surface_box: cropped plate surface box in the input ROI coordinate system.
        """

        plate_surface, surface_box = self._crop_plate_surface(plate_roi)
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

        white_binary = self._build_white_plate_binary(gray)
        green_binary = self._build_green_plate_binary(gray)
        white_boxes = self._find_character_boxes(white_binary, resized_w, resized_h)
        green_boxes = self._find_character_boxes(green_binary, resized_w, resized_h, is_green_plate=True)

        white_score = self._score_character_boxes(white_boxes, resized_w, resized_h)
        green_score = self._score_character_boxes(green_boxes, resized_w, resized_h)
        if green_score > white_score:
            binary = green_binary
            boxes = green_boxes
        else:
            binary = white_binary
            boxes = white_boxes

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
        for x, y, w, h in boxes:
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
            self._save_segment_debug(
                save_dir,
                basename,
                plate_surface,
                binary,
                resized,
                resized_boxes,
                roi_boxes,
                char_images,
            )

        return char_images, roi_boxes, surface_box

    def boxes_to_original(self, char_boxes, plate_box):
        plate_x, plate_y, _, _ = plate_box
        return [
            (plate_x + x, plate_y + y, w, h)
            for x, y, w, h in char_boxes
        ]

    def save_original_debug(self, image, char_boxes, save_dir, basename):
        output_dir = os.path.join(save_dir, basename)
        os.makedirs(output_dir, exist_ok=True)

        debug_img = image.copy()
        for idx, (x, y, w, h) in enumerate(char_boxes, start=1):
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

    def _build_white_plate_binary(self, gray):
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
        return cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
            iterations=1,
        )

    def _build_green_plate_binary(self, gray):
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
            iterations=1,
        )

    def _find_character_boxes(self, binary, image_w, image_h, is_green_plate=False):
        search_x = int(image_w * 0.04)
        search_y = int(image_h * 0.18)
        search_w = max(1, int(image_w * 0.92))
        search_h = max(1, int(image_h * 0.68))
        search_binary = binary[search_y : search_y + search_h, search_x : search_x + search_w]

        contours, _ = cv2.findContours(search_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x += search_x
            y += search_y
            area = w * h
            aspect = w / float(h + 1e-6)
            center_y = (y + h * 0.5) / image_h
            fill_ratio = cv2.contourArea(contour) / float(area + 1e-6)

            if not (image_h * 0.28 <= h <= image_h * 0.74):
                continue
            if not (image_w * 0.012 <= w <= image_w * 0.18):
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
            split_boxes.extend(self._split_wide_character_box(binary, box))

        boxes = self._filter_character_artifacts(split_boxes, image_w, image_h)
        boxes = self._merge_character_fragments(boxes)
        if is_green_plate:
            boxes = self._filter_green_plate_border_artifacts(boxes, image_w)
        return sorted(boxes, key=lambda item: item[0])

    def _score_character_boxes(self, boxes, image_w, image_h):
        if not boxes:
            return 0.0

        count = len(boxes)
        count_score = max(0.0, 1.0 - abs(count - 5.5) / 4.0)
        heights = np.array([box[3] for box in boxes], dtype=np.float32)
        centers_y = np.array([box[1] + box[3] * 0.5 for box in boxes], dtype=np.float32)
        span = max(box[0] + box[2] for box in boxes) - min(box[0] for box in boxes)

        height_score = max(0.0, 1.0 - float(np.std(heights)) / (float(np.median(heights)) + 1e-6))
        align_score = max(0.0, 1.0 - float(np.std(centers_y)) / (image_h * 0.18))
        span_score = self._score_similarity(span / float(image_w + 1e-6), 0.42, 0.34)

        return (2.0 * count_score) + height_score + align_score + span_score

    def _crop_plate_surface(self, plate_roi):
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

            score = area * self._score_similarity(aspect, 3.2, 2.6)
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

    def _split_wide_character_box(self, binary, box):
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

    def _filter_character_artifacts(self, boxes, image_w, image_h):
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

    def _merge_character_fragments(self, boxes):
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

    def _filter_green_plate_border_artifacts(self, boxes, image_w):
        if len(boxes) <= 5:
            return boxes

        edge_limit = image_w * 0.08
        filtered = [
            box
            for box in boxes
            if not (box[0] < edge_limit or box[0] + box[2] > image_w - edge_limit)
        ]

        return filtered if len(filtered) >= 5 else boxes

    def _save_segment_debug(
        self,
        save_dir,
        basename,
        plate_surface,
        binary,
        resized,
        resized_boxes,
        roi_boxes,
        char_images,
    ):
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
        for idx, char in enumerate(char_images, start=1):
            cv2.imwrite(os.path.join(output_dir, f"char_{idx:02d}.jpg"), char)

    @staticmethod
    def _score_similarity(value, target, tolerance):
        return max(0.0, 1.0 - abs(value - target) / tolerance)
