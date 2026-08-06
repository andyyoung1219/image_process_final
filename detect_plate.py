import os

import cv2
import numpy as np


class PlateDetector:
    def detect(self, image_path, debug=False, save_dir="debug_roi"):
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
        gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray) #直方圖等化
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, 85]), #lower
            np.array([180, 115, 255]), #upper
        )
        green_mask = cv2.inRange(
            hsv,
            np.array([55, 15, 15]),
            np.array([130, 210, 240]),
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

        edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5)) # 19x5的SE
        edge_morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, edge_kernel, iterations=2) #close
        edge_morph = cv2.morphologyEx(
            edge_morph,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        ) #open

        green_edges = cv2.bitwise_and(green_mask, edge_mask)
        green_edge_morph = cv2.morphologyEx(
            green_edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5)),
            iterations=1,
        )
        green_edge_morph = cv2.morphologyEx(
            green_edge_morph,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        green_edge_morph_wide = cv2.morphologyEx(
            green_edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (23, 7)),
            iterations=1,
        )
        green_edge_morph_wide = cv2.morphologyEx(
            green_edge_morph_wide,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )

        plate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 7)) # 23x7的SE
        white_morph = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, plate_kernel, iterations=2)
        white_morph = cv2.morphologyEx(
            white_morph,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
            iterations=1,
        )

        raw_candidates = []
        raw_candidates.extend(self._collect_candidates(edge_morph, "edge"))
        raw_candidates.extend(self._collect_candidates(white_morph, "white"))
        raw_candidates.extend(self._collect_candidates(green_edge_morph, "green_edge"))
        raw_candidates.extend(self._collect_candidates(green_edge_morph_wide, "green_edge"))

        candidates = []
        for candidate in self._dedupe_candidates(raw_candidates):
            x, y, w, h = candidate["box"]
            area = w * h #車牌候選面積
            aspect = w / float(h + 1e-6) #車牌候選寬長比例
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
            if y < height * 0.12:
                continue

            roi_gray = gray[y : y + h, x : x + w]
            roi_white = white_mask[y : y + h, x : x + w]
            roi_green = green_mask[y : y + h, x : x + w]
            roi_edges = edge_mask[y : y + h, x : x + w]

            white_ratio = float(np.mean(roi_white > 0))
            green_ratio = float(np.mean(roi_green > 0))
            dark_ratio = float(np.mean(roi_gray < 105))
            edge_density = float(np.mean(roi_edges > 0))
            fill_ratio = candidate["contour_area"] / float(area + 1e-6) # 輪廓實際面積佔外接矩形面積的比例

            green_source = candidate["source"] == "green_edge"
            strong_green_box = green_ratio >= 0.45 and fill_ratio >= 0.45 and edge_density >= 0.16
            green_like = (green_source and green_ratio >= 0.20) or strong_green_box
            if white_ratio < 0.14 and not green_like:
                continue
            max_dark_ratio = 0.90 if green_like else 0.70
            if not (0.04 <= dark_ratio <= max_dark_ratio):
                continue
            if edge_density < 0.035:
                continue
            if green_source and (fill_ratio < 0.18 or edge_density < 0.10 or area_ratio > 0.025):
                continue

            aspect_score = self._score_similarity(aspect, 2.35, 1.85)
            white_score = self._score_similarity(white_ratio, 0.62, 0.45)
            dark_score = self._score_similarity(dark_ratio, 0.30, 0.34)
            edge_score = self._score_similarity(edge_density, 0.24, 0.22)
            fill_score = self._score_similarity(fill_ratio, 0.80, 0.40)
            size_score = self._score_similarity(area_ratio, 0.018, 0.050)

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

            plate_type = "white"
            if green_like:
                green_aspect_score = self._score_similarity(aspect, 2.60, 1.10)
                green_color_score = self._score_similarity(min(green_ratio, 0.75), 0.62, 0.55)
                green_white_score = self._score_similarity(min(white_ratio, 0.80), 0.58, 0.50)
                green_dark_score = self._score_similarity(dark_ratio, 0.55, 0.40)
                green_edge_score = self._score_similarity(edge_density, 0.30, 0.25)
                green_size_score = self._score_similarity(area_ratio, 0.0075, 0.022)
                green_fill_score = self._score_similarity(fill_ratio, 0.52, 0.45)

                green_score = (
                    (3.0 * green_aspect_score)
                    + (2.0 * green_color_score)
                    + (1.5 * green_white_score)
                    + (2.0 * green_dark_score)
                    + (3.0 * green_edge_score)
                    + (1.8 * green_size_score)
                    + (1.2 * green_fill_score)
                    + 1.2
                ) * road_penalty

                if green_score > score:
                    score = green_score
                    plate_type = "green"

            candidates.append(
                {
                    **candidate,
                    "area": area,
                    "aspect": aspect,
                    "area_ratio": area_ratio,
                    "white_ratio": white_ratio,
                    "green_ratio": green_ratio,
                    "dark_ratio": dark_ratio,
                    "edge_density": edge_density,
                    "fill_ratio": fill_ratio,
                    "score": score,
                    "plate_type": plate_type,
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
            cv2.imwrite(os.path.join(save_dir, f"{basename}_green_mask.jpg"), green_mask)
            cv2.imwrite(os.path.join(save_dir, f"{basename}_edge_mask.jpg"), edge_mask)
            cv2.imwrite(os.path.join(save_dir, f"{basename}_combined.jpg"), combined)
            cv2.imwrite(os.path.join(save_dir, f"{basename}_morph.jpg"), edge_morph)
            cv2.imwrite(os.path.join(save_dir, f"{basename}_green_edge_morph.jpg"), green_edge_morph)
            cv2.imwrite(os.path.join(save_dir, f"{basename}_green_edge_morph_wide.jpg"), green_edge_morph_wide)
            cv2.imwrite(os.path.join(save_dir, f"{basename}_white_morph.jpg"), white_morph)
            cv2.imwrite(os.path.join(save_dir, f"{basename}_detected.jpg"), debug_img)
            cv2.imwrite(os.path.join(save_dir, f"{basename}_roi.jpg"), plate_roi)

        return plate_roi, plate_box, candidates

    def _collect_candidates(self, mask, source_name, min_area=250):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #找外圍框

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

    def _dedupe_candidates(self, candidates):
        candidates = sorted(
            candidates,
            key=lambda item: item["box"][2] * item["box"][3],
            reverse=True,
        )

        deduped = []
        for candidate in candidates:
            if all(self._iou(candidate["box"], old["box"]) < 0.55 for old in deduped):
                deduped.append(candidate)

        return deduped

    @staticmethod
    def _score_similarity(value, target, tolerance):
        return max(0.0, 1.0 - abs(value - target) / tolerance)

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
