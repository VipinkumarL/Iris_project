import cv2
import numpy as np


FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
ALT_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
)
PROFILE_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)


MULTI_PERSON_WARNING = "More than one person appears in the frame."


def _significant_eyes(eyes, frame_shape):
    frame_height, frame_width = frame_shape
    frame_area = frame_width * frame_height
    significant_eyes = []

    for x, y, w, h in eyes:
        area_ratio = (w * h) / frame_area
        if area_ratio < 0.0025:
            continue

        significant_eyes.append((int(x), int(y), int(w), int(h)))

    return significant_eyes


def _estimated_subject_count_from_eyes(eyes):
    if len(eyes) <= 2:
        return 1 if eyes else 0

    centers = sorted((x + (w / 2.0), y + (h / 2.0)) for x, y, w, h in eyes)
    groups = 1
    last_center_x, last_center_y = centers[0]

    for center_x, center_y in centers[1:]:
        if abs(center_x - last_center_x) > 120 or abs(center_y - last_center_y) > 70:
            groups += 1
        last_center_x, last_center_y = center_x, center_y

    return groups


def _significant_faces(faces, frame_shape):
    if len(faces) <= 1:
        return [tuple(int(value) for value in face) for face in faces]

    frame_height, frame_width = frame_shape
    frame_area = frame_width * frame_height
    significant_faces = []

    for x, y, w, h in faces:
        area_ratio = (w * h) / frame_area
        if area_ratio < 0.025:
            continue

        significant_faces.append((int(x), int(y), int(w), int(h)))

    return significant_faces


def _rect_iou(rect_a, rect_b):
    ax, ay, aw, ah = rect_a
    bx, by, bw, bh = rect_b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)

    if right <= left or bottom <= top:
        return 0.0

    intersection = float((right - left) * (bottom - top))
    area_a = float(aw * ah)
    area_b = float(bw * bh)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def _merge_face_detections(detections):
    merged = []

    for detection in detections:
        matched_index = None
        for index, existing in enumerate(merged):
            if _rect_iou(detection, existing) > 0.28:
                matched_index = index
                break

        if matched_index is None:
            merged.append(detection)
            continue

        ex, ey, ew, eh = merged[matched_index]
        dx, dy, dw, dh = detection
        x1 = min(ex, dx)
        y1 = min(ey, dy)
        x2 = max(ex + ew, dx + dw)
        y2 = max(ey + eh, dy + dh)
        merged[matched_index] = (x1, y1, x2 - x1, y2 - y1)

    return merged


def _detect_faces(gray_image):
    if FACE_CASCADE.empty() and ALT_FACE_CASCADE.empty() and PROFILE_FACE_CASCADE.empty():
        return []

    detections = []
    detection_inputs = [gray_image]
    enhanced = enhance_eye_image(gray_image)
    if enhanced is not None:
        detection_inputs.append(enhanced)

    frontal_cascades = [cascade for cascade in (FACE_CASCADE, ALT_FACE_CASCADE) if not cascade.empty()]
    for source_image in detection_inputs:
        for cascade in frontal_cascades:
            try:
                frontal_faces = cascade.detectMultiScale(
                    source_image,
                    scaleFactor=1.08,
                    minNeighbors=6,
                    minSize=(80, 80),
                )
            except cv2.error:
                frontal_faces = []
            detections.extend(_significant_faces(frontal_faces, gray_image.shape))

    if not PROFILE_FACE_CASCADE.empty():
        for source_image in detection_inputs:
            try:
                profile_faces = PROFILE_FACE_CASCADE.detectMultiScale(
                    source_image,
                    scaleFactor=1.08,
                    minNeighbors=5,
                    minSize=(80, 80),
                )
            except cv2.error:
                profile_faces = []
            detections.extend(_significant_faces(profile_faces, gray_image.shape))

            flipped = cv2.flip(source_image, 1)
            try:
                flipped_faces = PROFILE_FACE_CASCADE.detectMultiScale(
                    flipped,
                    scaleFactor=1.08,
                    minNeighbors=5,
                    minSize=(80, 80),
                )
            except cv2.error:
                flipped_faces = []
            frame_width = gray_image.shape[1]
            for x, y, w, h in _significant_faces(flipped_faces, gray_image.shape):
                detections.append((frame_width - x - w, y, w, h))

    return _merge_face_detections(detections)


def _eye_center(eye):
    x, y, w, h = eye
    return x + (w / 2.0), y + (h / 2.0)


def _eyes_outside_primary_face(eyes, face):
    if not face:
        return eyes

    face_x, face_y, face_w, face_h = face
    outside = []

    for eye in eyes:
        center_x, center_y = _eye_center(eye)
        if not (face_x <= center_x <= face_x + face_w and face_y <= center_y <= face_y + face_h):
            outside.append(eye)

    return outside


def _secondary_eye_clusters(eyes, primary_face):
    outside_eyes = _eyes_outside_primary_face(eyes, primary_face)
    if not outside_eyes:
        return 0

    sorted_eyes = sorted(outside_eyes, key=lambda eye: (_eye_center(eye)[0], _eye_center(eye)[1]))
    clusters = 1
    last_x, last_y = _eye_center(sorted_eyes[0])

    for eye in sorted_eyes[1:]:
        center_x, center_y = _eye_center(eye)
        if abs(center_x - last_x) > 90 or abs(center_y - last_y) > 55:
            clusters += 1
        last_x, last_y = center_x, center_y

    return clusters


def _has_strong_multi_person_eye_evidence(eyes, primary_face=None):
    if not eyes:
        return False

    estimated_subjects = _estimated_subject_count_from_eyes(eyes)
    if primary_face:
        outside_eyes = _eyes_outside_primary_face(eyes, primary_face)
        secondary_clusters = _secondary_eye_clusters(eyes, primary_face)

        # Eye cascades are noisy around eyebrows, eyelids, and reflections.
        # Treat them as a second person only when there are multiple out-of-face
        # detections forming a separate cluster, or a very large number overall.
        if len(outside_eyes) >= 2 and secondary_clusters > 0:
            return True

        return estimated_subjects >= 3 or len(eyes) >= 5

    return estimated_subjects >= 3 or len(eyes) >= 5


def enhance_eye_image(gray_image):
    if gray_image is None or gray_image.size == 0:
        return None

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast_image = clahe.apply(gray_image)
    denoised_image = cv2.GaussianBlur(contrast_image, (5, 5), 0)
    return denoised_image


def _select_primary_eye_region(gray_image):
    if EYE_CASCADE.empty():
        return None

    height, width = gray_image.shape
    candidates = []

    faces = _detect_faces(gray_image)
    for face_x, face_y, face_w, face_h in faces[:1]:
            face_roi = gray_image[face_y : face_y + face_h, face_x : face_x + face_w]
            try:
                eyes = EYE_CASCADE.detectMultiScale(
                    face_roi,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20),
                )
            except cv2.error:
                eyes = []

            for eye_x, eye_y, eye_w, eye_h in eyes:
                absolute_eye = (
                    int(face_x + eye_x),
                    int(face_y + eye_y),
                    int(eye_w),
                    int(eye_h),
                )
                candidates.append(absolute_eye)

    if not candidates:
        try:
            eyes = EYE_CASCADE.detectMultiScale(
                gray_image,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(24, 24),
            )
        except cv2.error:
            eyes = []
        candidates.extend(_significant_eyes(eyes, gray_image.shape))

    if not candidates:
        return None

    def candidate_score(candidate):
        eye_x, eye_y, eye_w, eye_h = candidate
        eye_center_x = eye_x + (eye_w / 2.0)
        eye_center_y = eye_y + (eye_h / 2.0)
        horizontal_centering = 1.0 - abs((eye_center_x / width) - 0.5)
        vertical_preference = 1.0 - abs((eye_center_y / height) - 0.38)
        size_score = (eye_w * eye_h) / float(width * height)
        return (size_score * 8.0) + horizontal_centering + vertical_preference

    eye_x, eye_y, eye_w, eye_h = max(candidates, key=candidate_score)
    padding_x = int(eye_w * 1.4)
    padding_top = int(eye_h * 1.1)
    padding_bottom = int(eye_h * 1.6)

    crop_x1 = max(0, eye_x - padding_x)
    crop_y1 = max(0, eye_y - padding_top)
    crop_x2 = min(width, eye_x + eye_w + padding_x)
    crop_y2 = min(height, eye_y + eye_h + padding_bottom)

    if crop_x2 - crop_x1 < 60 or crop_y2 - crop_y1 < 60:
        return None

    return crop_x1, crop_y1, crop_x2, crop_y2


def assess_subject_layout(gray_image):
    if gray_image is None or gray_image.size == 0:
        return {"ok": False, "face_count": None, "eye_count": None, "reason": "Captured image could not be read."}

    if FACE_CASCADE.empty() and ALT_FACE_CASCADE.empty() and PROFILE_FACE_CASCADE.empty() and EYE_CASCADE.empty():
        return {"ok": True, "face_count": None, "eye_count": None, "reason": None}

    height, width = gray_image.shape
    scale = 640.0 / max(width, height) if max(width, height) > 640 else 1.0
    resized = cv2.resize(gray_image, None, fx=scale, fy=scale) if scale != 1.0 else gray_image

    try:
        faces = _detect_faces(resized)
    except cv2.error:
        faces = []

    if len(faces) > 1:
        return {
            "ok": False,
            "face_count": int(len(faces)),
            "eye_count": None,
            "reason": MULTI_PERSON_WARNING,
        }

    eyes = []
    if not EYE_CASCADE.empty():
        try:
            eyes = EYE_CASCADE.detectMultiScale(
                resized,
                scaleFactor=1.1,
                minNeighbors=8,
                minSize=(24, 24),
            )
        except cv2.error:
            eyes = []
        eyes = _significant_eyes(eyes, resized.shape)

    if len(faces) == 1 and not EYE_CASCADE.empty():
        x, y, w, h = faces[0]
        face_roi = resized[y : y + h, x : x + w]
        try:
            face_eyes = EYE_CASCADE.detectMultiScale(
                face_roi,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(18, 18),
            )
        except cv2.error:
            face_eyes = []
        if _has_strong_multi_person_eye_evidence(eyes, faces[0]):
            return {
                "ok": False,
                "face_count": 1,
                "eye_count": int(len(eyes)),
                "reason": MULTI_PERSON_WARNING,
            }

        return {
            "ok": True,
            "face_count": 1,
            "eye_count": int(len(face_eyes)),
            "reason": None,
        }

    if not EYE_CASCADE.empty():
        if _has_strong_multi_person_eye_evidence(eyes):
            return {
                "ok": False,
                "face_count": int(len(faces)),
                "eye_count": int(len(eyes)),
                "reason": MULTI_PERSON_WARNING,
            }

    return {
        "ok": True,
        "face_count": int(len(faces)),
        "eye_count": int(len(eyes)) if not EYE_CASCADE.empty() and 'eyes' in locals() else None,
        "reason": None,
    }


def assess_image_quality(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return {
            "ok": False,
            "reason": "Captured image could not be read.",
            "brightness": None,
            "sharpness": None,
        }

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    subject_layout = assess_subject_layout(gray_image)
    if not subject_layout["ok"]:
        return {
            "ok": False,
            "reason": subject_layout["reason"],
            "brightness": None,
            "sharpness": None,
        }

    brightness = float(np.mean(gray_image))
    sharpness = float(cv2.Laplacian(gray_image, cv2.CV_64F).var())

    if brightness < 55:
        return {
            "ok": False,
            "reason": "Lighting is too low. Please move to a brighter place and try again.",
            "brightness": round(brightness, 2),
            "sharpness": round(sharpness, 2),
        }

    if sharpness < 45:
        return {
            "ok": False,
            "reason": "Image is too blurry. Keep the camera steady and try again.",
            "brightness": round(brightness, 2),
            "sharpness": round(sharpness, 2),
        }

    return {
        "ok": True,
        "reason": None,
        "brightness": round(brightness, 2),
        "sharpness": round(sharpness, 2),
    }


def _circle_score(gray_image, x, y, r):
    height, width = gray_image.shape

    if r < 20 or r > min(width, height) * 0.35:
        return None

    if x - r < 0 or y - r < 0 or x + r >= width or y + r >= height:
        return None

    mask = np.zeros_like(gray_image)
    cv2.circle(mask, (int(x), int(y)), int(r * 0.82), 255, -1)
    pixels = gray_image[mask == 255]
    if pixels.size == 0:
        return None

    mean_intensity = float(np.mean(pixels))
    std_intensity = float(np.std(pixels))

    # Darker and more textured circles are more likely to be valid iris regions.
    return (170 - mean_intensity) + (0.35 * std_intensity)


def _detect_with_hough(gray_image):
    if gray_image is None or gray_image.size == 0 or min(gray_image.shape) < 60:
        return None

    blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)
    try:
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(60, gray_image.shape[1] // 6),
            param1=80,
            param2=28,
            minRadius=20,
            maxRadius=max(21, int(min(gray_image.shape) * 0.35)),
        )
    except cv2.error:
        return None

    if circles is None:
        return None

    best_circle = None
    best_score = None

    for x, y, r in np.round(circles[0]).astype(int):
        score = _circle_score(gray_image, x, y, r)
        if score is None:
            continue

        if best_score is None or score > best_score:
            best_circle = (x, y, r)
            best_score = score

    if best_circle is None:
        return None

    x, y, r = best_circle
    return [(int(x), int(y), int(r))]


def _detect_with_contours(gray_image):
    if gray_image is None or gray_image.size == 0:
        return None

    blurred = cv2.GaussianBlur(gray_image, (7, 7), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    best_circle = None
    best_score = None

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area < 1200 or perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if not (0.68 < circularity < 1.2):
            continue

        (x, y), r = cv2.minEnclosingCircle(contour)
        score = _circle_score(gray_image, int(x), int(y), int(r))
        if score is None:
            continue

        score += area / 1000.0

        if best_score is None or score > best_score:
            best_circle = (int(x), int(y), int(r))
            best_score = score

    if best_circle is None:
        return None

    return [best_circle]


def detect_iris(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    crop = _select_primary_eye_region(gray_image)
    target_image = gray_image
    offset_x = 0
    offset_y = 0

    if crop:
        crop_x1, crop_y1, crop_x2, crop_y2 = crop
        target_image = gray_image[crop_y1:crop_y2, crop_x1:crop_x2]
        offset_x = crop_x1
        offset_y = crop_y1

    enhanced_image = enhance_eye_image(target_image)

    circles = _detect_with_hough(enhanced_image)
    if circles:
        return [(x + offset_x, y + offset_y, r) for x, y, r in circles]

    circles = _detect_with_contours(enhanced_image)
    if circles:
        return [(x + offset_x, y + offset_y, r) for x, y, r in circles]

    if crop:
        fallback_image = enhance_eye_image(gray_image)
        circles = _detect_with_hough(fallback_image)
        if circles:
            return circles
        return _detect_with_contours(fallback_image)

    return None
