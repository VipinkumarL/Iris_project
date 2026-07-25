import cv2
import numpy as np

from iris_processing import enhance_eye_image


POLAR_ANGLE_SAMPLES = 72
POLAR_RADIUS_SAMPLES = 32
IRIS_INNER_RATIO = 0.28
IRIS_OUTER_RATIO = 0.95
RADIAL_BANDS = 4
ANGULAR_SECTORS = 12


def _unwrap_iris(image, center, radius):
    polar_image = cv2.warpPolar(
        image,
        (POLAR_ANGLE_SAMPLES, POLAR_RADIUS_SAMPLES),
        center,
        radius,
        cv2.WARP_POLAR_LINEAR,
    )

    if polar_image is None or polar_image.size == 0:
        return None

    iris_start = max(0, int(POLAR_RADIUS_SAMPLES * IRIS_INNER_RATIO))
    iris_end = min(POLAR_RADIUS_SAMPLES, int(POLAR_RADIUS_SAMPLES * IRIS_OUTER_RATIO))
    iris_strip = polar_image[iris_start:iris_end, :]
    if iris_strip.size == 0:
        return None

    iris_strip = cv2.normalize(iris_strip, None, 0, 255, cv2.NORM_MINMAX)
    return iris_strip.astype(np.uint8)


def _sector_feature_maps(iris_strip):
    rows, cols = iris_strip.shape
    band_height = rows / RADIAL_BANDS
    sector_width = cols / ANGULAR_SECTORS

    sobel_x = cv2.Sobel(iris_strip, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(iris_strip, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    intensity_map = np.zeros((RADIAL_BANDS, ANGULAR_SECTORS), dtype=np.float32)
    deviation_map = np.zeros((RADIAL_BANDS, ANGULAR_SECTORS), dtype=np.float32)
    gradient_map = np.zeros((RADIAL_BANDS, ANGULAR_SECTORS), dtype=np.float32)

    for band_index in range(RADIAL_BANDS):
        row_start = int(round(band_index * band_height))
        row_end = int(round((band_index + 1) * band_height))

        for sector_index in range(ANGULAR_SECTORS):
            col_start = int(round(sector_index * sector_width))
            col_end = int(round((sector_index + 1) * sector_width))

            sector = iris_strip[row_start:row_end, col_start:col_end]
            gradient_sector = gradient_magnitude[row_start:row_end, col_start:col_end]

            if sector.size == 0:
                continue

            intensity_map[band_index, sector_index] = float(np.mean(sector)) / 255.0
            deviation_map[band_index, sector_index] = float(np.std(sector)) / 128.0
            gradient_map[band_index, sector_index] = float(np.mean(gradient_sector)) / 255.0

    return intensity_map, deviation_map, gradient_map


def _flatten_sector_maps(*maps):
    features = []

    for feature_map in maps:
        for value in feature_map.flatten():
            features.append(round(float(value), 4))

    return features


def _iris_code_bits(intensity_map, deviation_map, gradient_map):
    bits = []

    for band_index in range(RADIAL_BANDS):
        band_intensity_mean = float(np.mean(intensity_map[band_index, :]))
        band_gradient_mean = float(np.mean(gradient_map[band_index, :]))

        for sector_index in range(ANGULAR_SECTORS):
            previous_sector = (sector_index - 1) % ANGULAR_SECTORS
            current_intensity = float(intensity_map[band_index, sector_index])
            current_deviation = float(deviation_map[band_index, sector_index])
            current_gradient = float(gradient_map[band_index, sector_index])

            bits.append(1.0 if current_intensity >= band_intensity_mean else 0.0)
            bits.append(
                1.0
                if current_intensity >= float(intensity_map[band_index, previous_sector])
                else 0.0
            )
            bits.append(1.0 if current_gradient >= band_gradient_mean else 0.0)

            if band_index > 0:
                bits.append(
                    1.0
                    if current_deviation >= float(deviation_map[band_index - 1, sector_index])
                    else 0.0
                )
            else:
                bits.append(1.0 if current_deviation >= 0.5 else 0.0)

    return bits


def _texture_transitions(iris_strip):
    band_means = np.mean(iris_strip, axis=1)
    sector_means = np.mean(iris_strip, axis=0)
    sector_deltas = np.diff(sector_means, append=sector_means[0])
    band_deltas = np.diff(band_means)

    features = []

    for value in sector_means[:: max(1, len(sector_means) // 18)]:
        features.append(float(value) / 255.0)

    for value in sector_deltas[:: max(1, len(sector_deltas) // 18)]:
        features.append((float(value) + 255.0) / 510.0)

    for value in band_deltas:
        features.append((float(value) + 255.0) / 510.0)

    return features


def extract_features(image_path, circles):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []

    image = enhance_eye_image(image)
    height, width = image.shape
    features = []

    for (x, y, r) in circles:
        x = int(x)
        y = int(y)
        r = int(r)

        if r < 20:
            continue

        if x - r < 0 or y - r < 0 or x + r >= width or y + r >= height:
            continue

        iris_strip = _unwrap_iris(image, (float(x), float(y)), float(r))
        if iris_strip is None:
            continue

        normalized_circle = [
            round(x / width, 4),
            round(y / height, 4),
            round(r / min(width, height), 4),
        ]

        intensity_map, deviation_map, gradient_map = _sector_feature_maps(iris_strip)
        texture_features = _flatten_sector_maps(intensity_map, deviation_map, gradient_map)
        transition_features = _texture_transitions(iris_strip)
        iris_code_bits = _iris_code_bits(intensity_map, deviation_map, gradient_map)
        feature_vector = tuple(
            round(float(value), 4)
            for value in normalized_circle + texture_features + transition_features + iris_code_bits
        )
        features.append(feature_vector)

    return features
