def _flatten_features(features):
    flattened = []

    for feature_group in features:
        flattened.extend(feature_group)

    return flattened


def create_constellation(features):
    if not features:
        return []

    flattened = _flatten_features(features)
    scaled = [int(round(value * 1000)) for value in flattened]
    pattern = scaled[:]

    for index in range(len(scaled) - 1):
        pattern.append(abs(scaled[index + 1] - scaled[index]))

    return pattern
