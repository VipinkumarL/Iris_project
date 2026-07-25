import hashlib


def generate_hash(pattern):
    pattern_str = "".join(map(str, pattern))
    return hashlib.sha256(pattern_str.encode()).hexdigest()
