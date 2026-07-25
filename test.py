import sys

from iris_processing import detect_iris


def main():
    if len(sys.argv) < 2:
        print("Usage: python test.py <image_path>")
        return

    image_path = sys.argv[1]
    result = detect_iris(image_path)
    print(result)


if __name__ == "__main__":
    main()
