"""Fetch build-time resources from OpenCV Zoo and verify the pinned model digests."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

FILES = (
    (
        "face_detection_yunet",
        "face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    (
        "face_recognition_sface",
        "face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
)


def download(url, limit):
    with urlopen(url, timeout=60) as response:
        data = response.read(limit + 1)
    if not data or len(data) > limit:
        raise ValueError(f"Unexpected download size for {url}")
    return data


def main(destination):
    destination.mkdir(parents=True, exist_ok=True)
    for directory, name, digest in FILES:
        relative = f"opencv/opencv_zoo/main/models/{directory}"
        data = download(f"https://media.githubusercontent.com/media/{relative}/{name}", 40_000_000)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"SHA-256 mismatch for {name}")
        license_text = download(f"https://raw.githubusercontent.com/{relative}/LICENSE", 32_768)
        (destination / name).write_bytes(data)
        (destination / f"{directory}-LICENSE.txt").write_bytes(license_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    main(parser.parse_args().destination)
