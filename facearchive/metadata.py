"""Read the camera's calendar date without retaining other EXIF metadata."""

import struct
from datetime import date, timedelta

from PIL import ExifTags

EPOCH = date(1970, 1, 1)
NANOSECONDS_PER_DAY = 86_400_000_000_000


def capture_day(image, mtime):
    fields = (OSError, ValueError, SyntaxError, struct.error)
    try:
        exif = image.getexif()
    except fields:
        exif = {}
    try:
        camera_tags = exif.get_ifd(ExifTags.IFD.Exif) if exif else {}
    except fields:
        camera_tags = {}
    for tag in (ExifTags.Base.DateTimeOriginal, ExifTags.Base.DateTimeDigitized):
        for tags in (camera_tags, exif):
            value = tags.get(tag)
            if isinstance(value, str):
                try:
                    return date.fromisoformat(value[:10].replace(":", "-")).isoformat()
                except ValueError:
                    pass
    return (EPOCH + timedelta(days=mtime // NANOSECONDS_PER_DAY)).isoformat()
