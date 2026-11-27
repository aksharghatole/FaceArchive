"""Extract selected EXIF fields without storing the full metadata block."""

import math
import struct
from datetime import date, timedelta

from PIL import ExifTags

EPOCH = date(1970, 1, 1)
NANOSECONDS_PER_DAY = 86_400_000_000_000


EXIF_ERRORS = (OSError, ValueError, SyntaxError, struct.error)


def read_exif(image):
    try:
        return image.getexif()
    except EXIF_ERRORS:
        return None


def capture_day(exif, mtime):
    try:
        camera_tags = exif.get_ifd(ExifTags.IFD.Exif) if exif else {}
    except EXIF_ERRORS:
        camera_tags = {}
    for tag in (ExifTags.Base.DateTimeOriginal, ExifTags.Base.DateTimeDigitized):
        for tags in (camera_tags, exif or {}):
            value = tags.get(tag)
            if isinstance(value, str):
                try:
                    return date.fromisoformat(value[:10].replace(":", "-")).isoformat()
                except ValueError:
                    pass
    return (EPOCH + timedelta(days=mtime // NANOSECONDS_PER_DAY)).isoformat()


def gps_axis(gps, reference, values, positive, negative, maximum):
    direction = gps[reference]
    if direction not in (positive, negative):
        raise ValueError("Invalid GPS direction")
    degrees, minutes, seconds = map(float, gps[values])
    if not all(math.isfinite(v) and v >= 0 for v in (degrees, minutes, seconds)):
        raise ValueError("Invalid GPS angle")
    if minutes >= 60 or seconds >= 60:
        raise ValueError("Invalid GPS minutes or seconds")
    coordinate = degrees + minutes / 60 + seconds / 3600
    if coordinate > maximum:
        raise ValueError("GPS coordinate out of range")
    return round(coordinate * (-100 if direction == negative else 100))


def location_area(exif):
    """Represent an approximate location by a pair of centidegrees."""
    if not exif:
        return ""
    try:
        gps = exif.get_ifd(ExifTags.IFD.GPSInfo)
        latitude = gps_axis(
            gps, ExifTags.GPS.GPSLatitudeRef, ExifTags.GPS.GPSLatitude, "N", "S", 90
        )
        longitude = gps_axis(
            gps, ExifTags.GPS.GPSLongitudeRef, ExifTags.GPS.GPSLongitude, "E", "W", 180
        )
    except (*EXIF_ERRORS, KeyError, TypeError, OverflowError, ZeroDivisionError):
        return ""
    return f"{latitude / 100:.2f}, {longitude / 100:.2f}"


def indexed_metadata(image, mtime, read_date, read_location):
    exif = read_exif(image)
    day = capture_day(exif, mtime) if read_date else None
    area = location_area(exif) if read_location else None
    return day, area
