"""Incremental, read-only traversal and bounded image decoding."""

import hashlib
import os
import sqlite3
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from PIL import Image, ImageOps

from facearchive.metadata import indexed_metadata

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
THUMBNAIL_SIZE = 256
RECOGNITION_SIZE = 1280
MAX_PIXELS = 48_000_000
REPARSE_POINT = 0x400


def ordinary(info):
    return stat.S_ISREG(info.st_mode) and not (
        getattr(info, "st_file_attributes", 0) & REPARSE_POINT
    )


@contextmanager
def open_original(path):
    expected = os.lstat(path)
    if not ordinary(expected):
        raise OSError("Photo is no longer an ordinary file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        actual = os.fstat(stream.fileno())
        if not ordinary(actual) or not os.path.samestat(expected, actual):
            raise OSError("Photo was replaced while opening")
        yield stream


def open_scaled(path, size):
    with open_original(path) as stream, Image.open(stream) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Image exceeds the 48-megapixel decoding limit")
        image.draft("RGB", (size, size))
        image.thumbnail((size, size))
        return ImageOps.exif_transpose(image).convert("RGB")


def thumbnail_path(library, path, info):
    key = hashlib.sha256(os.fsencode(path))
    key.update(f":{info.st_mtime_ns}:{info.st_size}".encode())
    return library.thumbnails / (key.hexdigest() + ".jpg")


def write_thumbnail(image, target):
    image.thumbnail((THUMBNAIL_SIZE, THUMBNAIL_SIZE))
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with stream:
                image.save(stream, "JPEG", quality=82)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def scan(library, cancelled, progress, engine=None, *, report_issue=None, locations=False):
    with library.connect() as db:
        folders = db.execute("SELECT * FROM folders ORDER BY id").fetchall()
    indexed = errors = visited = 0

    def record_issue(stage, path, error):
        nonlocal errors
        errors += 1
        if report_issue is not None:
            report_issue(stage, path, error)

    for folder in folders:
        if cancelled():
            break
        root = Path(folder["path"])
        incomplete = False

        def traversal_error(error):
            nonlocal incomplete
            incomplete = True
            record_issue("Folder access", error.filename or root, error)

        with library.connect() as db:
            db.execute("CREATE TEMP TABLE seen(path TEXT PRIMARY KEY) WITHOUT ROWID")
            try:
                root_info = root.lstat()
                if not stat.S_ISDIR(root_info.st_mode) or (
                    getattr(root_info, "st_file_attributes", 0) & REPARSE_POINT
                ):
                    raise OSError("Library root is no longer an ordinary directory")
            except OSError as error:
                traversal_error(error)
                continue
            for directory, dirs, files in os.walk(root, onerror=traversal_error):
                if cancelled():
                    break
                for name in dirs[:]:
                    try:
                        info = (Path(directory) / name).lstat()
                        if stat.S_ISLNK(info.st_mode) or (
                            getattr(info, "st_file_attributes", 0) & REPARSE_POINT
                        ):
                            dirs.remove(name)
                    except OSError as error:
                        dirs.remove(name)
                        traversal_error(error)
                dirs.sort()
                for name in sorted(files):
                    if cancelled():
                        break
                    path = Path(directory) / name
                    if path.suffix.lower() not in EXTENSIONS:
                        continue
                    db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (str(path),))
                    visited += 1
                    try:
                        info = path.lstat()
                        if ordinary(info):
                            updated, inference_error = index_photo(
                                db, library, folder["id"], path, info, engine, locations
                            )
                            indexed += updated
                            if inference_error is not None:
                                record_issue("Face analysis", path, inference_error)
                        else:
                            db.execute("DELETE FROM photos WHERE path = ?", (str(path),))
                    except (OSError, ValueError, Image.DecompressionBombError) as error:
                        record_issue("Photo indexing", path, error)
                    if visited % 32 == 0:
                        db.commit()
                        progress(visited, indexed, errors)
            if not incomplete and not cancelled():
                db.execute(
                    """
                    DELETE FROM photos WHERE folder_id = ?
                    AND NOT EXISTS (SELECT 1 FROM seen WHERE seen.path = photos.path)
                """,
                    (folder["id"],),
                )
            library.prune_people(db)
    if not cancelled():
        try:
            clean_cache(library, cancelled, record_issue)
        except OSError as error:
            record_issue("Thumbnail cleanup", library.thumbnails, error)
    return visited, indexed, errors


def clean_cache(library, cancelled, record_issue):
    with library.connect() as db, os.scandir(library.thumbnails) as entries:
        for entry in entries:
            if cancelled():
                break
            stem, suffix = os.path.splitext(entry.name)
            if (
                suffix != ".jpg"
                or len(stem) != 64
                or any(c not in "0123456789abcdef" for c in stem)
            ):
                continue
            if (
                db.execute("SELECT 1 FROM photos WHERE thumbnail = ?", (entry.name,)).fetchone()
                is None
            ):
                try:
                    Path(entry.path).unlink(missing_ok=True)
                except OSError as error:
                    record_issue("Thumbnail cleanup", entry.path, error)


def index_photo(db, library, folder_id, path, info, engine, locations):
    row = db.execute("SELECT * FROM photos WHERE path = ?", (str(path),)).fetchone()
    changed = row is None or (row["mtime"], row["size"]) != (info.st_mtime_ns, info.st_size)
    if (
        changed
        and row is not None
        and (row["content_hash"] is not None or row["location_area"] is not None)
    ):
        db.execute(
            "UPDATE photos SET content_hash = NULL, location_area = NULL WHERE id = ?", (row["id"],)
        )
    target = thumbnail_path(library, path, info)
    needs_thumbnail = not target.is_file()
    analyze = engine is not None and (changed or row["face_version"] != engine.version)
    inference_error = None
    faces = []
    image = None
    if analyze:
        try:
            image = open_scaled(path, RECOGNITION_SIZE)
            faces = engine.detect_image(image)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            inference_error = error
            analyze = False
    if needs_thumbnail:
        if image is None:
            image = open_scaled(path, THUMBNAIL_SIZE)
        write_thumbnail(image, target)
    needs_date = changed or row["captured_on"] is None
    needs_location = locations and (changed or row["location_area"] is None)
    if not (changed or analyze or needs_date or needs_location):
        return 0, inference_error
    captured_on = None if changed else row["captured_on"]
    area = None if changed else row["location_area"]
    if needs_date or needs_location:
        if image is None:
            with open_original(path) as stream, Image.open(stream) as header:
                day, found_area = indexed_metadata(
                    header, info.st_mtime_ns, needs_date, needs_location
                )
        else:
            day, found_area = indexed_metadata(image, info.st_mtime_ns, needs_date, needs_location)
        if needs_date:
            captured_on = day
        if needs_location:
            area = found_area
    current = path.lstat()
    if (current.st_mtime_ns, current.st_size, current.st_ino) != (
        info.st_mtime_ns,
        info.st_size,
        info.st_ino,
    ):
        raise OSError("Photo changed while scanning")
    if not changed and not analyze:
        db.execute(
            "UPDATE photos SET captured_on = ?, location_area = ? WHERE id = ?",
            (captured_on, area, row["id"]),
        )
        return 0, inference_error
    db.execute("SAVEPOINT photo")
    try:
        photo = db.execute(
            """
            INSERT INTO photos(
                folder_id, path, mtime, size, thumbnail, face_version, captured_on, location_area
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime, size=excluded.size,
                thumbnail=excluded.thumbnail, face_version=excluded.face_version,
                captured_on=excluded.captured_on, location_area=excluded.location_area
            RETURNING id
        """,
            (
                folder_id,
                str(path),
                info.st_mtime_ns,
                info.st_size,
                target.name,
                engine.version if analyze else None,
                captured_on,
                area,
            ),
        ).fetchone()[0]
        db.execute("DELETE FROM faces WHERE photo_id = ?", (photo,))
        if analyze:
            engine.store(db, photo, faces)
    except (sqlite3.Error, ValueError):
        db.execute("ROLLBACK TO photo")
        raise
    finally:
        db.execute("RELEASE photo")
    return int(changed), inference_error
