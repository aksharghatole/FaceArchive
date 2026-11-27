"""Copy indexed originals without overwriting any destination file."""

import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp

from facearchive.scanner import open_original, ordinary

CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ExportPhoto:
    path: Path
    mtime: int
    size: int


@dataclass(frozen=True)
class ExportResult:
    directory: Path | None
    copied: int
    failures: tuple[str, ...]
    cancelled: bool


def open_copy(directory, original):
    candidate = directory / original.name
    attempt = 2
    while True:
        try:
            return candidate, candidate.open("xb")
        except FileExistsError:
            candidate = directory / f"{original.stem} ({attempt}){original.suffix}"
            attempt += 1


def copy_photo(photo, directory, cancelled):
    if photo.path.resolve(strict=True) != photo.path:
        raise OSError("Photo path now resolves to a different location")
    with open_original(photo.path) as source:
        initial = os.fstat(source.fileno())
        if (initial.st_mtime_ns, initial.st_size) != (photo.mtime, photo.size):
            raise OSError("Photo changed since indexing. Scan the folder again.")
        target, output = open_copy(directory, photo.path)
        complete = False
        identity = None
        try:
            copied = 0
            with output:
                identity = os.fstat(output.fileno())
                while not cancelled():
                    chunk = source.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    output.write(chunk)
                    copied += len(chunk)
            if cancelled():
                return False
            current = os.lstat(photo.path)
            if (
                not ordinary(current)
                or not os.path.samestat(initial, current)
                or (current.st_mtime_ns, current.st_size, copied)
                != (photo.mtime, photo.size, photo.size)
            ):
                raise OSError("Photo changed during export")
            complete = True
            return True
        finally:
            if not complete and identity is not None:
                try:
                    current = os.lstat(target)
                except FileNotFoundError:
                    pass
                else:
                    if ordinary(current) and os.path.samestat(identity, current):
                        target.unlink(missing_ok=True)


def export_photos(photos, destination, cancelled, progress):
    if not photos or cancelled():
        return ExportResult(None, 0, (), cancelled())
    directory = Path(mkdtemp(prefix="FaceArchive Export ", dir=destination))
    copied = 0
    failures = []
    for index, photo in enumerate(photos, 1):
        if cancelled():
            break
        try:
            copied += copy_photo(photo, directory, cancelled)
        except (OSError, RuntimeError) as error:
            failures.append(f"{photo.path.name}: {error}")
        progress(index)
    if not copied:
        try:
            directory.rmdir()
        except OSError:
            pass
        else:
            directory = None
    return ExportResult(directory, copied, tuple(failures), cancelled())
