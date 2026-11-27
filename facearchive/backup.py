"""Create a consistent, standalone SQLite snapshot of the local index."""

import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp


class BackupCancelled(Exception):
    pass


def backup_index(library, destination, cancelled, progress):
    if cancelled():
        return None

    last_progress = -1

    def checkpoint(_status, remaining, total):
        nonlocal last_progress
        if cancelled():
            raise BackupCancelled
        percent = (total - remaining) * 99 // total if total else 0
        if percent != last_progress:
            progress(percent)
            last_progress = percent

    with TemporaryDirectory(prefix=".facearchive-backup-", dir=destination) as temporary:
        staging = Path(temporary) / library.database.name
        source = library.database.resolve().as_uri() + "?mode=ro"
        try:
            with closing(sqlite3.connect(source, uri=True)) as current:
                with closing(sqlite3.connect(staging)) as snapshot:
                    current.backup(snapshot, pages=64, progress=checkpoint)
                    if snapshot.execute("PRAGMA journal_mode = DELETE").fetchone()[0] != "delete":
                        raise sqlite3.DatabaseError("Could not finalize a standalone backup")
                    result = snapshot.execute("PRAGMA quick_check(1)").fetchone()
                    if result is None or result[0] != "ok":
                        raise sqlite3.DatabaseError("Backup failed SQLite’s integrity check")
        except BackupCancelled:
            return None
        if cancelled():
            return None
        folder = Path(mkdtemp(prefix="FaceArchive Backup ", dir=destination))
        try:
            output = staging.replace(folder / staging.name)
        except OSError:
            folder.rmdir()
            raise
    progress(100)
    return output
