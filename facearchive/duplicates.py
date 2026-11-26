"""Find exact duplicates with size prefiltering and cancellable streaming hashes."""

import hashlib
import os
from pathlib import Path

from facearchive.scanner import open_original, ordinary

CHUNK_SIZE = 1024 * 1024
BATCH_SIZE = 32


def digest_photo(path, mtime, size, cancelled):
    if path.resolve(strict=True) != path:
        raise OSError("Photo path now resolves to a different location")
    with open_original(path) as stream:
        initial = os.fstat(stream.fileno())
        if (initial.st_mtime_ns, initial.st_size) != (mtime, size):
            raise OSError("Photo changed since indexing")
        digest = hashlib.sha256()
        while not cancelled():
            chunk = stream.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
        if cancelled():
            return None
        current = os.lstat(path)
        if (
            not ordinary(current)
            or not os.path.samestat(initial, current)
            or (current.st_mtime_ns, current.st_size) != (mtime, size)
        ):
            raise OSError("Photo changed while hashing")
        return digest.digest()


def find_duplicates(library, cancelled, progress, report_issue):
    hashed = 0
    with library.connect() as db:
        db.execute("""
            CREATE TEMP TABLE duplicate_sizes(size INTEGER PRIMARY KEY) WITHOUT ROWID
        """)
        db.execute("""
            INSERT INTO duplicate_sizes
            SELECT size FROM photos GROUP BY size HAVING count(*) > 1
        """)
        db.commit()
        after = 0
        while not cancelled():
            rows = db.execute(
                """
                SELECT p.id, p.path, p.size, p.mtime FROM photos p
                WHERE p.id > ? AND p.content_hash IS NULL
                AND EXISTS (SELECT 1 FROM duplicate_sizes s WHERE s.size = p.size)
                ORDER BY p.id LIMIT ?
            """,
                (after, BATCH_SIZE),
            ).fetchall()
            if not rows:
                break
            updates = []
            for row in rows:
                if cancelled():
                    break
                path = Path(row["path"])
                try:
                    digest = digest_photo(path, row["mtime"], row["size"], cancelled)
                except OSError as error:
                    report_issue("Exact duplicate search", path, error)
                    continue
                if digest is not None:
                    updates.append((digest, row["id"], row["path"], row["mtime"], row["size"]))
            with db:
                db.executemany(
                    """
                    UPDATE photos SET content_hash = ?
                    WHERE id = ? AND path = ? AND mtime = ? AND size = ?
                """,
                    updates,
                )
            hashed += len(updates)
            progress(hashed)
            after = rows[-1]["id"]
    return hashed
