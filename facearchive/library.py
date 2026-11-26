"""Local library storage. Each thread owns its own SQLite connection."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS folders (
    id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL
);
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY, folder_id INTEGER NOT NULL REFERENCES folders ON DELETE CASCADE,
    path TEXT UNIQUE NOT NULL, mtime INTEGER NOT NULL, size INTEGER NOT NULL,
    thumbnail TEXT NOT NULL, face_version TEXT
);
CREATE INDEX IF NOT EXISTS photos_folder ON photos(folder_id);
CREATE INDEX IF NOT EXISTS photos_thumbnail ON photos(thumbnail);
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY, name TEXT
);
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY, photo_id INTEGER NOT NULL REFERENCES photos ON DELETE CASCADE,
    person_id INTEGER REFERENCES people ON DELETE SET NULL,
    x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
    embedding BLOB NOT NULL, rejected INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS faces_photo ON faces(photo_id);
CREATE INDEX IF NOT EXISTS faces_person ON faces(person_id, photo_id);
PRAGMA user_version = 1;
"""

DUPLICATE_MIGRATION = """
ALTER TABLE photos ADD COLUMN content_hash BLOB;
CREATE INDEX photos_size ON photos(size);
CREATE INDEX photos_content_hash ON photos(content_hash, size) WHERE content_hash IS NOT NULL;
PRAGMA user_version = 2;
"""

TIMELINE_MIGRATION = """
ALTER TABLE photos ADD COLUMN captured_on TEXT;
CREATE INDEX photos_captured ON photos(captured_on, id) WHERE captured_on IS NOT NULL;
PRAGMA user_version = 3;
"""

DUPLICATE_FILTER = """
p.content_hash IS NOT NULL AND EXISTS (
    SELECT 1 FROM photos copy WHERE copy.content_hash = p.content_hash
    AND copy.size = p.size AND copy.id != p.id
)
"""


class Library:
    def __init__(self, directory: Path):
        self.directory = directory
        self.thumbnails = directory / "thumbnails"
        self.thumbnails.mkdir(parents=True, exist_ok=True)
        self.database = directory / "library.sqlite3"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2, 3):
                raise RuntimeError(f"Unsupported library version {version}")
            migrations = (SCHEMA, DUPLICATE_MIGRATION, TIMELINE_MIGRATION)[version:]
            if migrations:
                db.executescript("BEGIN;\n" + "\n".join(migrations) + "\nCOMMIT;")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=15)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            with db:
                yield db
        finally:
            db.close()

    def add_folder(self, path: Path):
        path = path.resolve(strict=True)
        if not path.is_dir():
            raise ValueError("Choose a directory.")
        data = self.directory.resolve()
        if data.is_relative_to(path) or path.is_relative_to(data):
            raise ValueError("Choose a folder outside FaceArchive’s own data tree.")
        with self.connect() as db:
            for row in db.execute("SELECT path FROM folders"):
                existing = Path(row["path"])
                if path.is_relative_to(existing) or existing.is_relative_to(path):
                    raise ValueError("This folder overlaps a folder already in the library.")
            db.execute("INSERT INTO folders(path) VALUES (?)", (str(path),))

    def remove_folder(self, folder_id):
        with self.connect() as db:
            db.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
            self.prune_people(db)

    @staticmethod
    def prune_people(db):
        db.execute("""
            DELETE FROM people WHERE name IS NULL
            AND NOT EXISTS (SELECT 1 FROM faces WHERE person_id = people.id)
        """)

    def photos(
        self, before, limit, folder=None, person=None, *, also_person=None, duplicates=False
    ):
        if person is None:
            person, also_person = also_person, None
        if person is None:
            conditions = ["p.id < ?"] if before is not None else []
            args = [before] if before is not None else []
            if folder is not None:
                conditions.append("p.folder_id = ?")
                args.append(folder)
            if duplicates:
                conditions.append(DUPLICATE_FILTER)
            where = " WHERE " + " AND ".join(conditions) if conditions else ""
            query = "SELECT p.* FROM photos p" + where + " ORDER BY p.id DESC LIMIT ?"
        else:
            conditions = ["f.person_id = ?"]
            args = [person]
            if before is not None:
                conditions.append("f.photo_id < ?")
                args.append(before)
            photo_conditions = ["p.id = f.photo_id"]
            if folder is not None:
                photo_conditions.append("p.folder_id = ?")
                args.append(folder)
            if duplicates:
                photo_conditions.append(DUPLICATE_FILTER)
            if folder is not None or duplicates:
                conditions.append(
                    "EXISTS (SELECT 1 FROM photos p WHERE " + " AND ".join(photo_conditions) + ")"
                )
            if also_person is not None and also_person != person:
                conditions.append(
                    "EXISTS (SELECT 1 FROM faces other "
                    "WHERE other.person_id = ? AND other.photo_id = f.photo_id)"
                )
                args.append(also_person)
            query = (
                "SELECT p.* FROM (SELECT DISTINCT f.photo_id FROM faces f WHERE "
                + " AND ".join(conditions)
                + " ORDER BY f.photo_id DESC LIMIT ?) album "
                "JOIN photos p ON p.id = album.photo_id ORDER BY p.id DESC"
            )
        with self.connect() as db:
            return db.execute(query, (*args, limit)).fetchall()

    def capture_months(self):
        with self.connect() as db:
            rows = db.execute("""
                SELECT DISTINCT substr(captured_on, 1, 7) AS month
                FROM photos WHERE captured_on IS NOT NULL ORDER BY month DESC
            """)
            return [row["month"] for row in rows]

    def timeline(self, month, before, limit, folder=None, people=(), *, duplicates=False):
        conditions = ["p.captured_on >= ?", "p.captured_on < ?"]
        args = [month + "-01", month + "-32"]
        if before is not None:
            conditions.append("(p.captured_on, p.id) < (?, ?)")
            args.extend(before)
        if folder is not None:
            conditions.append("p.folder_id = ?")
            args.append(folder)
        for person in people:
            conditions.append("""
                EXISTS (SELECT 1 FROM faces f WHERE f.person_id = ? AND f.photo_id = p.id)
            """)
            args.append(person)
        if duplicates:
            conditions.append(DUPLICATE_FILTER)
        query = (
            "SELECT p.* FROM photos p WHERE "
            + " AND ".join(conditions)
            + " ORDER BY p.captured_on DESC, p.id DESC LIMIT ?"
        )
        with self.connect() as db:
            return db.execute(query, (*args, limit)).fetchall()

    def exact_copies(self, photo, limit):
        with self.connect() as db:
            return db.execute(
                """
                SELECT copy.path FROM photos p JOIN photos copy
                ON copy.content_hash = p.content_hash AND copy.size = p.size
                WHERE p.id = ? AND copy.id != p.id
                ORDER BY copy.id LIMIT ?
            """,
                (photo, limit),
            ).fetchall()

    def people(self):
        with self.connect() as db:
            return db.execute("""
                SELECT p.id, p.name, count(DISTINCT f.photo_id) AS total FROM people p
                LEFT JOIN faces f ON f.person_id = p.id GROUP BY p.id
                ORDER BY p.name IS NULL, p.name COLLATE NOCASE, p.id
            """).fetchall()

    def rename(self, person, name):
        with self.connect() as db:
            db.execute("UPDATE people SET name = ? WHERE id = ?", (name.strip() or None, person))

    def merge(self, source, target):
        if source == target:
            return
        with self.connect() as db:
            db.execute(
                """
                UPDATE people SET name = coalesce(name, (SELECT name FROM people WHERE id = ?))
                WHERE id = ?
            """,
                (source, target),
            )
            db.execute("UPDATE faces SET person_id = ? WHERE person_id = ?", (target, source))
            db.execute("DELETE FROM people WHERE id = ?", (source,))

    def correct(self, face, person=None, *, split=False, reject=False):
        with self.connect() as db:
            if split:
                row = db.execute("SELECT id FROM faces WHERE id = ?", (face,)).fetchone()
                if row is None:
                    return
                person = db.execute("INSERT INTO people DEFAULT VALUES").lastrowid
            db.execute(
                "UPDATE faces SET person_id = ?, rejected = ? WHERE id = ?", (person, reject, face)
            )
            self.prune_people(db)
