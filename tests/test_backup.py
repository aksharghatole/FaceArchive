import sqlite3
from contextlib import closing

import pytest

from facearchive.backup import backup_index
from facearchive.library import Library


def test_backup_is_standalone_and_reads_committed_wal(tmp_path):
    library = Library(tmp_path / "library")
    destination = tmp_path / "backups"
    destination.mkdir()
    progress = []
    with library.connect() as writer:
        writer.execute("INSERT INTO smart_albums(name, duplicates) VALUES ('Saved', 1)")
        writer.commit()
        writer.execute("INSERT INTO smart_albums(name, duplicates) VALUES ('Pending', 0)")
        assert library.database.with_name("library.sqlite3-wal").is_file()
        backup = backup_index(library, destination, lambda: False, progress.append)
        writer.rollback()

    assert backup.name == "library.sqlite3"
    assert list(backup.parent.iterdir()) == [backup]
    assert progress[-1] == 100
    with closing(sqlite3.connect(backup)) as snapshot:
        assert snapshot.execute("PRAGMA journal_mode").fetchone() == ("delete",)
        assert snapshot.execute("SELECT name, duplicates FROM smart_albums").fetchall() == [
            ("Saved", 1),
        ]
    assert Library(backup.parent).smart_albums()[0]["name"] == "Saved"
    assert library.smart_albums()[0]["name"] == "Saved"


def test_backup_cancellation_leaves_destination_empty(tmp_path):
    library = Library(tmp_path / "library")
    destination = tmp_path / "backups"
    destination.mkdir()
    seen = []
    assert backup_index(library, destination, lambda: bool(seen), seen.append) is None
    assert seen
    assert not list(destination.iterdir())


def test_corrupt_source_does_not_publish_a_backup(tmp_path):
    library = Library(tmp_path / "library")
    library.database.write_bytes(b"not a database")
    destination = tmp_path / "backups"
    destination.mkdir()
    with pytest.raises(sqlite3.DatabaseError):
        backup_index(library, destination, lambda: False, lambda *_: None)
    assert not list(destination.iterdir())
