import os
import sqlite3

import numpy as np
import pytest
from PIL import Image

from facearchive.duplicates import find_duplicates
from facearchive.library import SCHEMA, Library
from facearchive.recognition import Recognition
from facearchive.scanner import scan


@pytest.fixture
def archive(tmp_path):
    library = Library(tmp_path / "data")
    source = tmp_path / "photos"
    source.mkdir()
    library.add_folder(source)
    return library, source


def write_photo(path, color="red"):
    Image.new("RGB", (400, 300), color).save(path)


def run_scan(library, engine=None, cancelled=lambda: False):
    return scan(library, cancelled, lambda *_: None, engine)


def test_incremental_index_cache_and_originals(archive):
    library, source = archive
    path = source / "photo.jpg"
    write_photo(path)
    original = path.read_bytes()
    assert run_scan(library) == (1, 1, 0)
    row = library.photos(None, 10)[0]
    old_cache = library.thumbnails / row["thumbnail"]
    with Image.open(old_cache) as thumb:
        assert thumb.size == (256, 192)
    assert run_scan(library) == (1, 0, 0)
    assert path.read_bytes() == original
    write_photo(path, "green")
    os.utime(path, ns=(row["mtime"] + 10_000, row["mtime"] + 10_000))
    assert run_scan(library) == (1, 1, 0)
    assert library.photos(None, 10)[0]["id"] == row["id"]
    assert not old_cache.exists()
    path.unlink()
    assert run_scan(library) == (0, 0, 0)
    assert not library.photos(None, 10)
    assert not list(library.thumbnails.iterdir())


def test_offline_root_and_cancelled_scan_preserve_index(archive):
    library, source = archive
    write_photo(source / "photo.jpg")
    run_scan(library)
    source.rename(source.with_name("offline"))
    assert run_scan(library) == (0, 0, 1)
    assert run_scan(library, cancelled=lambda: True) == (0, 0, 0)
    assert len(library.photos(None, 10)) == 1


def test_overlapping_root_and_bad_photo(archive):
    library, source = archive
    child = source / "nested"
    child.mkdir()
    with pytest.raises(ValueError, match="overlaps"):
        library.add_folder(child)
    (child / "broken.jpg").write_bytes(b"not an image")
    write_photo(source / "ok.png")
    assert run_scan(library) == (2, 1, 1)


class SyntheticEngine:
    version = "test-model"

    def __init__(self):
        self.calls = 0

    def detect_image(self, _image):
        self.calls += 1
        feature = np.zeros(128, dtype="<f4")
        feature[0] = 1
        return [(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32), feature)]

    store = staticmethod(Recognition.store)


def test_clustering_backfill_and_people_corrections(archive):
    library, source = archive
    for name in ("first.jpg", "second.jpg"):
        write_photo(source / name)
    assert run_scan(library) == (2, 2, 0)
    engine = SyntheticEngine()
    assert run_scan(library, engine) == (2, 0, 0)
    assert engine.calls == 2
    person = library.people()[0]["id"]
    assert library.people()[0]["total"] == 2
    assert len(library.photos(None, 10, person=person)) == 2
    with library.connect() as db:
        row = db.execute("SELECT id, x FROM faces ORDER BY id").fetchone()
        face = row["id"]
        assert row["x"] == pytest.approx(0.1)
    library.rename(person, "Taylor")
    library.correct(face, split=True)
    target = next(row["id"] for row in library.people() if row["id"] != person)
    assert {row["total"] for row in library.people()} == {1}
    library.merge(target, person)
    library.correct(face, reject=True)
    assert library.people()[0]["total"] == 1
    assert run_scan(library, engine) == (2, 0, 0)
    assert engine.calls == 2
    with library.connect() as db:
        assert db.execute("SELECT rejected FROM faces WHERE id = ?", (face,)).fetchone()[0] == 1
    library.correct(face, person)
    assert library.people()[0]["name"] == "Taylor"
    assert library.people()[0]["total"] == 2
    with library.connect() as db:
        folder = db.execute("SELECT id FROM folders").fetchone()[0]
    library.remove_folder(folder)
    assert not library.photos(None, 10)
    assert all((source / name).exists() for name in ("first.jpg", "second.jpg"))


def test_inference_failure_does_not_block_browsing(archive):
    library, source = archive
    write_photo(source / "photo.jpg")

    class FailingEngine(SyntheticEngine):
        def detect_image(self, _image):
            raise ValueError("inference failed")

    assert run_scan(library, FailingEngine()) == (1, 1, 1)
    assert library.photos(None, 10)[0]["face_version"] is None
    assert run_scan(library, SyntheticEngine()) == (1, 0, 0)
    assert len(library.people()) == 1


def test_symlinks_are_not_indexed(archive, tmp_path):
    library, source = archive
    external = tmp_path / "external"
    external.mkdir()
    write_photo(external / "outside.jpg")
    try:
        (source / "link.jpg").symlink_to(external / "outside.jpg")
        (source / "linked-folder").symlink_to(external, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable")
    assert run_scan(library) == (1, 0, 0)
    assert library.photos(None, 10) == []


def test_thumbnail_applies_exif_orientation(archive):
    library, source = archive
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (400, 300)).save(source / "rotated.jpg", exif=exif)
    assert run_scan(library) == (1, 1, 0)
    row = library.photos(None, 10)[0]
    with Image.open(library.thumbnails / row["thumbnail"]) as thumb:
        assert thumb.size == (192, 256)
        assert thumb.getexif().get(274) is None


def test_similar_faces_in_the_same_photo_are_kept_apart(archive):
    library, source = archive
    write_photo(source / "group.jpg")

    class GroupEngine(SyntheticEngine):
        def detect_image(self, image):
            face = super().detect_image(image)[0]
            return [face, ((0.6, 0.2, 0.3, 0.4), face[1])]

    assert run_scan(library, GroupEngine()) == (1, 1, 0)
    assert [person["total"] for person in library.people()] == [1, 1]


def test_recognition_and_thumbnail_share_a_single_decode(archive, monkeypatch):
    from facearchive import scanner

    library, source = archive
    write_photo(source / "photo.jpg")
    original = scanner.open_scaled
    sizes = []

    def track_decode(path, size):
        sizes.append(size)
        return original(path, size)

    monkeypatch.setattr(scanner, "open_scaled", track_decode)
    engine = SyntheticEngine()
    assert run_scan(library, engine) == (1, 1, 0)
    assert run_scan(library, engine) == (1, 0, 0)
    assert sizes == [scanner.RECOGNITION_SIZE]
    row = library.photos(None, 1)[0]
    with Image.open(library.thumbnails / row["thumbnail"]) as thumbnail:
        assert thumbnail.size == (256, 192)


def test_album_paging_is_unique_and_combines_with_folder_filter(archive, tmp_path):
    library, source = archive
    other = tmp_path / "other"
    other.mkdir()
    library.add_folder(other)
    for path in (source / "first.jpg", source / "second.jpg", other / "third.jpg"):
        write_photo(path)
    assert run_scan(library, SyntheticEngine()) == (3, 3, 0)
    person = library.people()[0]["id"]
    newest, middle, oldest = library.photos(None, 3)
    with library.connect() as db:
        db.execute(
            """
            INSERT INTO faces(photo_id, person_id, x, y, w, h, embedding)
            SELECT photo_id, person_id, x, y, w, h, embedding FROM faces WHERE photo_id = ?
        """,
            (newest["id"],),
        )
    assert [r["id"] for r in library.photos(None, 2, person=person)] == [
        newest["id"],
        middle["id"],
    ]
    assert [r["id"] for r in library.photos(middle["id"], 2, person=person)] == [oldest["id"]]
    assert [
        r["id"] for r in library.photos(None, 3, folder=oldest["folder_id"], person=person)
    ] == [middle["id"], oldest["id"]]


def test_cancelled_scan_keeps_missing_rows_until_a_complete_scan(archive):
    library, source = archive
    missing = source / "old.jpg"
    write_photo(missing)
    run_scan(library)
    missing.unlink()
    for i in range(35):
        write_photo(source / f"new-{i:02}.jpg")
    stop = False

    def cancel_after_batch(*_):
        nonlocal stop
        stop = True

    assert scan(library, lambda: stop, cancel_after_batch) == (32, 32, 0)
    assert any(row["path"] == str(missing) for row in library.photos(None, 100))
    assert run_scan(library) == (35, 3, 0)
    assert len(library.photos(None, 100)) == 35
    assert all(row["path"] != str(missing) for row in library.photos(None, 100))


def write_bmp(path, color, size=(32, 32)):
    Image.new("RGB", size, color).save(path)


def test_existing_database_migrates_without_losing_records(tmp_path):
    with sqlite3.connect(tmp_path / "library.sqlite3") as db:
        db.executescript(SCHEMA)
        db.execute("INSERT INTO folders VALUES (1, 'photos')")
        db.execute("INSERT INTO photos VALUES (1, 1, 'test.jpg', 7, 8, 'thumbnail', NULL)")
    library = Library(tmp_path)
    row = library.photos(None, 1)[0]
    assert row["path"] == "test.jpg"
    assert row["content_hash"] is None
    with library.connect() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3


def test_duplicates_filter_caches_hashes_and_invalidates_changed_copies(archive):
    library, source = archive
    for name, color in (("a.bmp", "red"), ("b.bmp", "red"), ("different.bmp", "blue")):
        write_bmp(source / name, color)
    write_bmp(source / "unique.bmp", "red", (35, 35))
    assert run_scan(library) == (4, 4, 0)
    issues = []
    args = (library, lambda: False, lambda *_: None, lambda *issue: issues.append(issue))
    assert find_duplicates(*args) == 3
    assert find_duplicates(*args) == 0
    assert not issues

    copies = library.photos(None, 20, duplicates=True)
    assert [row["path"] for row in copies] == [str(source / "b.bmp"), str(source / "a.bmp")]
    assert library.exact_copies(copies[0]["id"], 10)[0]["path"] == str(source / "a.bmp")

    (source / "b.bmp").write_bytes(b"corrupt")
    assert run_scan(library) == (4, 0, 1)
    assert not library.photos(None, 20, duplicates=True)


def test_hashing_can_cancel_before_processing_more_paths(archive):
    library, source = archive
    for i in range(42):
        write_bmp(source / f"photo-{i}.bmp", "red")
    assert run_scan(library) == (42, 42, 0)
    progress = []
    assert find_duplicates(library, lambda: bool(progress), progress.append, lambda *_: None) == 32
    assert find_duplicates(library, lambda: False, progress.append, lambda *_: None) == 10
    assert len(library.photos(None, 50, duplicates=True)) == 42


def test_timeline_uses_capture_dates_and_stable_paging(archive):
    library, source = archive
    for name, value in (
        ("a.jpg", "2019:12:31 10:00:00"),
        ("b.jpg", "2019:12:28 08:00:00"),
        ("c.jpg", "2019:12:31 08:00:00"),
    ):
        exif = Image.Exif()
        exif[36867] = value
        Image.new("RGB", (100, 100)).save(source / name, exif=exif)
    path = source / "fallback.png"
    write_photo(path)
    os.utime(path, (1_577_923_200, 1_577_923_200))

    assert run_scan(library) == (4, 4, 0)
    assert library.capture_months() == ["2020-01", "2019-12"]
    assert library.timeline("2020-01", None, 1)[0]["captured_on"] == "2020-01-02"

    first = library.timeline("2019-12", None, 1)[0]
    assert first["path"] == str(source / "c.jpg")
    remaining = library.timeline("2019-12", (first["captured_on"], first["id"]), 10)
    assert [row["path"] for row in remaining] == [str(source / "a.jpg"), str(source / "b.jpg")]


def test_date_backfill_keeps_face_assignments_and_reuses_thumbnail(archive, monkeypatch):
    from facearchive import scanner

    library, source = archive
    write_photo(source / "photo.jpg")
    engine = SyntheticEngine()
    assert run_scan(library, engine) == (1, 1, 0)
    with library.connect() as db:
        db.execute("UPDATE photos SET captured_on = NULL")

    def reject_decode(*_):
        pytest.fail("backfill should only read the image header")

    monkeypatch.setattr(scanner, "open_scaled", reject_decode)
    assert run_scan(library, engine) == (1, 0, 0)
    assert library.photos(None, 1)[0]["captured_on"] is not None
    assert library.people()[0]["total"] == 1
    assert engine.calls == 1
