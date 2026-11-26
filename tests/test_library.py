import os

import numpy as np
import pytest
from PIL import Image

from facearchive.library import Library
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
