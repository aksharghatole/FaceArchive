from facearchive.export import CHUNK_SIZE, ExportPhoto, export_photos


def indexed(path):
    stat = path.stat()
    return ExportPhoto(path, stat.st_mtime_ns, stat.st_size)


def test_export_suffixes_names_and_copies_bytes(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    existing = output / "same.jpg"
    existing.write_bytes(b"preserve this")
    inputs = []
    for name in ("one", "two"):
        source = tmp_path / name
        source.mkdir()
        path = source / "same.jpg"
        path.write_bytes(name.encode())
        inputs.append(indexed(path))

    result = export_photos(inputs, output, lambda: False, lambda *_: None)
    assert result.copied == 2
    assert not result.failures
    assert (result.directory / "same.jpg").read_bytes() == b"one"
    assert (result.directory / "same (2).jpg").read_bytes() == b"two"
    assert existing.read_bytes() == b"preserve this"
    assert [photo.path.read_bytes() for photo in inputs] == [b"one", b"two"]


def test_export_skips_changed_original_and_continues(tmp_path):
    changed = tmp_path / "changed.jpg"
    other = tmp_path / "other.jpg"
    changed.write_bytes(b"before")
    other.write_bytes(b"other")
    photos = [indexed(changed), indexed(other)]
    changed.write_bytes(b"changed contents")

    result = export_photos(photos, tmp_path, lambda: False, lambda *_: None)
    assert result.copied == 1
    assert "changed since indexing" in result.failures[0]
    assert sorted(path.name for path in result.directory.iterdir()) == ["other.jpg"]
    assert changed.read_bytes() == b"changed contents"


def test_export_cancel_removes_partial_copy_and_keeps_completed_photo(tmp_path):
    first, second = tmp_path / "first.jpg", tmp_path / "second.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"2" * (2 * CHUNK_SIZE))
    progress = []
    checks = 0

    def cancelled():
        nonlocal checks
        if progress:
            checks += 1
        return checks >= 3

    result = export_photos([indexed(first), indexed(second)], tmp_path, cancelled, progress.append)
    assert result.cancelled
    assert result.copied == 1
    assert sorted(path.name for path in result.directory.iterdir()) == ["first.jpg"]
    assert second.stat().st_size == 2 * CHUNK_SIZE
