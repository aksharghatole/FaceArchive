import numpy as np
from PIL import Image
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
)

from facearchive import ui
from facearchive.library import Library
from facearchive.ui import Window, run_window
from tests.test_library import SyntheticEngine, run_scan, write_located_photo, write_photo


def finish_scan(app, window):
    window.start_scan()
    loop = QEventLoop()
    window.worker.finished.connect(loop.quit)
    QTimer.singleShot(5000, loop.quit)
    loop.exec()
    app.processEvents()
    assert window.worker is None


def test_background_scan_populates_native_browser(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    library = Library(tmp_path / "data")
    source = tmp_path / "photos"
    source.mkdir()
    write_photo(source / "sample.jpg")
    library.add_folder(source)
    window = Window(library)
    window.show()
    finish_scan(app, window)
    assert window.grid.count() == 1
    assert window.grid.item(0).text() == "sample.jpg"
    assert "1 updated" in window.statusBar().currentMessage()
    window.close()


def test_face_review_draws_box_and_splits_album(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    library = Library(tmp_path / "data")
    source = tmp_path / "photos"
    source.mkdir()
    for name in ("a.jpg", "b.jpg"):
        write_photo(source / name)
    library.add_folder(source)
    assert run_scan(library, SyntheticEngine()) == (2, 2, 0)
    window = Window(library)

    def review(dialog):
        faces = dialog.findChild(QComboBox)
        assert faces.count() == 1
        pixmap = dialog.findChild(QLabel).pixmap()
        assert not pixmap.isNull()
        border = pixmap.toImage().pixelColor(40, 60)
        assert border.blue() > 200 and border.green() > 150
        next(
            button
            for button in dialog.findChildren(QPushButton)
            if button.text() == "Split to new person"
        ).click()

    monkeypatch.setattr(QDialog, "exec", review)
    window.open_photo(window.grid.item(0))
    assert [person["total"] for person in library.people()] == [1, 1]
    window.close()
    app.processEvents()


def test_startup_initializes_library_and_releases_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QApplication, "exec", lambda *_: 0)
    assert run_window(app, tmp_path) == 0
    assert (tmp_path / "library.sqlite3").is_file()
    assert not (tmp_path / "instance.lock").exists()


def test_scan_explains_missing_models_and_bad_photo(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(ui, "BUNDLED_MODELS", tmp_path / "no-bundle")
    app = QApplication.instance() or QApplication([])
    library = Library(tmp_path / "data")
    source = tmp_path / "photos"
    source.mkdir()
    library.add_folder(source)
    write_photo(source / "valid.jpg")
    (source / "broken.jpg").write_bytes(b"not an image")

    window = Window(library)
    window.recognize.setChecked(True)
    finish_scan(app, window)
    assert window.grid.count() == 1
    assert window.grid.item(0).text() == "valid.jpg"
    assert "Face recognition unavailable" in window.statusBar().currentMessage()

    shown_reports = []

    def read_report(dialog):
        report = dialog.findChild(QPlainTextEdit)
        assert report.isReadOnly()
        shown_reports.append(report.toPlainText())

    assert window.scan_details.isEnabled()
    monkeypatch.setattr(QDialog, "exec", read_report)
    window.scan_details.click()
    assert len(shown_reports) == 1
    assert "Photo indexing" in shown_reports[0]
    assert "broken.jpg" in shown_reports[0]
    assert "face_detection_yunet_2023mar.onnx" in shown_reports[0]
    window.close()


def test_two_person_filter_intersects_albums(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    library = Library(tmp_path / "data")
    source = tmp_path / "vacaciones-夏"
    source.mkdir()
    library.add_folder(source)
    write_photo(source / "group-été.png", "red")
    write_photo(source / "solo.png", "blue")

    class TwoPeople(SyntheticEngine):
        def detect_image(self, image):
            face = super().detect_image(image)[0]
            if image.getpixel((0, 0))[0] == 0:
                return [face]
            other = np.zeros(128, dtype="<f4")
            other[1] = 1
            return [face, ((0.6, 0.2, 0.3, 0.4), other)]

    assert run_scan(library, TwoPeople()) == (2, 2, 0)
    first, second = (row["id"] for row in library.people())
    window = Window(library)
    assert not window.also_people.isEnabled()
    window.people.setCurrentIndex(window.people.findData(first))
    assert window.grid.count() == 2
    window.also_people.setCurrentIndex(window.also_people.findData(second))
    assert window.grid.count() == 1
    assert window.grid.item(0).text() == "group-été.png"
    window.people.setCurrentIndex(0)
    assert window.also_people.currentData() is None
    assert window.grid.count() == 2
    window.close()
    app.processEvents()


def test_duplicate_scan_and_gallery(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    library = Library(tmp_path / "data")
    source = tmp_path / "photos"
    source.mkdir()
    library.add_folder(source)
    write_photo(source / "first.png")
    (source / "copy.png").write_bytes((source / "first.png").read_bytes())
    (source / "different.png").write_bytes((source / "first.png").read_bytes() + b"padding")

    window = Window(library)
    window.find_copies.setChecked(True)
    finish_scan(app, window)
    assert window.grid.count() == 3
    window.only_copies.setChecked(True)
    assert {window.grid.item(i).text() for i in range(window.grid.count())} == {
        "first.png",
        "copy.png",
    }
    assert "2 files hashed" in window.statusBar().currentMessage()
    window.close()


def test_timeline_selection_and_navigation(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(ui, "PAGE_SIZE", 1)
    app = QApplication.instance() or QApplication([])
    library = Library(tmp_path / "data")
    source = tmp_path / "photos"
    source.mkdir()
    library.add_folder(source)
    for name, stamp in (
        ("a.jpg", "2023:08:21 12:00:00"),
        ("b.jpg", "2023:08:15 12:00:00"),
        ("c.jpg", "2023:09:12 12:00:00"),
    ):
        exif = Image.Exif()
        exif[36867] = stamp
        Image.new("RGB", (100, 100)).save(source / name, exif=exif)
    window = Window(library)
    finish_scan(app, window)
    window.months.setCurrentIndex(window.months.findData("2023-08"))
    assert window.grid.item(0).text() == "a.jpg"
    assert window.next_page.isEnabled()
    window.next_page.click()
    assert window.grid.item(0).text() == "b.jpg"
    assert not window.next_page.isEnabled()
    window.previous_page.click()
    assert window.grid.item(0).text() == "a.jpg"
    window.close()


def test_location_filter_and_clear_action(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    library = Library(tmp_path / "data")
    source = tmp_path / "photos"
    source.mkdir()
    library.add_folder(source)
    write_located_photo(source / "located.jpg")
    write_photo(source / "other.jpg")
    window = Window(library)
    window.index_locations.setChecked(True)
    finish_scan(app, window)
    window.location_filter.setCurrentIndex(window.location_filter.findData("-12.50, -45.33"))
    assert window.grid.count() == 1
    assert window.grid.item(0).text() == "located.jpg"
    monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.StandardButton.Yes)
    window.clear_locations_action.trigger()
    assert window.location_filter.count() == 1
    assert not window.index_locations.isChecked()
    assert window.grid.count() == 2
    window.close()
