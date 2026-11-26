"""Native library browser. Scanning and inference run off the GUI thread."""

import sqlite3
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

from PIL import Image
from PySide6.QtCore import QLockFile, QRectF, QSize, QStandardPaths, Qt, QThread, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from facearchive.library import Library
from facearchive.scanner import open_scaled, scan

PAGE_SIZE = 160
SCAN_ISSUE_LIMIT = 100
ID_ROLE = Qt.ItemDataRole.UserRole


BUNDLED_MODELS = Path(__file__).resolve().parent / "models"


class ScanThread(QThread):
    update = Signal(int, int, int)
    result = Signal(str, object, int)

    def __init__(self, library, recognize, parent):
        super().__init__(parent)
        self.library = library
        self.recognize = recognize
        self.cancelled = Event()

    def run(self):
        issues = []
        issue_count = 0

        def report_issue(stage, path, error):
            nonlocal issue_count
            issue_count += 1
            if len(issues) < SCAN_ISSUE_LIMIT:
                location = f" · {path}" if path else ""
                issues.append(f"{stage}{location}\n{error}")

        try:
            engine = None
            if self.recognize:
                models = self.library.directory / "models"
                models = BUNDLED_MODELS if BUNDLED_MODELS.is_dir() else models
                try:
                    from facearchive.recognition import Recognition

                    engine = Recognition(models)
                except (ImportError, OSError, ValueError) as error:
                    report_issue("Face recognition unavailable", models, error)
            visited, indexed, errors = scan(
                self.library,
                self.cancelled.is_set,
                self.update.emit,
                engine,
                report_issue=report_issue,
            )
            status = "Stopped" if self.cancelled.is_set() else "Scan finished"
            message = f"{status} · {visited} visited · {indexed} updated · {errors} file errors"
            if self.recognize and engine is None:
                message += " · Face recognition unavailable"
        except Exception as error:
            message = "Scan failed"
            report_issue("Scan stopped unexpectedly", None, error)
        self.result.emit(message, issues, issue_count)


class Window(QMainWindow):
    def __init__(self, library):
        super().__init__()
        self.library = library
        self.worker = None
        self.scan_report = ""
        self.pages = [None]
        self.next_cursor = None
        self.setWindowTitle("FaceArchive")
        self.scan_details = QPushButton("Scan details")
        self.scan_details.clicked.connect(self.show_scan_details)
        self.scan_details.setEnabled(False)
        self.statusBar().addPermanentWidget(self.scan_details)
        self.resize(1050, 720)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        toolbar = QHBoxLayout()
        layout.addLayout(toolbar)
        self.add = self.button(toolbar, "Add folder", self.add_folder)
        self.remove = self.button(toolbar, "Remove folder", self.remove_folder)
        self.scan_button = self.button(toolbar, "Scan", self.start_scan)
        self.stop = self.button(toolbar, "Stop", self.cancel_scan)
        self.stop.setEnabled(False)
        self.recognize = QCheckBox("Recognize faces")
        toolbar.addWidget(self.recognize)
        self.button(toolbar, "Model folder", self.show_models)
        filters = QHBoxLayout()
        layout.addLayout(filters)
        self.folders = QComboBox()
        self.people = QComboBox()
        filters.addWidget(self.folders, 1)
        filters.addWidget(self.people, 1)
        self.rename_button = self.button(filters, "Name person", self.rename_person)
        self.merge_button = self.button(filters, "Merge person", self.merge_person)
        self.grid = QListWidget()
        self.grid.setViewMode(QListView.ViewMode.IconMode)
        self.grid.setResizeMode(QListView.ResizeMode.Adjust)
        self.grid.setMovement(QListView.Movement.Static)
        self.grid.setIconSize(QSize(150, 130))
        self.grid.setGridSize(QSize(168, 160))
        self.grid.itemActivated.connect(self.open_photo)
        layout.addWidget(self.grid, 1)
        paging = QHBoxLayout()
        layout.addLayout(paging)
        self.button(paging, "First page", self.refresh_grid)
        self.previous_page = self.button(paging, "Previous page", self.go_back)
        self.next_page = self.button(paging, "Next page", self.go_forward)
        self.folders.currentIndexChanged.connect(self.refresh_grid)
        self.people.currentIndexChanged.connect(self.refresh_grid)
        self.refresh_filters()

    @staticmethod
    def button(layout, label, slot):
        button = QPushButton(label)
        button.clicked.connect(slot)
        layout.addWidget(button)
        return button

    def refresh_filters(self):
        with self.library.connect() as db:
            folders = db.execute("SELECT * FROM folders ORDER BY path").fetchall()
        people = self.library.people()
        for combo, label, items in (
            (self.folders, "All folders", [(row["path"], row["id"]) for row in folders]),
            (
                self.people,
                "All people",
                [
                    (f"{row['name'] or 'Person ' + str(row['id'])} ({row['total']})", row["id"])
                    for row in people
                ],
            ),
        ):
            selected = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(label, None)
            for title, value in items:
                combo.addItem(title, value)
            combo.setCurrentIndex(max(0, combo.findData(selected)))
            combo.blockSignals(False)
        self.refresh_grid()

    def refresh_grid(self):
        self.pages = [None]
        self.load_page()

    def go_back(self):
        if len(self.pages) > 1:
            self.pages.pop()
            self.load_page()

    def go_forward(self):
        if self.next_cursor is not None:
            self.pages.append(self.next_cursor)
            self.load_page()

    def load_page(self):
        folder, person = self.folders.currentData(), self.people.currentData()
        self.remove.setEnabled(folder is not None and self.worker is None)
        self.rename_button.setEnabled(person is not None and self.worker is None)
        self.merge_button.setEnabled(person is not None and self.worker is None)
        photos = self.library.photos(self.pages[-1], PAGE_SIZE + 1, folder, person)
        self.grid.clear()
        for photo in photos[:PAGE_SIZE]:
            path = Path(photo["path"])
            item = QListWidgetItem(
                QIcon(str(self.library.thumbnails / photo["thumbnail"])), path.name
            )
            item.setToolTip(str(path))
            item.setData(ID_ROLE, (photo["id"], str(path)))
            self.grid.addItem(item)
        self.next_cursor = photos[PAGE_SIZE - 1]["id"] if len(photos) > PAGE_SIZE else None
        self.previous_page.setEnabled(len(self.pages) > 1)
        self.next_page.setEnabled(self.next_cursor is not None)

    def add_folder(self):
        chosen = QFileDialog.getExistingDirectory(self, "Add photo folder")
        if chosen:
            try:
                self.library.add_folder(Path(chosen))
            except (OSError, ValueError) as error:
                QMessageBox.warning(self, "Cannot add folder", str(error))
                return
            self.refresh_filters()
            self.start_scan()

    def remove_folder(self):
        folder = self.folders.currentData()
        answer = QMessageBox.question(self, "Remove folder", "Remove this folder from the index?")
        if folder is not None and answer == QMessageBox.StandardButton.Yes:
            self.library.remove_folder(folder)
            self.refresh_filters()

    def show_models(self):
        models = self.library.directory / "models"
        models.mkdir(exist_ok=True)
        QMessageBox.information(
            self,
            "Model folder",
            f"{models}\n\nSource installations can copy the README’s ONNX files here.",
        )

    def start_scan(self):
        if self.worker is not None:
            return
        self.worker = ScanThread(self.library, self.recognize.isChecked(), self)
        self.worker.update.connect(self.show_progress)
        self.worker.result.connect(self.show_result)
        self.worker.finished.connect(self.scan_finished)
        self.stop.setEnabled(True)
        self.add.setEnabled(False)
        self.scan_button.setEnabled(False)
        self.scan_details.setEnabled(False)
        self.statusBar().showMessage("Scanning…")
        self.refresh_grid()
        self.worker.start()

    def show_result(self, message, issues, issue_count):
        self.statusBar().showMessage(message)
        sections = [message, *issues]
        if issue_count > len(issues):
            sections.append(f"{issue_count - len(issues)} additional issues omitted.")
        self.scan_report = "\n\n".join(sections)
        self.scan_details.setEnabled(True)

    def show_scan_details(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Scan details")
        dialog.resize(720, 460)
        layout = QVBoxLayout(dialog)
        report = QPlainTextEdit(self.scan_report)
        report.setReadOnly(True)
        layout.addWidget(report)
        self.button(layout, "Close", dialog.accept)
        dialog.exec()

    def show_progress(self, visited, indexed, errors):
        self.statusBar().showMessage(f"{visited} visited · {indexed} updated · {errors} errors")

    def cancel_scan(self):
        if self.worker is not None:
            self.worker.cancelled.set()

    def scan_finished(self):
        self.worker.deleteLater()
        self.worker = None
        self.stop.setEnabled(False)
        self.add.setEnabled(True)
        self.scan_button.setEnabled(True)
        self.refresh_filters()

    def rename_person(self):
        name, accepted = QInputDialog.getText(self, "Name person", "Name")
        if accepted:
            self.library.rename(self.people.currentData(), name)
            self.refresh_filters()

    def choose_person(self, title, exclude=None):
        choices = {
            f"{row['name'] or 'Unnamed'} [#{row['id']}]": row["id"]
            for row in self.library.people()
            if row["id"] != exclude
        }
        if not choices:
            QMessageBox.information(self, title, "There are no destination people.")
            return None
        choice, accepted = QInputDialog.getItem(
            self, title, "Destination", list(choices), editable=False
        )
        return choices[choice] if accepted else None

    def merge_person(self):
        source = self.people.currentData()
        target = self.choose_person("Merge into person", source)
        if target is not None:
            self.library.merge(source, target)
            self.refresh_filters()

    def open_photo(self, item):
        photo, path = item.data(ID_ROLE)
        dialog = QDialog(self)
        dialog.setWindowTitle(Path(path).name)
        layout = QVBoxLayout(dialog)
        preview = QLabel()
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        screen = self.screen().availableGeometry()
        size = max(160, min(1000, screen.width() - 100, screen.height() - 260))
        original = None
        try:
            image = open_scaled(path, size)
            from PIL.ImageQt import ImageQt

            original = QPixmap.fromImage(ImageQt(image))
            preview.setPixmap(original)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            preview.setText(f"Cannot open photo · {error}")
        layout.addWidget(preview)
        faces = QComboBox()
        layout.addWidget(faces)
        actions = QHBoxLayout()
        layout.addLayout(actions)

        def highlight():
            if original is None:
                return
            box = faces.currentData(ID_ROLE + 1)
            canvas = original.copy()
            if box is not None:
                x, y, w, h = box
                with QPainter(canvas) as painter:
                    painter.setPen(QPen(QColor("#00d4ff"), 3))
                    painter.drawRect(
                        QRectF(
                            x * canvas.width(),
                            y * canvas.height(),
                            w * canvas.width(),
                            h * canvas.height(),
                        )
                    )
            preview.setPixmap(canvas)

        def reload_faces():
            faces.clear()
            with self.library.connect() as db:
                rows = db.execute(
                    """
                    SELECT f.*, p.name FROM faces f LEFT JOIN people p ON p.id = f.person_id
                    WHERE f.photo_id = ? ORDER BY f.x, f.y
                """,
                    (photo,),
                ).fetchall()
            for index, row in enumerate(rows, 1):
                label = row["name"] or (
                    f"Person {row['person_id']}" if row["person_id"] else "Unassigned"
                )
                if row["rejected"]:
                    label = "Dismissed"
                faces.addItem(
                    f"Face {index} · {label} · ({row['x']:.0%}, {row['y']:.0%})", row["id"]
                )
                faces.setItemData(
                    index - 1, tuple(row[k] for k in ("x", "y", "w", "h")), ID_ROLE + 1
                )
            highlight()

        def edit(action):
            face = faces.currentData()
            if face is None or self.worker is not None:
                return
            if action == "move":
                target = self.choose_person("Assign face")
                if target is None:
                    return
                self.library.correct(face, target)
            else:
                self.library.correct(face, split=action == "split", reject=action == "dismiss")
            reload_faces()
            self.refresh_filters()

        faces.currentIndexChanged.connect(highlight)
        reload_faces()
        for label, action in (
            ("Assign", "move"),
            ("Split to new person", "split"),
            ("Dismiss", "dismiss"),
            ("Unassign", "unassign"),
        ):
            button = self.button(actions, label, lambda checked=False, action=action: edit(action))
            button.setEnabled(self.worker is None and faces.count() > 0)
        dialog.exec()

    def closeEvent(self, event):
        if self.worker is not None:
            self.cancel_scan()
            self.statusBar().showMessage("Stopping scan. Close the window when it finishes.")
            event.ignore()
        else:
            event.accept()


def run_window(app, directory, check_bundle=False):
    try:
        directory.mkdir(parents=True, exist_ok=True)
        lock = QLockFile(str(directory / "instance.lock"))
        lock.setStaleLockTime(0)
        if not lock.tryLock(0):
            if not check_bundle:
                QMessageBox.warning(None, "FaceArchive", "The library is already open or locked.")
            return 1
        window = Window(Library(directory))
    except (OSError, RuntimeError, sqlite3.Error) as error:
        if not check_bundle:
            QMessageBox.critical(None, "Cannot open library", str(error))
        return 1
    window.show()
    if check_bundle:
        from facearchive.recognition import Recognition

        app.processEvents()
        window.close()
        try:
            Recognition(BUNDLED_MODELS).check_models()
        except (OSError, ValueError, RuntimeError):
            return 1
        return 0
    return app.exec()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FaceArchive")
    if sys.argv[1:] == ["--check-bundle"]:
        with TemporaryDirectory(prefix="facearchive-check-") as temporary:
            return run_window(app, Path(temporary), check_bundle=True)
    directory = Path(
        QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    )
    return run_window(app, directory)


if __name__ == "__main__":
    sys.exit(main())
