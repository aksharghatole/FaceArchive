# FaceArchive

FaceArchive is an offline Windows desktop photo organizer. It indexes existing folders, caches thumbnails, and can group detected faces into people albums using local OpenCV models. Face recognition is opt-in. The desktop app makes no network requests.

## Run from source

Use Python 3.12 on Windows. From PowerShell in the repository root:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m facearchive
```

1. Choose **Add folder** to index a folder and its subfolders. Use **Scan** after adding, changing, or deleting files on disk. Open **Scan details** to inspect file and model errors from the most recent scan. If model initialization fails, the scan still indexes photos and reports that recognition was unavailable.
2. Filter by folder or person and use **Next page** to browse. With a person selected, use **Also contains…** to find photos that include a second person. Double-click a thumbnail to view the original at screen size.
3. For face albums, open **Model folder** to find the local data path. Copy in both ONNX files described below, enable **Recognize faces**, and select **Scan**. This also analyzes photos already in the index.
4. Select a person to name or merge their album. Open a photo and select a face to highlight it. Assign it to another album, split it into a new album, unassign it, or dismiss it as a false detection.

## Exact duplicates

Enable **Find exact duplicates** before scanning. FaceArchive hashes indexed files
that share a file size and reuses hashes on later scans. Enable **Show duplicates
only** to narrow the gallery, then open a photo and choose **Show exact copies**
to see the other indexed paths. Files with visually similar content but different
bytes are outside this search. Results reflect the last scan, so rescan after
external changes. Originals are read-only.

## Models

The executable built by CI includes both models and their license files. For a source installation, download them on a connected development machine and copy the output to the model folder. The fetch script checks the SHA-256 digest of each ONNX file. The model folder can also be populated from offline media.

```powershell
.venv\Scripts\python scripts/fetch_models.py build/models
```

- YuNet uses `face_detection_yunet_2023mar.onnx` under the MIT license. [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- SFace uses `face_recognition_sface_2021dec.onnx` under Apache 2.0. [OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)

Scan details stay in memory for the current session and show up to 100 issues per scan. The app stores local face embeddings, face boxes, names, source paths and thumbnails alongside its SQLite database in the Qt application-local data directory. Treat that directory and its backups as sensitive. Close the app before copying the directory for backup. Originals are opened read-only. Removing a folder removes its indexed photos and faces, and the next completed scan clears unused thumbnails. Named, empty albums are retained.

## Architecture and limits

- **Library** uses SQLite WAL, foreign keys, and versioned schema initialization. Keyset pagination keeps gallery memory bounded. People albums page through the indexed face assignments and show each photo once.
- **Scanning** runs on one background worker. It compares modification timestamp and file size, generates thumbnails with atomic replacement, and checks for cancellation between files. When face recognition and thumbnail generation are both needed, they share one image decode. An inaccessible root or incomplete traversal does not trigger missing-file cleanup for that folder. Symlinks and Windows reparse points encountered under registered roots are skipped.
- **Recognition** uses YuNet at up to 1280 pixels on the long edge and normalized SFace embeddings. New detections join the matching album whose representative passes a cosine threshold, or create a new album. Matching compares album representatives in bounded batches. Its runtime grows with the number of albums. Grouping is heuristic, so review matches before relying on them. Small or obscured faces may be missed.
- Corrections persist for unchanged photos using the same model version. Replacing an image or switching to a different model version can replace its face records and require reviewing it again. Corrections cover existing detections.
- Supported formats include JPEG, PNG, WebP, BMP, TIFF and GIF. Animated and multi-page files use the first frame. Images larger than 48 megapixels are skipped. The modification timestamp and size check can miss an edit that preserves both.

## Development and packaging

```powershell
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m ruff format --check .
.venv\Scripts\python -m pytest -ra
```

The real-model smoke test exercises both models with generated images. To run it locally, set `FACEARCHIVE_TEST_MODELS` to the directory produced by the fetch script. Without that setting, pytest skips the model smoke test.

The Windows GitHub Actions workflow installs dependencies, runs lint and tests with the checked ONNX models, and builds a PyInstaller one-folder distribution. Extract the complete **FaceArchive-Windows-x64** artifact before opening `FaceArchive.exe`. The executable is unsigned.

See the [architecture notes](docs/architecture.md) and [changelog](CHANGELOG.md).
