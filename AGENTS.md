# FaceArchive agent instructions
Build an offline-first, native Windows photo organizer. Python 3.12, PySide6, OpenCV YuNet/SFace, SQLite, Pillow, NumPy. Use CPU-first inference; no cloud recognition, telemetry, accounts, or paid services.
- Never move, modify, or delete originals by default. Exports copy only; require explicit user action for destructive operations. Never upload photos, embeddings, metadata or paths. Keep personal data and real photo fixtures out of git, logs and CI.
- Store photo references, file identity, metadata, thumbnails and embeddings locally. Handle missing/disconnected drives without deleting records. Support incremental scans, cancellation, resume and background workers with bounded memory.
- Provide reviewable, reversible face grouping: naming, merge, split, reject incorrect matches, unknown faces, multi-person search. Do not assume model matches are ground truth.
- Prioritize a working end-to-end vertical slice: add folders, index images, view thumbnails, detect faces, create people albums and correct groups. Then duplicates, timeline, locations, smart albums, export, backup and video.
- Separate GUI, domain, storage, indexing, recognition and background workers. Use SQLite migrations, transactions and testable service boundaries.
- Use headless OpenCV wheels with PySide6 to avoid Qt conflicts. Check model and dependency licenses before redistribution; pin model versions/checksums. No runtime model downloads without explicit consent. Missing models must yield clear actionable UI.
- Use synthetic test fixtures; test EXIF rotation, corrupt files, Unicode paths, disconnected folders and cancellation. Run Ruff and pytest. Test actual Windows packaging via GitHub Actions; do not claim Windows behavior verified from Linux.
- Aim for 4 GB RAM compatibility and 100k photos; benchmark before claiming performance. Avoid quadratic full-library matching; introduce ANN indexing after benchmark and wheel compatibility checks.
- Maintain README, architecture notes, changelog and build instructions. Do not label incomplete features as working. Small coherent commits and clear PR descriptions.
