# Architecture

The PySide6 window owns navigation, review controls and scan status. It launches a
single background scan worker and receives progress and results through Qt signals.
The GUI paginates the SQLite gallery and requests thumbnails from the local cache.
A Qt lock file prevents multiple instances from editing the same library.

`Library` owns the SQLite schema, connection lifetime, browsing queries and manual
face corrections. Connections belong to the thread that opens them. Foreign keys
cascade when folders or photos leave the index. WAL allows photo browsing while
the worker commits a scan batch. Person album queries start with the indexed face
assignments and use an additional indexed membership check for two-person searches.

The scanner reads originals without writing to them. It checks timestamps and
sizes, records which paths it sees, and commits progress in batches. Only a complete
folder traversal removes entries for absent files. Thumbnails are written by atomic
replacement and cleaned separately from original files. Image decoding has a pixel
limit and checks file identity when opening. Scan diagnostics stay in memory.

Exact-duplicate scanning groups indexed file sizes in SQLite and hashes candidates
in cancellable chunks. Unchanged paths reuse stored SHA-256 values. The scanner
clears a stored hash when the indexed file's timestamp or size changes. Duplicate
queries use a partial index over known hashes and file sizes.

The timeline stores a calendar date and indexes it with the photo ID for stable
keyset pagination. Capture dates are extracted during indexing and backfilled
from image headers for older records.

Location indexing is opt-in. The scanner extracts only a rounded latitude and
longitude pair from valid GPS EXIF, and remembers when an unchanged file has
already been checked. Changed files invalidate that state. The library can
clear these areas without opening originals.

Smart albums store gallery filter parameters in SQLite. Opening an album runs
the existing gallery queries against the current library. Folder removal
cascades to its albums. Person merges move references to the destination
person. Clearing indexed locations also removes saved location filters.

Exports run on a separate Qt worker and reuse the scanner's read-only source
opening and identity checks. Each request creates its own destination folder.
Copies use exclusive creation, are streamed in bounded chunks and are checked
against the indexed file metadata. Cancellation removes an incomplete copy.

Index backups use SQLite’s online backup API to capture committed data,
including uncheckpointed WAL entries. The worker checks the standalone copy
with SQLite before moving it out of staging. Originals, models and the
thumbnail cache are separate from the snapshot.

Recognition runs in the scan worker and is opt-in. YuNet detects faces and landmarks.
SFace aligns crops and computes normalized embeddings. Matching compares each new
embedding with a representative from each compatible album in bounded batches.
Face assignments and photos are updated transactionally. Review and correct matches
in the app. A replacement image or model version can require new review.

The Windows workflow builds an unsigned one-folder executable with PyInstaller.
It verifies the official model checksums at build time and smoke-tests inference
with generated images. The application loads the packaged models or local models
and has no runtime network client.
