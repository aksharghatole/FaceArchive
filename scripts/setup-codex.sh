#!/usr/bin/env bash
set -euo pipefail
python --version
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-dev.txt
QT_QPA_PLATFORM=offscreen .venv/bin/python -c 'import cv2, numpy, PIL; from PySide6.QtWidgets import QApplication; app=QApplication([]); print("FaceArchive dependency smoke test OK")'
