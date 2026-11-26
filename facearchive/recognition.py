"""Offline YuNet and SFace inference via OpenCV's model APIs."""

import hashlib

import cv2
import numpy as np

from facearchive.scanner import RECOGNITION_SIZE, open_scaled

DETECTOR = "face_detection_yunet_2023mar.onnx"
RECOGNIZER = "face_recognition_sface_2021dec.onnx"
MATCH_THRESHOLD = 0.50


class Recognition:
    def __init__(self, models):
        detector = models / DETECTOR
        recognizer = models / RECOGNIZER
        if not detector.is_file() or not recognizer.is_file():
            raise ValueError(f"Install {DETECTOR} and {RECOGNIZER} in {models}")
        hashes = []
        for model in (detector, recognizer):
            with model.open("rb") as stream:
                hashes.append(hashlib.file_digest(stream, "sha256").hexdigest())
        self.version = "1:" + ":".join(hashes)
        cv2.setNumThreads(1)
        try:
            self.detector = cv2.FaceDetectorYN.create(str(detector), "", (320, 320), 0.85)
            self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer), "")
        except cv2.error as error:
            raise ValueError("OpenCV could not load the configured ONNX models") from error

    def detect(self, path):
        return self.detect_image(open_scaled(path, RECOGNITION_SIZE))

    def detect_image(self, image):
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        try:
            _, detections = self.detector.detect(image)
            result = []
            for face in detections if detections is not None else ():
                crop = self.recognizer.alignCrop(image, face)
                feature = self.recognizer.feature(crop).reshape(-1).astype("<f4")
                norm = np.linalg.norm(feature)
                if feature.size == 128 and np.isfinite(norm) and norm > 0:
                    x, y, w, h = map(float, face[:4])
                    left, top = max(0, x / width), max(0, y / height)
                    right, bottom = min(1, (x + w) / width), min(1, (y + h) / height)
                    if right > left and bottom > top:
                        box = (left, top, right - left, bottom - top)
                        result.append((box, feature / norm))
            return result
        except cv2.error as error:
            raise ValueError("OpenCV failed to process the photo") from error

    @staticmethod
    def store(db, photo, faces):
        used = set()
        for box, feature in faces:
            person = None
            best = MATCH_THRESHOLD
            candidates = db.execute(
                """
                SELECT p.id, (
                    SELECT f.embedding FROM faces f JOIN photos ph ON ph.id = f.photo_id
                    WHERE f.person_id = p.id AND ph.face_version =
                        (SELECT face_version FROM photos WHERE id = ?)
                    ORDER BY f.id LIMIT 1
                ) AS prototype FROM people p ORDER BY p.id
            """,
                (photo,),
            )
            while rows := candidates.fetchmany(256):
                rows = [
                    row for row in rows if row["id"] not in used and row["prototype"] is not None
                ]
                if not rows:
                    continue
                vectors = np.frombuffer(b"".join(row["prototype"] for row in rows), dtype="<f4")
                scores = np.einsum("ij,j->i", vectors.reshape(-1, 128), feature)
                index = int(scores.argmax())
                if scores[index] >= best:
                    best, person = float(scores[index]), rows[index]["id"]
            if person is None:
                person = db.execute("INSERT INTO people DEFAULT VALUES").lastrowid
            used.add(person)
            db.execute(
                """
                INSERT INTO faces(photo_id, person_id, x, y, w, h, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (photo, person, *map(float, box), feature.tobytes()),
            )
