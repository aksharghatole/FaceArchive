import os
from pathlib import Path

import numpy as np
import pytest

from facearchive.recognition import Recognition


@pytest.mark.skipif(not os.getenv("FACEARCHIVE_TEST_MODELS"), reason="ONNX resources not supplied")
def test_yunet_sface_inference():
    models = Path(os.environ["FACEARCHIVE_TEST_MODELS"])
    engine = Recognition(models)
    portrait = Path(os.environ["FACEARCHIVE_TEST_PORTRAIT"])
    faces = engine.detect(portrait)
    assert faces
    box, embedding = max(faces, key=lambda item: item[0][2] * item[0][3])
    assert 0.30 < box[0] < 0.40
    assert 0.10 < box[1] < 0.25
    assert all(0 <= value <= 1 for value in box)
    assert embedding.shape == (128,)
    assert np.linalg.norm(embedding) == pytest.approx(1)
