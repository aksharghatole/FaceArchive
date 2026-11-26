import os
from pathlib import Path

import pytest

from facearchive.recognition import Recognition


@pytest.mark.skipif(not os.getenv("FACEARCHIVE_TEST_MODELS"), reason="ONNX resources not supplied")
def test_yunet_sface_inference():
    models = Path(os.environ["FACEARCHIVE_TEST_MODELS"])
    Recognition(models).check_models()
