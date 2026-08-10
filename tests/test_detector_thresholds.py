from PIL import Image

from gom.detectors.base import Detector
from gom.types import Detection


class BatchDetector(Detector):
    def __init__(self):
        super().__init__("batch", device="cpu", score_threshold=0.5)

    def detect(self, image):
        return self.detect_batch([image])[0]

    def detect_batch(self, images):
        return [
            [
                Detection(box=(0, 0, 4, 4), label="low", score=0.49),
                Detection(box=(1, 1, 5, 5), label="high", score=0.50),
            ]
            for _ in images
        ]


def test_run_batch_applies_same_threshold_as_run():
    detector = BatchDetector()
    result = detector.run_batch([Image.new("RGB", (8, 8))])
    assert [d.label for d in result[0]] == ["high"]
