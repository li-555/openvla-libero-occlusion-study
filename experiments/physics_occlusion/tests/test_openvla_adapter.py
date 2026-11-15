import unittest

import numpy as np

from openvla_adapter import predict_action_from_compensated_image


class Batch(dict):
    def to(self, device, dtype=None):
        return self


class FakeProcessor:
    def __init__(self):
        self.image = None

    def __call__(self, prompt, image):
        self.image = np.asarray(image).copy()
        import torch

        return Batch(
            input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
        )


class FakeModel:
    def __init__(self):
        self.kwargs = None

    def predict_action(self, **kwargs):
        self.kwargs = kwargs
        return np.zeros(7)


class AdapterTests(unittest.TestCase):
    def test_compensated_pixels_reach_processor_and_model(self):
        processor = FakeProcessor()
        model = FakeModel()
        image = np.full((8, 8, 3), 77, dtype=np.uint8)
        result = predict_action_from_compensated_image(
            model,
            processor,
            "openvla",
            image,
            "pick up the bowl",
            "libero_spatial",
            "cpu",
            None,
        )
        np.testing.assert_array_equal(processor.image, image)
        self.assertEqual(result.shape, (7,))
        self.assertEqual(model.kwargs["unnorm_key"], "libero_spatial")
        self.assertFalse(model.kwargs["do_sample"])


if __name__ == "__main__":
    unittest.main()
