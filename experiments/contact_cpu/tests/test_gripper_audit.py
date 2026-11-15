import tempfile
import unittest
from pathlib import Path

import numpy as np

from gripper_audit import audit


class GripperAuditTests(unittest.TestCase):
    def test_audit_does_not_claim_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.npz"
            np.savez(path, step=np.arange(2), gripper_qpos=np.array([[.04, -.04], [0., 0.]]))
            result = audit(path)
        self.assertEqual(result["aperture_min"], 0.0)
        self.assertEqual(result["aperture_max"], 0.08)
        self.assertEqual(result["calibration_status"], "not_calibrated")

    def test_rejects_bad_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.npz"
            np.savez(path, gripper_qpos=np.zeros((2, 1)))
            with self.assertRaises(ValueError):
                audit(path)


if __name__ == "__main__":
    unittest.main()
