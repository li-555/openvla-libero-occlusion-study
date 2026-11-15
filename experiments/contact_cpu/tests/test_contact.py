import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

from contact_gate import Config, Visible, Robot, ContactGate, pose, inv, errors
from evaluate import load, predict_window, score, summarize


def T(x=0., y=0., z=0., theta=0.):
    return pose([x, y, z], [0, 0, np.sin(theta/2), np.cos(theta/2)], "xyzw")


def config():
    return Config(grip_closed=0., grip_open=.08)


def history(attached=True, moving=True):
    hs = []
    offset = T(z=.05)
    for i in range(5):
        ee = T(x=i*.01 if moving else 0)
        obj = ee @ offset if attached else T(z=.05)
        hs.append(Visible(i, ee, obj, np.zeros(2)))
    return hs


def trace():
    n = 20
    pos = np.column_stack([np.arange(n)*.01, np.zeros(n), np.zeros(n)])
    quat = np.tile([0., 0., 0., 1.], (n, 1))
    return dict(step=np.arange(n), eef_pos=pos, eef_quat=quat,
                bowl_pos=pos+[0., 0., .05], bowl_quat=quat.copy(), gripper_qpos=np.zeros((n, 2)))


class GeometryTests(unittest.TestCase):
    def test_quaternion_orders_and_sign(self):
        q = np.array([.2, -.3, .1, .8])
        np.testing.assert_allclose(pose([1,2,3], q, "xyzw"), pose([1,2,3], q[[3,0,1,2]], "wxyz"))
        np.testing.assert_allclose(pose([0,0,0], q, "xyzw"), pose([0,0,0], -q, "xyzw"))

    def test_rotation_propagates_offset(self):
        gate = ContactGate([Visible(0, T(), T(x=.1), [0,0])])
        p = gate.predict(Robot(1, T(theta=np.pi/2), [0,0]))
        np.testing.assert_allclose(p.physics[:3, 3], [0, .1, 0], atol=1e-12)
        self.assertAlmostEqual(errors(p.physics, T(y=.1, theta=np.pi/2))[1], 0.)

    def test_nonidentity_anchor(self):
        ee = T(1, 2, 3, .4)
        obj = T(2, 3, 4, -.7)
        now = T(3, 1, 2, -.6)
        gate = ContactGate([Visible(0, ee, obj, [0,0])])
        p = gate.predict(Robot(1, now, [0,0]))
        np.testing.assert_allclose(p.physics, now @ inv(ee) @ obj)
        np.testing.assert_array_equal(p.memory, obj)

    def test_bad_geometry(self):
        for q in ([0,0,0,0], [0,0,0,np.nan]):
            with self.assertRaises(ValueError):
                pose([0,0,0], q, "xyzw")
        with self.assertRaises(ValueError):
            ContactGate([Visible(0, np.zeros((4,4)), T(), [0,0])])


class GateTests(unittest.TestCase):
    def test_rigid_motion(self):
        gate = ContactGate(history(), config())
        self.assertEqual(gate.mode, "physics")
        p = gate.predict(Robot(5, T(x=.05), [0,0]))
        np.testing.assert_allclose(p.selected, T(x=.05, z=.05))

    def test_static_object(self):
        gate = ContactGate(history(attached=False), config())
        self.assertEqual(gate.mode, "memory")
        np.testing.assert_allclose(gate.predict(Robot(5, T(x=.05), [0,0])).selected, T(z=.05))

    def test_stationary_pair_ambiguous(self):
        self.assertEqual(ContactGate(history(moving=False), config()).mode, "uncertain")

    def test_no_calibration_no_positive_contact(self):
        gate = ContactGate(history())
        self.assertEqual(gate.mode, "uncertain")
        self.assertIsNone(gate.predict(Robot(5, T(x=.05), [0,0])).selected)

    def test_far_object(self):
        hs = history()
        hs = [Visible(h.step, h.eef, h.obj @ T(z=1), h.gripper) for h in hs]
        self.assertEqual(ContactGate(hs, config()).mode, "uncertain")

    def test_opening_latches_uncertainty(self):
        gate = ContactGate(history(), config())
        self.assertEqual(gate.predict(Robot(5, T(x=.05), [.04,-.04])).mode, "uncertain")
        self.assertEqual(gate.predict(Robot(6, T(x=.06), [0,0])).mode, "uncertain")

    def test_out_of_calibration_latches_uncertainty(self):
        gate = ContactGate(history(), config())
        self.assertEqual(gate.predict(Robot(5, T(), [10,-10])).mode, "uncertain")

    def test_short_and_gapped_history(self):
        self.assertEqual(ContactGate(history()[:2], config()).mode, "uncertain")
        hs = history()
        hs[-1] = Visible(7, hs[-1].eef, hs[-1].obj, hs[-1].gripper)
        self.assertEqual(ContactGate(hs, config()).mode, "uncertain")
        gate = ContactGate(history(), config())
        self.assertEqual(gate.predict(Robot(7, T(), [0,0])).mode, "uncertain")

    def test_bad_step_and_type(self):
        gate = ContactGate(history(), config())
        with self.assertRaises(ValueError):
            gate.predict(Robot(4, T(), [0,0]))
        with self.assertRaises(TypeError):
            gate.predict(history()[-1])
        with self.assertRaises(ValueError):
            ContactGate(history()[::-1])

    def test_copy_isolation(self):
        hs = history()
        gate = ContactGate(hs, config())
        hs[-1].obj[:3,3] = 100
        p = gate.predict(Robot(5, T(x=.05), [0,0]))
        np.testing.assert_allclose(p.physics, T(x=.05,z=.05))

    def test_memory_not_trusted_after_hidden_close(self):
        c = config()
        # Visible object static while robot moves, with gripper open.
        hs = [Visible(h.step, h.eef, h.obj, [.04,-.04]) for h in history(attached=False)]
        gate = ContactGate(hs, c)
        self.assertEqual(gate.mode, "memory")
        # Once the gripper closes out of sight, the old static hypothesis is no
        # longer a justified exclusive estimate. No hidden truth was inspected.
        p = gate.predict(Robot(5, T(x=.05), [0,0]))
        self.assertEqual(p.mode, "uncertain")
        self.assertIsNone(p.selected)
        self.assertEqual(gate.predict(Robot(6,T(x=.06), [.04,-.04])).mode, "uncertain")

    def test_memory_latches_uncertainty_on_telemetry_gap(self):
        gate = ContactGate(history(attached=False), config())
        self.assertEqual(gate.mode, "memory")
        self.assertEqual(gate.predict(Robot(7, T(x=.07), [0,0])).mode, "uncertain")

    def test_config(self):
        for kw in (dict(history=2), dict(history=3.1), dict(max_gap_steps=1.5),
                   dict(grip_closed=0), dict(grip_closed=0,grip_open=0),
                   dict(proximity_m=-1), dict(closed_fraction_max=.9)):
            with self.assertRaises(ValueError):
                Config(**kw)


class EvaluationTests(unittest.TestCase):
    def test_future_truth_invariance_and_unidentifiable_slip(self):
        d = trace()
        p, m = predict_window(d, 5, 10, config(), "xyzw", "xyzw")
        altered = {k:v.copy() for k,v in d.items()}
        # Slip changes hidden object truth but not robot observations.
        altered["bowl_pos"][5:] += [4, 5, 6]
        altered["bowl_quat"][5:] = [0, 0, 1, 0]
        q, n = predict_window(altered, 5, 10, config(), "xyzw", "xyzw")
        self.assertEqual(m, n)
        for a, b in zip(p,q):
            self.assertEqual(a.mode,b.mode)
            np.testing.assert_array_equal(a.memory,b.memory)
            np.testing.assert_array_equal(a.physics,b.physics)
        self.assertGreater(score(altered,q,"xyzw","test","5:10")[0]["m1_pos_m"], 1)

    def test_window_boundary_and_scores(self):
        d = trace()
        p, _ = predict_window(d, 5, 10, config(), "xyzw", "xyzw")
        self.assertEqual([v.step for v in p], list(range(5,11)))
        s = summarize(score(d,p,"xyzw","test","5:10"))
        self.assertEqual(s["resolved_fraction"], 1.)
        self.assertLess(s["m1_mean_pos_m"], 1e-12)
        self.assertGreater(s["m0_mean_pos_m"], .01)
        for w in [(0,14),(10,5),(19,20)]:
            with self.assertRaises(ValueError):
                predict_window(d,*w,config(),"xyzw","xyzw")

    def test_uncertainty_metrics_not_oracle(self):
        d = trace()
        p, _ = predict_window(d, 5, 10, Config(), "xyzw", "xyzw")
        s = summarize(score(d,p,"xyzw","test","5:10"))
        self.assertEqual(s["resolved_fraction"], 0.)
        self.assertIsNone(s["m2_CONDITIONAL_mean_pos_m"])

    def test_plot_result_headless_if_matplotlib_available(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest("Matplotlib is optional")
        with tempfile.TemporaryDirectory(dir=os.environ.get("CONTACT_TEST_TMP")) as tmp:
            root = Path(tmp)
            f = root/"synthetic.npz"
            np.savez(f, **trace())
            script = Path(__file__).resolve().parents[1]/"evaluate.py"
            plot = Path(__file__).resolve().parents[1]/"plot_results.py"
            folder = root/"result"
            cmd = [sys.executable, str(script), str(f), "--windows", "5:10",
                   "--object-quat-order", "xyzw", "--eef-quat-order", "xyzw",
                   "--out", str(folder)]
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            draw = subprocess.run([sys.executable, str(plot), str(folder)], capture_output=True, text=True)
            self.assertEqual(draw.returncode, 0, draw.stderr)
            imgs = list((folder/"plots").glob("*.png"))
            self.assertEqual(len(imgs), 1)
            self.assertGreater(imgs[0].stat().st_size, 1000)

    def test_cli_and_input_validation(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("CONTACT_TEST_TMP")) as tmp:
            root = Path(tmp)
            f = root/"synthetic.npz"
            np.savez(f, **trace())
            self.assertEqual(load(f)["step"].shape, (20,))
            script = Path(__file__).resolve().parents[1]/"evaluate.py"
            cmd = [sys.executable, str(script), str(f), "--windows", "5:10",
                   "--object-quat-order", "xyzw", "--eef-quat-order", "xyzw",
                   "--out", str(root/"result")]
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            summary = json.loads((root/"result"/"summary.json").read_text())
            self.assertEqual(summary["windows"][0]["anchor"], 4)
            self.assertTrue((root/"result"/"predictions.jsonl").exists())
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            bad = trace()
            bad["step"][2] = 1
            np.savez(f, **bad)
            with self.assertRaises(ValueError):
                load(f)


if __name__ == "__main__":
    unittest.main()
