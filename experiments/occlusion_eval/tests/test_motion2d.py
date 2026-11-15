"""CPU tests for opt-in pixel-space experiment; no OpenVLA or simulator needed."""
import unittest
import numpy as np
from experiments.contact_cpu.contact_gate import Config
from experiments.physics_occlusion.pipeline import CameraModel,OcclusionSchedule
from experiments.occlusion_eval.motion2d import PixelMotion2D,Pixel2DConfig,DarkPixel2D
from experiments.occlusion_eval.core import TrialController


def camera():
    C=np.eye(4)
    C[2,3]=1
    return CameraModel(np.array([[100.,0,32],[0,100.,32],[0,0,1]]),C)


class MockVisiblePixel:
    def estimate(self,image):
        return np.array([32.,32.])


class TestMotion2D(unittest.TestCase):
    def setUp(self):
        self.cam=camera()
        self.grip=Config(grip_closed=0.,grip_open=.08)
        self.c=Pixel2DConfig(history=3,min_robot_motion_px=2,min_object_motion_px=1,
                              relative_tolerance_px=1,static_tolerance_px=.5)

    def test_m1_propagates_eef_pixels_without_true_object(self):
        p=PixelMotion2D('m1_2d',self.cam,self.grip,self.c)
        for i in range(3):
            p.observe_visible(i,[32,32],[.01*i,0,0],[.04,-.04])
        xy,mode,why=p.predict_hidden(3,[.03,0,0],[.04,-.04])
        self.assertEqual(mode,'physics')
        np.testing.assert_allclose(xy,[33,32])

    def test_m1_rejects_stale_anchor(self):
        p=PixelMotion2D('m1_2d',self.cam,self.grip,self.c)
        p.observe_visible(0,[32,32],[0,0,0],[.04,-.04])
        xy,mode,why=p.predict_hidden(3,[.03,0,0],[.04,-.04])
        self.assertIsNone(xy)
        self.assertEqual(mode,'uncertain')

    def test_m2_selects_physics_for_visible_comotion_and_closed(self):
        p=PixelMotion2D('m2_2d',self.cam,self.grip,self.c)
        for i in range(3):
            p.observe_visible(i,[32.+2*i,32],[.02*i,0,0],[0.,0.])
        xy,mode,reason=p.predict_hidden(3,[.06,0,0],[0.,0.])
        self.assertEqual(mode,'physics')
        np.testing.assert_allclose(xy,[38,32])

    def test_m2_selects_memory_for_visible_static_object_open_grip(self):
        p=PixelMotion2D('m2_2d',self.cam,self.grip,self.c)
        for i in range(3):
            p.observe_visible(i,[32,32],[.02*i,0,0],[.04,-.04])
        xy,mode,reason=p.predict_hidden(3,[.06,0,0],[.04,-.04])
        self.assertEqual(mode,'memory')
        np.testing.assert_allclose(xy,[32,32])

    def test_no_gripper_calibration_means_no_claimed_physics(self):
        p=PixelMotion2D('m2_2d',self.cam,Config(),self.c)
        for i in range(3):
            p.observe_visible(i,[32+2*i,32],[.02*i,0,0],[0.,0.])
        xy,mode,_=p.predict_hidden(3,[.06,0,0],[0.,0.])
        self.assertIsNone(xy)
        self.assertEqual(mode,'uncertain')

    def test_loss_of_closed_grip_latches_uncertain(self):
        p=PixelMotion2D('m2_2d',self.cam,self.grip,self.c)
        for i in range(3):
            p.observe_visible(i,[32+2*i,32],[.02*i,0,0],[0.,0.])
        p.predict_hidden(3,[.06,0,0],[0.,0.])
        xy,mode,_=p.predict_hidden(4,[.08,0,0],[.04,-.04])
        self.assertIsNone(xy)
        self.assertEqual(mode,'uncertain')
        xy,mode,_=p.predict_hidden(5,[.10,0,0],[0,0])
        self.assertIsNone(xy) # no re-enabling physics while invisible
        self.assertEqual(mode,'uncertain')

    def test_telemetry_gap_is_uncertain_in_m2(self):
        p=PixelMotion2D('m2_2d',self.cam,self.grip,self.c)
        for i in range(3):
            p.observe_visible(i,[32+2*i,32],[.02*i,0,0],[0,0])
        xy,mode,_=p.predict_hidden(5,[.10,0,0],[0,0])
        self.assertIsNone(xy)
        self.assertEqual(mode,'uncertain')

    def test_rgb_detector_only_sees_pixels(self):
        rgb=np.full((64,64,3),170,dtype=np.uint8)
        rgb[28:36,30:34]=0
        d=DarkPixel2D((.35,.35,.65,.65),Pixel2DConfig())
        xy=d.estimate(rgb)
        np.testing.assert_allclose(xy,[31.5,31.5])
        rgb[28:36,30:34]=170
        self.assertIsNone(d.estimate(rgb))

    def test_controller_uses_2d_cue_only_during_occlusion(self):
        c=TrialController('m1_2d',self.cam,OcclusionSchedule(3,4,.25,.25,.75,.75),
                          self.grip,allow_oracle=False,pixel_estimator=MockVisiblePixel(),
                          motion2d_cfg=self.c)
        rgb=np.full((64,64,3),170,dtype=np.uint8)
        q=[0,0,0,1]
        for i in range(3):
            result=c.process(step=i,raw=rgb,eef_pos=[.02*i,0,0],eef_quat=q,
                         gripper_qpos=[0.,0.])
            np.testing.assert_array_equal(result.action_image,rgb)
        result=c.process(step=3,raw=rgb,eef_pos=[.06,0,0],eef_quat=q,
                         gripper_qpos=[0.,0.])
        self.assertEqual(result.metadata['gate'],'physics')
        self.assertTrue(result.metadata['compensation_applied'])
        self.assertFalse(np.array_equal(result.action_image,result.occluded))

    def test_privileged_hidden_object_rejected(self):
        c=TrialController('m1_2d',self.cam,OcclusionSchedule(3,4,.25,.25,.75,.75),
                          self.grip,allow_oracle=False,pixel_estimator=MockVisiblePixel(),
                          motion2d_cfg=self.c)
        with self.assertRaisesRegex(ValueError,'Hidden privileged'):
            c.process(step=3,raw=np.full((64,64,3),170,dtype=np.uint8),
                      eef_pos=[0,0,0],eef_quat=[0,0,0,1],gripper_qpos=[0,0],
                      privileged_visible_observation={'cup_pos':[1,2,3]})

if __name__=='__main__': unittest.main()
