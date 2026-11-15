import unittest
import numpy as np

from experiments.contact_cpu.contact_gate import Config
from experiments.occlusion_eval.core import (
    RGBPlaneConfig, RGBPlaneEstimator, TrialController, backproject_to_horizontal_plane
)
from experiments.physics_occlusion.pipeline import CameraModel, OcclusionSchedule


def camera():
    T=np.eye(4)
    T[2,3]=1.0  # world origin -> camera Z=1
    return CameraModel(np.array([[100,0,32],[0,100,32],[0,0,1]],dtype=float),T)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.cam=camera()
        self.rgb=np.full((64,64,3),150,dtype=np.uint8)
        self.q=np.array([0.,0.,0.,1.])
        self.cfg=Config(grip_closed=0.,grip_open=.08)

    def controller(self, method, allow_oracle=True):
        return TrialController(method,self.cam,OcclusionSchedule(5,8,.2,.2,.8,.8),
                               self.cfg, allow_oracle=allow_oracle, target_name='cup')

    def obs(self, x=0.1):
        return {'cup_pos':np.array([x,0.,0.]),'cup_quat':self.q.copy()}

    def call(self,c,step, eefx=.1, objx=.1, hidden=False):
        return c.process(step=step,raw=self.rgb,eef_pos=[eefx,0,0],
                         eef_quat=self.q,gripper_qpos=[0,0],
                         privileged_visible_observation=None if hidden else self.obs(objx))

    def test_pre_occlusion_rigid_comotion_selects_physics(self):
        c=self.controller('m2')
        for i in range(5):
            self.call(c,i,eefx=.01*i,objx=.02+.01*i)
        res=self.call(c,5,eefx=.06,hidden=True)
        self.assertEqual(res.metadata['gate'],'physics')
        self.assertTrue(res.metadata['compensation_applied'])

    def test_static_object_selects_memory(self):
        c=self.controller('m2')
        for i in range(5):
            self.call(c,i,eefx=.02+.02*i,objx=.0)
        res=self.call(c,5,eefx=.12,hidden=True)
        self.assertEqual(res.metadata['gate'],'memory')

    def test_oracle_after_occlusion_is_hard_error(self):
        c=self.controller('m1')
        self.call(c,0, eefx=.0,objx=.0)
        with self.assertRaisesRegex(ValueError,'Hidden privileged'):
            self.call(c,5,eefx=.1,objx=.1,hidden=False)

    def test_hidden_world_state_is_not_an_api_input(self):
        c=self.controller('m0',allow_oracle=False)
        for i in range(5):
            self.call(c,i,hidden=True)
        r=self.call(c,5,hidden=True)
        self.assertFalse(r.metadata.get('compensation_applied',False))
        self.assertEqual(r.metadata.get('gate'),'no_visible_anchor')

    def test_camera_plane_backprojection(self):
        p=backproject_to_horizontal_plane(self.cam,(32,32),0.)
        np.testing.assert_allclose(p,[0.,0.,0.])

    def test_rgb_plane_uses_only_pixels(self):
        rgb=self.rgb.copy()
        rgb[26:39,26:39]=0
        detector=RGBPlaneEstimator(self.cam,RGBPlaneConfig((.3,.3,.7,.7),0.))
        T=detector.estimate(rgb)
        self.assertIsNotNone(T)
        np.testing.assert_allclose(T[:3,3],[0.,0.,0.],atol=.005)

    def test_clean_and_occluded_preserve_input_contract(self):
        a=self.controller('clean',allow_oracle=False)
        b=self.controller('occluded',allow_oracle=False)
        x=self.call(a,5,hidden=True)
        y=self.call(b,5,hidden=True)
        np.testing.assert_array_equal(x.action_image,self.rgb)
        self.assertFalse(np.array_equal(y.action_image,self.rgb))

    def test_mask_is_fixed_not_target_following(self):
        c=self.controller('m1',allow_oracle=False)
        r1=self.call(c,5,hidden=True)
        r2=self.call(c,6,hidden=True)
        np.testing.assert_array_equal(r1.occluded,r2.occluded)

if __name__=='__main__': unittest.main()
