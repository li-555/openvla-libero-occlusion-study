"""Optional minimum insertion into the EXISTING frozen OpenVLA run_gpu.py.

No runner rewrite. Default is --dry-run. --apply makes a timestamp-free .bak
and refuses to overwrite it. The assistant only tested insertion on the
provided local snapshot, not on the HKU runner.
"""
import argparse
import ast
from pathlib import Path

STAMP='# GRASP_V2_ACTION_CANDIDATE_PATCH_V1'


def insert_once(src, before, insertion):
    if src.count(before)!=1:
        raise ValueError(f'Expected exactly one source anchor: {before[:90]!r}')
    return src.replace(before, insertion+before,1)


def patch_text(src):
    if STAMP in src:
        raise ValueError('grasp v2 patch already installed; refusing duplicate')
    required=["p.add_argument('--unified-history', type=int, default=6)",
              "    for _ in range(args.warmup):\n",
              "        if getattr(args,'paired_capture_dev',False):\n",
              "        if ai1_recorder is not None and save:\n",
              "    if a.unified_estimator == 'learned_contact'"]
    for s in required:
        if src.count(s)!=1:
            raise ValueError(f'Source incompatible: expected one {s!r}')
    src=insert_once(src,"    p.add_argument('--unified-history', type=int, default=6)",
       "    p.add_argument('--grasp-v2-model', default=None, help='Frozen model trained on executed grasp probes')\n")
    guard=("    if a.grasp_v2_model and (a.methods != ['integrated'] or a.unified_estimator != 'memory'\n"
           "                             or a.unified_intervention != 'none' or getattr(a,'grasp_recovery_model',None)):\n"
           "        p.error('grasp-v2 requires integrated/memory/none, without demo-guided controller')\n")
    src=insert_once(src,"    if a.unified_estimator == 'learned_contact'",guard)
    setup=("    grasp_v2=None\n"
           "    if getattr(args,'grasp_v2_model',None):\n"
           "        from experiments.grasp_v2.controller import AdaptiveGraspController\n"
           "        grasp_v2=AdaptiveGraspController(args.grasp_v2_model)\n")
    src=insert_once(src,"    for _ in range(args.warmup):\n",setup)
    observe=("        if grasp_v2 is not None:\n"
             "            _v2_ep=camera.project(np.asarray(obs['robot0_eef_pos'],dtype=float))\n"
             "            _v2_visible=(unified_est.projected_xy_px if unified_est is not None and unified_est.valid and not schedule.active(step) else None)\n"
             "            from experiments.grasp_v2.aperture import aperture_from_qpos\n"
             "            grasp_v2.observe(step=step,occluded=schedule.active(step),\n"
             "                memory_pixel=_v2_visible,eef_pixel=_v2_ep,\n"
             "                eef_xyz=obs['robot0_eef_pos'],\n"
             "                gripper_aperture_m=aperture_from_qpos(obs['robot0_gripper_qpos']),\n"
             "                previous_action=executed_previous)\n")
    src=insert_once(src,"        if getattr(args,'paired_capture_dev',False):\n",observe)
    action=("        if grasp_v2 is not None:\n"
            "            executed,_v2_event=grasp_v2.override(executed,step=step)\n"
            "            output.metadata['grasp_v2_event']=_v2_event\n")
    src=insert_once(src,"        if ai1_recorder is not None and save:\n",action)
    src=STAMP+'\n'+src
    ast.parse(src)
    return src


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner',default='experiments/occlusion_eval/run_gpu.py')
    parser.add_argument('--apply',action='store_true',help='Explicitly modify the runner after validation')
    a=parser.parse_args(argv)
    path=Path(a.runner)
    original=path.read_text(encoding='utf-8')
    patched=patch_text(original)
    if not a.apply:
        print(f'DRY RUN OK: compatible runner {path}; 5 injection sites; no file modified')
        return
    bak=path.with_suffix(path.suffix+'.pre_grasp_v2.bak')
    if bak.exists():raise FileExistsError(f'Existing backup at {bak}; not overwriting')
    bak.write_text(original,encoding='utf-8')
    path.write_text(patched,encoding='utf-8')
    print(f'Applied: {path}; backup: {bak}')

if __name__=='__main__':main()
