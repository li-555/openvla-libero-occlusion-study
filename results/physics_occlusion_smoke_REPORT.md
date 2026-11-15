# Physics-Guided Visual Compensation interface smoke test

## Completed without GPU or MuJoCo

The existing OpenVLA path is:

1. `obs["agentview_image"]`
2. 180-degree image rotation in `get_libero_image`
3. JPEG encode/decode and Lanczos resize
4. `observation["full_image"]`
5. `get_action` → `get_vla_action`
6. `Image.fromarray` and optional center crop
7. `processor(prompt, image)`
8. existing attention-mask repair
9. `vla.predict_action(**inputs, unnorm_key=..., do_sample=False)`

The new [`openvla_adapter.py`](../../physics_occlusion/openvla_adapter.py)
uses the same processor and attention-mask path, and the test verifies that
the compensated pixel array reaches the fake processor and model call.

## Controlled occlusion and compensation

[`pipeline.py`](../../physics_occlusion/pipeline.py) applies one fixed
normalized rectangle (`0.32..0.70` in both axes) for the fixed inclusive
step interval configured by `OcclusionSchedule`. It does not inspect or track
the object. A `CameraModel` projects the predicted world pose into image
space, and the same translucent projected pose cue is used for M0, M1, and
M2. M2 uncertain leaves the occluded image unchanged and records
`applied=false`.

## Generated artifacts

For each method, the smoke run saved:

- raw camera-stage image
- fixed-occlusion image
- compensated image
- JSON prediction/mode/compensation record

The M0/M1 compensated images differ from their occluded images. The M2
uncertain compensated image is byte-identical to its occluded image by design.

## Tests

- Existing Contact-Aware package: **24/24 passed**
- New physics-occlusion package: **5/5 passed**
- Python static compilation: passed
- No OpenVLA model, MuJoCo environment, GPU, or hidden object state was used.

## GPU smoke test (not run on login gateway)

After obtaining an approved compute allocation and a valid checkpoint:

```bash
cd ~/openvla_eval
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import numpy as np
from experiments.physics_occlusion.openvla_adapter import predict_action_from_compensated_image

# Supply the existing loaded OpenVLA `model` and `processor` here.
# `image` must be the output of CompensationPipeline.process(...)[2].
action = predict_action_from_compensated_image(
    model, processor, "openvla", image,
    "pick up the black bowl between the plate and the ramekin",
    "libero_spatial", "cuda:0", model.dtype,
    artifact_dir="experiments/results/gpu_smoke_inputs", step=0,
)
assert np.asarray(action).shape == (7,)
print("processor/model smoke passed", np.asarray(action).shape)
PY
```

This command is intentionally only an interface smoke test; it is not a
closed-loop experiment or a claim of task-success improvement.
