# Physics-Guided Visual Compensation

This package is an independent second-stage CPU/interface implementation. It
does not modify `experiments/contact_cpu`, the original LIBERO runner, or
previous result directories.

The fixed occlusion schedule is an image-space normalized rectangle and
inclusive step interval. It is never moved using object truth. The same
`CompensationPipeline` renders the same projected pose cue for M0, M1, and M2.
M2 `uncertain` produces an unchanged occluded image and records
`applied=false`.

The pipeline accepts only a camera model, image, step, and `PosePrediction`.
There is no API for object truth, contact truth, segmentation masks, or MuJoCo
handles. `openvla_adapter.prepare_processor_inputs()` mirrors the existing
`processor(prompt, image)` plus attention-mask repair path and saves the exact
image passed to the processor.

This stage requires a camera calibration (`intrinsic` and
`camera_from_world`) for real LIBERO projection. No GPU or MuJoCo smoke test
is run on the login gateway. The remaining GPU work is to instantiate the
existing model/processor and compare clean, occluded, and compensated inputs
inside an approved compute allocation.
