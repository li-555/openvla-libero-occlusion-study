"""Optional adapter proving the compensated image reaches OpenVLA's processor."""
from pathlib import Path

import numpy as np


OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def build_prompt(base_vla_name, task_label):
    if "openvla-v01" in base_vla_name:
        return f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def prepare_processor_inputs(processor, image, prompt, device, dtype, artifact_dir=None, step=None):
    """Build the exact processor inputs from the supplied compensated image.

    This mirrors the existing OpenVLA call path without loading a model. The
    caller supplies the image produced by CompensationPipeline; no object
    state, simulator object, or segmentation value is accepted.
    """
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("OpenVLA adapter requires torch and Pillow") from exc
    image_array = np.asarray(image)
    if image_array.ndim != 3 or image_array.shape[2] != 3 or image_array.dtype != np.uint8:
        raise ValueError("image must be uint8 HxWx3")
    pil_image = Image.fromarray(image_array)
    if artifact_dir is not None and step is not None:
        path = Path(artifact_dir)
        path.mkdir(parents=True, exist_ok=True)
        pil_image.save(path / f"step_{int(step):06d}_processor_input.png")
    inputs = processor(prompt, pil_image).to(device, dtype=dtype)
    if "attention_mask" in inputs and not torch.all(inputs["input_ids"][:, -1] == 29871):
        inputs["attention_mask"] = torch.cat(
            [inputs["attention_mask"], torch.ones_like(inputs["attention_mask"][:, -1:])], dim=1
        )
    return inputs


def predict_action_from_compensated_image(
    vla,
    processor,
    base_vla_name,
    image,
    task_label,
    unnorm_key,
    device,
    dtype,
    artifact_dir=None,
    step=None,
):
    """Send the compensated image through the real OpenVLA processor and model."""
    prompt = build_prompt(base_vla_name, task_label)
    inputs = prepare_processor_inputs(
        processor, image, prompt, device, dtype, artifact_dir=artifact_dir, step=step
    )
    return vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
