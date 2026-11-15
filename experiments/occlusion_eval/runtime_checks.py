"""Small runtime guards; no change to OpenVLA inference or ContactGate."""
import hashlib
import importlib.metadata
import inspect
import platform
import sys
from pathlib import Path
import numpy as np


def resolve_sim(env):
    seen = set()
    for _ in range(8):
        if id(env) in seen:
            break
        seen.add(id(env))
        sim = getattr(env, 'sim', None)
        if sim is not None:
            return sim
        env = getattr(env, 'env', None)
        if env is None:
            break
    raise RuntimeError('No sim found in environment wrapper chain')


def success_check(env):
    # No done=>success fallback: unsupported versions fail explicitly.
    check = getattr(env, 'check_success', None)
    if callable(check):
        return check, 'env.check_success'
    raise RuntimeError('A callable env.check_success is required; done is not a success label')


def pair_seed(seed, task, episode):
    return int(np.random.SeedSequence([seed, task, episode]).generate_state(1)[0])


def method_order(methods, seed, task, episode):
    rng = np.random.default_rng(np.random.SeedSequence([seed, task, episode, 771]))
    return [str(x) for x in rng.permutation(methods)]


def observation_digest(obs):
    # Public policy inputs only. Captures reset/warmup reproducibility without
    # exposing object truth to a method. Not a proof of identical hidden state.
    digest = hashlib.sha256()
    for key in ('agentview_image', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos'):
        if key not in obs:
            continue
        arr = np.ascontiguousarray(obs[key])
        digest.update(key.encode())
        digest.update(str((arr.shape, arr.dtype.str)).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def configure_tensorflow_cpu():
    # TF performs image preprocessing only. CUDA_VISIBLE_DEVICES must NOT be
    # changed globally because PyTorch needs the allocated GPU.
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], 'GPU')
    except RuntimeError as exc:
        raise RuntimeError('TensorFlow GPU already initialized; use a fresh process') from exc


def runtime_metadata(root):
    result = {'python': platform.python_version(), 'numpy': np.__version__,
              'versions': {}, 'source_sha256': {}, 'note':
              'Source hashes are recorded; pin checkpoint weights separately before formal evaluation.'}
    for name in ('torch', 'tensorflow', 'transformers', 'tokenizers', 'accelerate',
                 'bitsandbytes', 'robosuite', 'mujoco', 'mujoco-py', 'libero'):
        try:
            result['versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result['versions'][name] = 'not_in_package_metadata'
    paths = list((Path(root)/'experiments'/'occlusion_eval').glob('*.py'))
    for name, module in list(sys.modules.items()):
        if name.startswith(('experiments.robot', 'experiments.contact_cpu',
                            'experiments.physics_occlusion', 'robosuite.utils.camera_utils')):
            path = getattr(module, '__file__', None)
            if path:
                paths.append(Path(path))
    for path in sorted(set(paths)):
        if path.is_file():
            result['source_sha256'][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        import robosuite.macros as macros
        result['image_convention'] = macros.IMAGE_CONVENTION
    except ImportError:
        result['image_convention'] = 'unavailable'
    return result
