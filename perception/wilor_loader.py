"""Load WiLoR (hand reconstruction) and its YOLO hand detector without relying on the current directory.

WiLoR's own ``load_wilor`` (third_party/WiLoR/wilor/models/__init__.py, commit fcb9113) uses paths relative to its repo
folder, so its demo only works with cwd = third_party/WiLoR. We never chdir; instead this module
- puts third_party/WiLoR on ``sys.path`` (here, in Python -- never through the PYTHONPATH variable);
- builds the same config as ``load_wilor`` with ABSOLUTE paths. Every relative path WiLoR touches at load time:
    * model_config.yaml / wilor_final.ckpt / detector.pt  -> passed as absolute paths below;
    * get_config(update_cachedir=True) prefixes MANO.MODEL_PATH / MEAN_PARAMS with "./pretrained_models/" -> we call it
      with update_cachedir=False and set both ourselves (yacs does not expand the yaml's ${MANO.DATA_DIR});
    * load_wilor overwrites MANO.DATA_DIR / MODEL_PATH / MEAN_PARAMS with "./mano_data/..." -> we set the absolute mano_data
      folder (smplx picks MANO_RIGHT.pkl inside it; MEAN_PARAMS is np.load-ed by the ViT head);
    * MODEL.BACKBONE.PRETRAINED_WEIGHTS = "training_data/vitpose_backbone.pth" -> popped, exactly as load_wilor does
      (it is a training-time initialisation, overwritten by the checkpoint);
  load_wilor also sets MODEL.BBOX_SHAPE = [192, 256] for the ViT backbone (ViTDetDataset needs it) -> same here.
- constructs ``WiLoR(cfg, init_renderer=False)`` (the renderer is only used for training logs and would need a GL context)
  and loads the checkpoint's state dict ourselves with torch.load, ``strict=False`` like load_wilor (all 490 keys match).
- ``model.eval()`` (the demo does this after load_wilor).

Checkpoint loading under torch 2.11: wilor_final.ckpt loads with torch's default ``weights_only=True`` (plain tensors in an
OrderedDict; checked on 2026-10-30). The YOLO detector does not: ultralytics calls ``torch.load(file)`` itself, and the
file is a dill-pickled ultralytics model. We cannot pass ``weights_only`` through ultralytics, so ``load_detector`` sets
``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`` (torch reads it on every call; only affects calls that do not pass weights_only)
before loading this trusted local file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WILOR_DIR = REPO / "third_party" / "WiLoR"
CONFIG_YAML = WILOR_DIR / "pretrained_models" / "model_config.yaml"
CHECKPOINT = WILOR_DIR / "pretrained_models" / "wilor_final.ckpt"
DETECTOR = WILOR_DIR / "pretrained_models" / "detector.pt"
MANO_DIR = WILOR_DIR / "mano_data"

DETECTOR_CONF = 0.3        # WiLoR demo.py:53  detector(img, conf=0.3)
RIGHT_CLASS = 1            # detector.names == {0: 'left', 1: 'right'}
RESCALE_FACTOR = 2.0       # WiLoR demo.py:22  --rescale_factor default (box enlargement for the crop)


def _add_wilor_to_path() -> None:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")   # wilor.utils imports pyrender; never opens a window
    sys.dont_write_bytecode = True                       # keep the third-party checkout clean (it tracks some .pyc files)
    if str(WILOR_DIR) not in sys.path:
        sys.path.insert(0, str(WILOR_DIR))


def wilor_config():
    """WiLoR's model config with absolute paths (same edits as load_wilor)."""
    _add_wilor_to_path()
    from wilor.configs import get_config
    cfg = get_config(str(CONFIG_YAML), update_cachedir=False)
    cfg.defrost()
    if "BBOX_SHAPE" not in cfg.MODEL:
        assert cfg.MODEL.IMAGE_SIZE == 256
        cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
    cfg.MANO.DATA_DIR = str(MANO_DIR)
    cfg.MANO.MODEL_PATH = str(MANO_DIR)
    cfg.MANO.MEAN_PARAMS = str(MANO_DIR / "mano_mean_params.npz")
    cfg.freeze()
    return cfg


def load_wilor(device: str = "cuda"):
    """-> (model in eval mode on `device`, cfg, MANO faces (1538, 3) int)."""
    import torch
    cfg = wilor_config()
    from wilor.models import WiLoR
    model = WiLoR(cfg, init_renderer=False)
    # weights_only passed explicitly: load_detector sets TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD, which only affects calls without it
    state = torch.load(str(CHECKPOINT), map_location="cpu", mmap=True, weights_only=True)["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    del state
    model = model.to(device).eval()
    return model, cfg, model.mano.faces.astype("int32")


def load_detector(device: str = "cuda"):
    """WiLoR's YOLO hand detector (classes 0 left, 1 right)."""
    os.environ["YOLO_AUTOINSTALL"] = "false"              # read when ultralytics is first imported
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"  # see module docstring
    from ultralytics import YOLO
    import ultralytics.hub.utils as hub_utils
    hub_utils.events.enabled = False                      # no analytics request on predict
    return YOLO(str(DETECTOR)).to(device)
