# Install

Everything here comes from `docs/log.md` (2026-10-29 hoi-sim, 2026-10-30 hoi-perception, later pip changes), `envs/`, `README.md` and the
code that loads the weights. A command marked **(unverified)** is written from the log or the code but was not re-run on a fresh
machine. The two locks in `envs/` were compared with the live environments on 2026-10-31 and are identical to them
(`pip freeze --exclude-editable`).

**What you need for what**
- Simulation (retargeting, HERO, MuJoCo): `hoi-sim` only, CPU only. The four demo files `data/demos/*.npz` are committed, so the
  simulation part runs without the raw clips, `hoi-perception` or any weights.
- Perception (WiLoR + SAM 2 on my clips): `hoi-perception`, an NVIDIA GPU (the code uses `"cuda"`; peak VRAM 2.8 GB for WiLoR and 0.8 GB
  for SAM 2), the weights and MANO below, and the four raw clips in `data/raw/<id>/` (all four are committed; see
  `data/raw/README.md`).
- Tested on: Ubuntu 24.04, RTX 5060 Laptop (Blackwell, 8 GB), NVIDIA driver 580.178.04, conda, ffmpeg 6.1.1. `perception/stray.py`
  needs the `ffmpeg` and `ffprobe` binaries on PATH (`sudo apt install ffmpeg`).

## 1. Always run with `env -u PYTHONPATH`
If your shell sources a ROS 2 install (mine does, through `~/.bashrc`), that puts `/opt/ros/jazzy/...` Python 3.12 packages on
`PYTHONPATH` ahead of every conda env's own site-packages. Our code never uses ROS; `env -u PYTHONPATH` removes them. Harmless without ROS.

```bash
P="env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python"
S="env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python"
```

## 2. Third-party code (cloned into `third_party/`, which is gitignored)
| repo | commit (checked with `git -C third_party/<dir> rev-parse --short HEAD`, equal to `docs/log.md`) | used for |
|---|---|---|
| `https://github.com/RunpeiDong/HERO` -> `third_party/HERO` | `416c4e0` (`416c4e007e22...`) | whole-body controller, G1 + Dex3 model, example policy |
| `https://github.com/rolpotamias/WiLoR` -> `third_party/WiLoR` | `fcb9113` (`fcb911312a38...`) | 3D hand (MANO) |
| `https://github.com/facebookresearch/sam2` -> `third_party/sam2` | `2b90b9f` (`2b90b9f5ceec...`) | SAM 2.1 masks |
| `https://github.com/StrayRobots/scanner` -> `third_party/stray-scanner` | `ec3e1dc` | only its `docs/format.md`, which `perception/stray.py` cites; not needed to run anything |

```bash
mkdir -p third_party
git clone https://github.com/RunpeiDong/HERO third_party/HERO              && git -C third_party/HERO checkout 416c4e0
git clone https://github.com/rolpotamias/WiLoR third_party/WiLoR           && git -C third_party/WiLoR checkout fcb9113
git clone https://github.com/facebookresearch/sam2 third_party/sam2        && git -C third_party/sam2 checkout 2b90b9f
```
(The clone and checkout lines are plain git on the recorded commits; the log does not record how the clones were first made.)

- HERO's example policy (`checkpoints/example/model.onnx` and `model_hero.json`, 5.5 MB) is inside the HERO git repo. Nothing to download.
- **No `patches/` directory exists, and nothing in third-party code was modified.** `git status` in HERO, SAM 2 and stray-scanner is
  clean; in WiLoR it shows only the untracked weights and MANO files below. The one incident the log records (2026-10-30): importing
  `wilor` rewrote 4 `.pyc` files that WiLoR's repo tracks; they were restored with `git checkout`, and `perception/wilor_loader.py`
  now sets `sys.dont_write_bytecode = True` so it cannot happen again. WiLoR is loaded by our own `perception/wilor_loader.py` with
  absolute paths (its `load_wilor` only works from inside its folder); WiLoR's files are not edited for that.

## 3. `hoi-sim` (Python 3.11; `numpy<2`, `mujoco==3.13.0`, CPU `onnxruntime`, no torch)
What the log did on 2026-10-29: `conda create -n hoi-sim python=3.11`, then `pip install -e third_party/HERO[bench] mujoco==3.13.0 pytest imageio[ffmpeg]`.
`pyyaml` and `matplotlib` were added later with `-c envs/hoi-sim.constraints.txt`. From the lock **(unverified on a fresh env)**:
```bash
conda create -n hoi-sim python=3.11
env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m pip install -r envs/hoi-sim.lock.txt
env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m pip install -e "third_party/HERO[bench]" -c envs/hoi-sim.constraints.txt
```
- Why these pins: HERO needs `numpy<2`; `mujoco==3.13.0` is the only MuJoCo version HERO names (3.17.0 is newer than HERO's code
  release); onnxruntime is the CPU build because HERO's policy is tiny and a GPU build risks CUDA/cuDNN trouble on Blackwell.
- HERO is installed **editable**; its `[bench]` extra adds nothing beyond the core dependencies (log, 2026-10-28). `pip list --editable`
  shows `hero-isaacsim 0.1.0` at `third_party/HERO`; the lock does not contain it (editable installs are left out of the freeze).
- Every later install into this env: `pip install ... -c envs/hoi-sim.constraints.txt`. After the install, `pip check` printed "No broken requirements found." (log).

## 4. `hoi-perception` (Python 3.10; torch 2.11.0+cu128 for Blackwell)
Blackwell needs a CUDA >= 12.8 PyTorch build; the pins are in `envs/hoi-perception.constraints.txt` (torch 2.11.0+cu128,
torchvision 0.26.0+cu128, triton 3.6.0, numpy 2.2.6, opencv-python 4.12.0.88). The lock holds the complete freeze (WiLoR's inference
dependencies, ultralytics 8.1.34, smplx 0.1.28, pyrender 0.1.45, dill, ...). The log does not record how the base env was first made
or which index served the cu128 wheels; the standard PyTorch index is used here **(unverified)**:
```bash
conda create -n hoi-perception python=3.10
env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m pip install --no-deps \
    -r envs/hoi-perception.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
SAM2_BUILD_CUDA=0 env -u PYTHONPATH conda run --no-capture-output -n hoi-perception python -m pip install --no-deps --no-build-isolation -e third_party/sam2
```
- `--no-deps` because the lock is already complete and `pyrender` pins `PyOpenGL==3.1.0` while the env has 3.1.10 (the log installed
  pyrender with `--no-deps` for the same reason; it is the only `pip check` complaint). Never let a requirements file downgrade torch
  (CLAUDE.md); that is also why WiLoR's own `requirements.txt` is not used.
- SAM 2: the log says "editable from `third_party/sam2` (`SAM2_BUILD_CUDA=0`)", i.e. the optional CUDA extension is skipped. The
  `--no-deps --no-build-isolation` flags are my addition **(unverified)**: SAM 2's `pyproject.toml` lists `torch>=2.5.1` as a build requirement, and isolation would
  fetch a second torch.
- chumpy is **not** installed (it was removed on 2026-10-30; see MANO below).
- Runtime variables are set by our code, you do not export them: `perception/wilor_loader.py` sets `PYOPENGL_PLATFORM=egl`,
  `YOLO_AUTOINSTALL=false` and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (the YOLO detector file is a dill-pickled model that torch 2.11's
  default `weights_only` loading rejects; it is a local file you downloaded).

## 5. Weights (not in the repo; gitignored: `*.pt`, `*.ckpt`, `*.pkl`, `third_party/*`)
Paths are the ones the code expects (`perception/wilor_loader.py`, `perception/masks_sam2.py`). Sizes and sha256 are from the log
(2026-10-30) and were re-checked against the files on this machine.

| file | put it at | size | sha256 |
|---|---|---|---|
| WiLoR hand detector `detector.pt` | `third_party/WiLoR/pretrained_models/detector.pt` | 53,582,271 B | `5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033` |
| WiLoR `wilor_final.ckpt` | `third_party/WiLoR/pretrained_models/wilor_final.ckpt` | 2,564,989,533 B (2.56 GB) | `3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2` |
| SAM 2.1 `sam2.1_hiera_base_plus.pt` | `third_party/sam2/checkpoints/sam2.1_hiera_base_plus.pt` | 323,606,802 B | `a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5` |

```bash
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt    -P third_party/WiLoR/pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P third_party/WiLoR/pretrained_models/
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt            -P third_party/sam2/checkpoints/
```
The two WiLoR URLs are from WiLoR's README; the SAM 2.1 URL is the one in `third_party/sam2/checkpoints/download_ckpts.sh` (that script
downloads all four sizes; we only use base_plus, config `configs/sam2.1/sam2.1_hiera_b+.yaml`, which lives inside the sam2 package).
`pretrained_models/model_config.yaml` and `mano_data/mano_mean_params.npz` are tracked in WiLoR's own repo.

### MANO (you register and download it yourself)
MANO is licensed for non-commercial use and cannot be redistributed, so it is not in this repo. WiLoR needs the right hand only.
1. Register at https://mano.is.tue.mpg.de and download `mano_v1_2.zip` (175 MB). The file you need is `mano_v1_2/models/MANO_RIGHT.pkl` (3,821,356 B).
2. Copy it, under the name `MANO_RIGHT_orig.pkl`, to `third_party/WiLoR/mano_data/`.
3. Convert it to a chumpy-free file. The official pickle stores `shapedirs` as a chumpy object and chumpy is not installed (it is unmaintained);
   `perception/tools/strip_chumpy.py` reads `MANO_RIGHT_orig.pkl` and writes `MANO_RIGHT.pkl` next to it, and refuses to write if any array is not
   one of the two chumpy classes it understands (`Ch`, `Select`) or has an unexpected shape. Run it from the repo root:
```bash
unzip -j mano_v1_2.zip mano_v1_2/models/MANO_RIGHT.pkl -d /tmp/mano
cp /tmp/mano/MANO_RIGHT.pkl third_party/WiLoR/mano_data/MANO_RIGHT_orig.pkl
$P perception/tools/strip_chumpy.py
```
(The `unzip` and `cp` lines are mine **(unverified)**; the log copied the file by hand. The conversion was checked once against chumpy 0.71 itself: all
13 keys equal, dtype included. Expected sha256 here: `MANO_RIGHT_orig.pkl` `45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767`,
converted `MANO_RIGHT.pkl` `ed3ac9bfe93c48319616e2d38010afc802405140b23079320680174652286830`.)

## 6. Check the install
```bash
$P -c "import torch; print(torch.__version__, torch.cuda.is_available())"        # 2.11.0+cu128 True (log)
$P tests/test_geometry.py                                                         # upright <-> original image maths (run: passes)
$S tests/test_retarget_grasp.py                                                   # grasp mapping (needs HERO; not run for this doc)
$S -m sim.m1_reach_test --heights 0.74 --out /tmp/teach_g1/reach                  # G1 stands and reaches (README)
```
The raw clips are committed where the code looks for them (`perception/common.py` has `RAW = data/raw`). WiLoR / SAM 2 smoke test:
`$P -m perception.frames --clip 4e103843db`, then `$P -m perception.wilor_smoke` **(unverified as a sequence)**. The full pipeline is
`$P -m perception.run_clip` (README); re-running it from a fresh clone was not tested. That the committed clips parse like the
originals (same rows, same frame alignment) was checked in a fresh clone: `results/final/raw_clips_check.json`.

## 7. Videos (MuJoCo rendering)
Only the commands that write video render (for example `sim.m4_execute videos`, `sim.m1_pick_mug videos`, `retarget.run_m3 videos`). They need
offscreen EGL, and `sim.m4_execute` refuses to start without both variables, set on the command line:
```bash
MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json $S -m sim.m4_execute videos
```
`50_mesa.json` makes EGL use Mesa (on this laptop the Intel iGPU) so rendering stays off the NVIDIA GPU, which perception may be using; the
libEGL warning about PCI id 10de:2d59 is Mesa skipping the NVIDIA card (log). On this Ubuntu the file belongs to the package `libegl-mesa0`
(`dpkg -S`). On another machine point the variable at a vendor file listed in `/usr/share/glvnd/egl_vendor.d/` **(unverified)**.
The `.mp4` files are gitignored (`*.mp4` in `.gitignore`); GIFs are not.

## 8. Licenses (none of these are redistributed in this repo)
- HERO: MIT (`third_party/HERO/LICENSE`).
- WiLoR code and models (`detector.pt`, `wilor_final.ckpt`): CC BY-NC-ND 4.0 (`third_party/WiLoR/license.txt`); WiLoR's README also points to
  the Ultralytics and MANO licenses.
- MANO: non-commercial license, https://mano.is.tue.mpg.de/license.html.
- SAM 2 code: Apache 2.0 (`third_party/sam2/LICENSE`); the checkpoint is downloaded from Meta.

## 9. Not covered
- `hoi-policy` (LeRobot, Python 3.12) from CLAUDE.md was never created: there is no lock for it in `envs/`. The M5 selector
  (`policy/train_selector.py`) trains in `hoi-perception` (PyTorch) and exports ONNX for `hoi-sim`. The export needs the `onnx`
  package; it was added to `hoi-perception` on 2026-10-31 (onnx 1.23.2, ml_dtypes 0.6.0, protobuf 7.36.2) and is in its lock and
  constraints (log, M5 "Environment change").
