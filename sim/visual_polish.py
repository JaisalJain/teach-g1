"""Track V tools: what the visual polish looks like, and the legacy check R0 with 2 worker processes.

  before-after -> results/visual_polish/before_after.png: the same robot states (my mug, profile mug_v1, trial 0: closed on the
                  mug / lifted / over the coaster) drawn from the new render camera with the OLD visuals (the base commit's
                  sim/, taken with git archive into tmp/) and the NEW ones, side by side, plus before_after.json
  video        -> results/visual_polish/r0_trial00_5s.mp4: R0 trial 0 (M1's oracle, cylinder cup) rendered through the
                  pipeline's own VideoWriter (new camera, 1280 x 720); the 5 s clip runs from 2 s before the hand starts to
                  close to 3 s after, i.e. the grasp (the whole trial is kept in tmp/visual_polish/, gitignored)
  r0           -> python -m sim.m1_grasp_ladder run --rung R0 --tag visual --out results/visual_polish, but with 2 worker
                  processes instead of the ladder's 4 (the task allows 2 CPU processes at a time); then r0_comparison.json:
                  every one of the 20 rows against M1's runs (results/m1_pick*/pick_results.json), B3's R0 and the main
                  checkout's R0 re-checks

Rendering needs MUJOCO_GL=egl and __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json (the NVIDIA GPU stays
free). Run from the repo root: env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m sim.visual_polish <command>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = Path("results/visual_polish")
SCRATCH = REPO / "tmp" / "visual_polish"
STATE_NAMES = ("grasp", "lifted", "place")
STATE_CAPTIONS = {"grasp": "hand closed on the mug", "lifted": "mug lifted 10 cm", "place": "mug over the coaster, before release"}
PANEL = (960, 540)
WINDOW_BEFORE_S, WINDOW_S, VIDEO_FPS = 2.0, 5.0, 25


def need_egl() -> None:
    if not os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES"):
        raise SystemExit("set __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json so rendering stays off the NVIDIA GPU")


# ---------------------------------------------------------------------------------------------------------------- before / after
def capture_states(path: Path) -> dict:
    """Run mug_v1 trial 0 (new tree) and save the robot + mug state at three moments, for ``render_worker``."""
    from hero_isaacsim.paths import CHECKPOINTS_ROOT
    from sim2sim.policy_hero_export import HeroExportPolicy

    from sim.m1_pick_mug import trials
    from sim.pick_place import GRASP_PROFILES, Grasp, PickPlace
    from sim.scene import mug_config
    from sim.visuals import CAMERA, VIDEO_HEIGHT, VIDEO_WIDTH

    task = PickPlace(HeroExportPolicy(CHECKPOINTS_ROOT / "example", stand_flag=0.0), cup_model="mug", cup_mass=mug_config()["mass_kg"],
                     **GRASP_PROFILES["mug_v1"])
    t = trials(Grasp.hero_side_grasp(task.scene.cup_height))[0]           # the legacy grasp's trial 0 (the ladder's layouts)
    states, tick = {}, task._tick

    def recording_tick(phase: str):
        tick(phase)
        d = task.scene.data
        snap = (d.qpos.copy(), d.qvel.copy())
        if phase == "lift":
            states.setdefault("grasp", snap)           # first step of the lift: hand closed, mug still on the table
            states["lifted"] = snap                    # ... and the last step of it: 10 cm up
        elif phase == "settle_place":
            states["place"] = snap
    task._tick = recording_tick
    r = task.run(t["cup_xy"], t["coaster_xy"], cup_yaw=math.radians(t["handle_yaw_deg"]))
    print(f"mug_v1 trial 0: {'SUCCESS' if r['success'] else 'FAIL ' + str(r.get('reason'))}; states captured: {sorted(states)}")
    assert set(states) == set(STATE_NAMES), sorted(states)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, coaster_xy=task.scene.coaster_xy, camera=json.dumps(dict(CAMERA, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)),
             **{f"{k}_qpos": v[0] for k, v in states.items()}, **{f"{k}_qvel": v[1] for k, v in states.items()})
    return dict(trial=t["trial"], cup_xy=[round(float(v), 4) for v in t["cup_xy"]], coaster_xy=[round(float(v), 4) for v in t["coaster_xy"]],
                handle_yaw_deg=round(t["handle_yaw_deg"], 2), success=bool(r["success"]))


def render_worker(root: str, states_path: str, out_dir: str) -> None:
    """Worker process: render the saved states with the ``sim`` package under ``root`` from the new camera (pixels -> PNGs)."""
    sys.path.insert(0, root)
    import imageio.v2 as imageio
    import mujoco
    from hero_isaacsim.paths import CHECKPOINTS_ROOT
    from sim2sim.policy_hero_export import HeroExportPolicy

    from sim.scene import TableScene

    data = np.load(states_path)
    cam = json.loads(str(data["camera"]))
    policy = HeroExportPolicy(CHECKPOINTS_ROOT / "example", stand_flag=0.0)
    scene = TableScene(0.74, policy.kp, policy.kd, policy.effort_limit, policy.default_dof_pos, cup_model="mug", contact_model="grasp")
    m, d = scene.model, scene.data
    has_named = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "grasp_right") >= 0
    m.vis.global_.offwidth, m.vis.global_.offheight = max(m.vis.global_.offwidth, cam["width"]), max(m.vis.global_.offheight, cam["height"])
    renderer = mujoco.Renderer(m, height=cam["height"], width=cam["width"])
    free = mujoco.MjvCamera()                                    # the new camera as a free camera: exists in every model
    free.lookat[:] = cam["lookat"]
    free.distance, free.azimuth, free.elevation = cam["distance"], cam["azimuth"], cam["elevation"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    renderer.update_scene(d, camera=free)
    renderer.render()                                            # the first frame of a renderer differs slightly (shadows): discard it
    info = dict(root=root, named_camera_in_model=has_named, camera_max_abs_pixel_diff_named_vs_free=None)
    for name in STATE_NAMES:
        scene.reset(coaster_xy=data["coaster_xy"])
        d.qpos[:] = data[f"{name}_qpos"]
        d.qvel[:] = data[f"{name}_qvel"]
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=free)
        img = renderer.render().copy()
        imageio.imwrite(out / f"{name}.png", img)
        if has_named:                                            # the model's own camera must frame exactly the same view
            renderer.update_scene(d, camera="grasp_right")
            diff = int(np.abs(renderer.render().astype(int) - img.astype(int)).max())
            info["camera_max_abs_pixel_diff_named_vs_free"] = max(info["camera_max_abs_pixel_diff_named_vs_free"] or 0, diff)
    (out / "info.json").write_text(json.dumps(info))
    renderer.close()


def run_render_worker(root: Path, states: Path, out_dir: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--render-worker", str(root), str(states), str(out_dir)],
                   check=True, env=env, cwd=root)
    return json.loads((out_dir / "info.json").read_text())


def cmd_before_after(args) -> None:
    need_egl()
    from PIL import Image, ImageDraw, ImageFont

    from sim.check_visual_only import git, materialise_base

    t0 = time.perf_counter()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    states = SCRATCH / "states.npz"
    trial = capture_states(states)
    base_root, base_full = materialise_base(args.base)
    infos = {name: run_render_worker(root, states, SCRATCH / name) for name, root in (("before", base_root), ("after", REPO))}
    assert infos["after"]["camera_max_abs_pixel_diff_named_vs_free"] == 0, infos["after"]
    head, bar, pad = ImageFont.load_default(size=30), ImageFont.load_default(size=22), 8
    W, H = PANEL
    top = 48
    sheet = Image.new("RGB", (2 * W + 3 * pad, top + len(STATE_NAMES) * (H + pad) + pad), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)
    for c, (name, title) in enumerate((("before", f"BEFORE  (base commit {base_full[:7]} visuals)"), ("after", "AFTER  (visual polish)"))):
        draw.text((pad + c * (W + pad) + 10, 8), title, font=head, fill=(235, 235, 235))
        for r, state in enumerate(STATE_NAMES):
            panel = Image.open(SCRATCH / name / f"{state}.png").convert("RGB").resize(PANEL, Image.LANCZOS)
            x, y = pad + c * (W + pad), top + r * (H + pad)
            sheet.paste(panel, (x, y))
            draw.rectangle((x, y, x + 10 + 11 * len(STATE_CAPTIONS[state]), y + 30), fill=(20, 20, 20))
            draw.text((x + 6, y + 3), STATE_CAPTIONS[state], font=bar, fill=(240, 240, 240))
    args.out.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out / "before_after.png", optimize=True)
    from sim.visuals import CAMERA, CAMERA_NAME, VIDEO_HEIGHT, VIDEO_WIDTH
    (args.out / "before_after.json").write_text(json.dumps(dict(
        what="the same three robot states of mug_v1 trial 0 (mug_v1 profile, my mug, config mass), rendered from the new camera with the "
             "base commit's visuals (before) and the new ones (after); both sides use the same free-camera numbers",
        base_commit=base_full, after_head=git("rev-parse", "HEAD"), trial=trial, camera=dict(name=CAMERA_NAME, **CAMERA),
        render_size=[VIDEO_WIDTH, VIDEO_HEIGHT], sheet_panel_size=list(PANEL), states=list(STATE_NAMES), render_checks=infos,
        runtime_s=round(time.perf_counter() - t0, 1), peak_rss_mb_parent=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        peak_rss_mb_render_worker=round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024)), indent=2))
    print(f"wrote {args.out / 'before_after.png'} ({sheet.size[0]} x {sheet.size[1]}, {(args.out / 'before_after.png').stat().st_size // 1024} KB); "
          f"named camera == free camera pixel for pixel: {infos['after']['camera_max_abs_pixel_diff_named_vs_free'] == 0}")


# ---------------------------------------------------------------------------------------------------------------- video
def cmd_video(args) -> None:
    need_egl()
    import imageio.v2 as imageio

    from sim.m1_grasp_ladder import b3_rows, fixed_trials, make_task
    from sim.m1_grasp_causes import RESULT_KEYS

    t0 = time.perf_counter()
    task = make_task("R0", {})                                    # R0: M1's cylinder cup, 0.15 kg, legacy settings
    t = fixed_trials(task)[0]
    SCRATCH.mkdir(parents=True, exist_ok=True)
    full = SCRATCH / "r0_trial00_full.mp4"
    r = task.run(t["cup_xy"], t["coaster_xy"], full, cup_yaw=0.0)
    ref = b3_rows("R0")[0]                                        # B3's R0 trial 0 (= M1's): the video run must be the same trial
    same = all(r.get(k) == ref[k] for k in RESULT_KEYS)
    print(f"R0 trial 0: {'SUCCESS' if r['success'] else 'FAIL ' + str(r.get('reason'))}; same result fields as B3's R0 row: {same}")
    # video frames are written on every 2nd control step (n = step index + 1 even): step i -> frame (i + 1) // 2 - 1
    first_close = task.log.phases.index("close")
    j_close = (first_close + 1) // 2 - 1
    j0 = max(0, j_close - int(WINDOW_BEFORE_S * VIDEO_FPS))
    n = int(WINDOW_S * VIDEO_FPS)
    args.out.mkdir(parents=True, exist_ok=True)
    clip = args.out / "r0_trial00_5s.mp4"
    reader = imageio.get_reader(full)
    writer = imageio.get_writer(clip, fps=VIDEO_FPS, quality=7, macro_block_size=8)
    written = 0
    for j, frame in enumerate(reader):
        if j0 <= j < j0 + n:
            writer.append_data(frame)
            written += 1
    writer.close()
    reader.close()
    print(f"wrote {clip} ({written} frames = {written / VIDEO_FPS:.1f} s at {VIDEO_FPS} fps, {frame.shape[1]} x {frame.shape[0]}, "
          f"{clip.stat().st_size // 1024} KB), trial time {j0 * 2 / 50:.1f}-{(j0 + written) * 2 / 50:.1f} s of {task.scene.data.time:.1f} s "
          f"(hand starts to close at {first_close / 50:.1f} s); {time.perf_counter() - t0:.0f} s, peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB")
    assert written == n and same


# ---------------------------------------------------------------------------------------------------------------- R0
def cmd_r0(args) -> None:
    import sim.m1_grasp_ladder as ladder
    from sim.m1_grasp_causes import m1_cylinder_runs

    t0 = time.perf_counter()
    ladder.N_PROCESSES = 2         # the ladder's own default is 4; nothing else of its code is changed
    ladder.cmd_run(argparse.Namespace(rung="R0", tag="visual", out=args.out, trials=ladder.N_TRIALS, overwrite=args.overwrite))
    path = args.out / "runs" / "R0_visual.json"
    res = json.loads(path.read_text())
    rows = res["runs"]
    m1 = m1_cylinder_runs()
    checks = {"results/m1_grasp_fix/b3_runs/R0.json": ladder.same_rows(rows, ladder.b3_rows("R0"), ladder.B3_RUNS / "R0.json", "b3")}
    for name in ("R0.json", "R0_after_l3.json"):                 # the main checkout's own R0 runs (4 processes), if present
        ref = ladder.OUT_DIR / "runs" / name
        if ref.exists():
            checks[str(ref)] = ladder.same_rows(rows, json.loads(ref.read_text())["runs"], ref, "ref")
    per_trial = []
    for r, m in zip(rows, m1):
        vs_m1 = {k: (r[k], m.get(k)) for k in ("success", "contact_closure", "lift_slip_in_hand_cm", "lift_cup_rise_cm",
                                                "cup_off_coaster_center_cm", "cup_tilt_deg")}
        vs_m1 |= dict(cup_final=(r["detail"]["cup_final"], m["cup_final"]), sim_seconds=(r["detail"]["sim_seconds"], m["sim_seconds"]))
        per_trial.append(dict(trial=r["trial"], m1_success=m["success"], visual_success=r["success"],
                              identical_to_m1=all(a == b for a, b in vs_m1.values()),
                              identical_to_b3_r0_all_fields=r["trial"] not in {x["trial"] for x in checks["results/m1_grasp_fix/b3_runs/R0.json"]["differences"]}))
    out = dict(
        question="After the visual polish, is R0 (M1's cylinder oracle, 20 trials) still identical to M1?",
        run_file=str(path), processes=res["processes"], successes=f"{res['summary']['successes']}/{res['summary']['trials']}",
        wilson95=res["summary"]["wilson95"], m1_successes=sum(bool(m["success"]) for m in m1[:len(rows)]),
        trials_identical_to_m1=res["legacy_check"]["vs_m1"]["trials_identical"],
        m1_fields_compared=res["legacy_check"]["vs_m1"]["fields_compared"], vs_m1_differences=res["legacy_check"]["vs_m1"]["differences"],
        full_row_comparisons=checks, per_trial=per_trial,
        all_identical=all(p["identical_to_m1"] and p["identical_to_b3_r0_all_fields"] for p in per_trial) and len(rows) == 20,
        resources=res["resources"], runtime_s=round(time.perf_counter() - t0, 1))
    (args.out / "r0_comparison.json").write_text(json.dumps(out, indent=2))
    print(f"R0 (visual polish): {out['successes']}, identical to M1: {out['trials_identical_to_m1']}/20, identical to B3's R0 (every field): "
          f"{checks['results/m1_grasp_fix/b3_runs/R0.json']['trials_identical']}/20; all identical: {out['all_identical']}; wrote "
          f"{args.out / 'r0_comparison.json'}")
    if not out["all_identical"]:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--render-worker", nargs=3, metavar=("ROOT", "STATES", "OUT"), help="internal: worker mode")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("before-after", "video", "r0"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, default=OUT_DIR)
        if name == "before-after":
            p.add_argument("--base", default="0eca38c")
        if name == "r0":
            p.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.render_worker:
        render_worker(*args.render_worker)
        return
    {"before-after": cmd_before_after, "video": cmd_video, "r0": cmd_r0}[args.cmd](args)


if __name__ == "__main__":
    main()
