"""Extra visual: my hand -> the robot, side by side, both motions hitting the same events at the same moment on screen.

Clip e8825dea53 (45 deg view, my right hand picks my mug and places it on the coaster). LEFT: my demonstration,
results/m2_perception_v3/e8825dea53/overlay.mp4 (the RGB frames with the tracked MANO hand, the fitted mug and the coaster
drawn on them; its top 40 px, perception's own frame / time / phase text, are cropped off). RIGHT: the robot doing the same
task in sim, results/m4b/videos/e8825dea53__b_hybrid__trial00.mp4 (the saved b_hybrid success on M4b layout trial00).

Outputs (results/extra/):
  hand_to_robot.mp4         the side-by-side video (25 fps, one output frame per robot video frame)
  hand_to_robot.gif         the same composition, smaller: the first (panel height, frame step, colours) of GIF_SETTINGS
                            under 5 MB (the robot still at real time: every step-th frame, each shown for step / 25 s)
  hand_to_robot.json        every event time used and its source, the warp segments, the outputs
  hand_to_robot_phases.json the robot's per-control-step phases (run-length) and the row comparison of ``phases``
  hand_to_robot_{approach,grasp,place}.png  three frames of the mp4

Clocks.
  Robot: tau [s] = control steps since the first logged step (the start of the 2 s stand) / 50. Control step j (an index
    into sim/pick_place.py Log.phases) runs from j / 50 to (j + 1) / 50. The robot video writes a frame after every 2nd
    logged step (sim/m1_reach_test.py VideoWriter: frame on call n = 2, 4, ...), so robot video frame f shows the state at
    the end of step 2f + 1, i.e. at tau_f = (2f + 2) / 50 = (f + 1) / 25. Output frame f = robot video frame f, at 25 fps:
    THE ROBOT PLAYS AT REAL TIME, every one of its frames exactly once, nothing re-timed.
  Me: s [s] = t[i] - t[0], the odometry timestamps of data/demos/e8825dea53.npz (never the video clock). overlay.mp4 frame
    k shows odometry index i = k + 1 (index 0 has no image; perception/report.py overlay_video writes i = 1 .. T - 1).

Robot phases. The saved row stores no per-step phases, so ``phases`` re-runs that ONE row once (sim.m4_execute.run_one,
no video) and reads task.log.phases from sim.m4_execute.worker_task(). It passes diagnostics=(), because the default
collision diagnostic builds a new Log (it would replace task.log); this only changes the row fields diagnostics_run and
collision_min. The new row is compared with the saved one by sim.m4_execute.row_differences on every other field (wall_s
and worker_peak_rss_mb are left out by row_differences itself, as in ``rerun``). ``render`` refuses to run unless the
re-run was identical and its step count equals the video's (results/m4b/videos/videos.json control_steps).

Alignment = piecewise-linear time warping of MY video onto the robot's timeline. Knots (human time <- robot time):
  approach  s(a0): a0 = retarget/warp.py approach_start (the CLAUDE.md a0 rule; asserted = the row's path.info.a0)
            <- first "pregrasp" step: the robot's arm starts to move. Its first 1.2 s are the robot-only prelude (arm up from
            its side, path frames 0 .. i_prelude_end), then my approach; the knot is put where both start moving.
  grasp     t_contact (perception S8)                  <- first "close" step (palm at the grasp pose, fingers start closing)
  lift      t_lift                                     <- first "lift" step
  carry     t[lift] + (t[place] - t[lift]) j_peak / n_t <- first "transport" step. The robot's "lift" slice ends at sample
            j_peak of my retargeted transport (its highest point; retarget/paths_v2.py hybrid_path, v2 split), and transport
            sample j is my palm at exactly that human time (retarget/warp.py resample, lift / place = event_frames' nearest
            frames). j_peak and n_t = the two slices' m3_intervals are read from the saved row (n_t re-checked with
            warp.intervals). So this knot is a correspondence of the retargeting, not a guess.
  place     t_place                                    <- first "settle_place" step (= the end of "transport": the palm
            at the place pose, the mug 1 cm above the coaster)
  release   t_release                                  <- first "open" step
  Between two knots my clip time is linear in tau; the frame shown is the image frame nearest to that time (no blending,
  no interpolated images). Before the first knot and after the last, BOTH PLAY AT REAL TIME (slope 1): my video holds its
  first image until it would start, and its last image once it has ended. The robot-only motions (prelude, the 3 s settle
  at the pre-grasp point, the oracle's 2 s final approach, the settles, the open ramp) have no counterpart in my motion,
  so my approach is slowed the most; hand_to_robot.json segments[*].human_speed gives each segment's factor, and the human
  header shows it on screen.

On screen: a label per panel, each panel's own clock (robot tau; my clip time of the frame shown), my playback speed, a
short description of what the robot is doing (its phase label; "(robot only)" where that motion is not from my hand), and
a phase strip (approach / grasp / lift / carry / place / release) that switches at the knots (the strip shows the phase of
the robot step that produced the frame, i.e. knot tau_k < tau_f).

Text is drawn with PIL in DejaVu Sans (matplotlib's bundled copy). Encoding: ffmpeg (libx264 CRF 20 for the mp4; a palette
GIF with palettegen stats_mode=diff / paletteuse bayer + diff_mode=rectangle, as sim/m4_execute.py make_gif, but with a
128-colour palette: the camera texture of my video makes a 256-colour GIF of 22 s too big; 64 colours washed out the
phase strip's accent colour in a test, 128 kept it).

Run from the repo root, CPU only:
  env -u PYTHONPATH nice -n 19 conda run --no-capture-output -n hoi-sim python -m sim.hand_to_robot_video phases
  env -u PYTHONPATH nice -n 19 conda run --no-capture-output -n hoi-sim python -m sim.hand_to_robot_video render
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLIP, VARIANT, LAYOUT = "e8825dea53", "b_hybrid", "trial00"
RUN = f"{CLIP}__{VARIANT}__{LAYOUT}"
HUMAN_VIDEO = Path(f"results/m2_perception_v3/{CLIP}/overlay.mp4")
ROBOT_VIDEO = Path(f"results/m4b/videos/{RUN}.mp4")
VIDEOS_JSON = Path("results/m4b/videos/videos.json")
SAVED_ROW = Path(f"results/m4b/runs/{RUN}.json")
DEMO = Path(f"data/demos/{CLIP}.npz")
OUT = Path("results/extra")
PHASES_JSON = OUT / "hand_to_robot_phases.json"

FPS = 50                      # control rate
VIDEO_FPS = 25                # robot video: one frame per 2 control steps
NOT_COMPARED = ("diagnostics_run", "collision_min")       # the only row fields diagnostics=() changes
ROBOT_SEQUENCE = ("stand", "pregrasp", "settle_pregrasp", "approach", "settle_grasp", "close", "lift", "transport",
                  "settle_place", "open", "retreat")     # Execute._object_aware with hybrid=True, in order, each once
# caption phase <- (my event, the robot phase whose first step is the knot)
KNOTS = (("approach", "a0", "pregrasp"), ("grasp", "t_contact", "close"), ("lift", "t_lift", "lift"),
         ("carry", "transport peak", "transport"), ("place", "t_place", "settle_place"), ("release", "t_release", "open"))
ROBOT_DOING = {"stand": "standing", "prelude": "arm up from its side (robot only)", "pregrasp": "following my approach",
               "settle_pregrasp": "settling at the pre-grasp point (robot only)", "approach": "final 8 cm approach (robot only)",
               "settle_grasp": "settling at the grasp pose (robot only)", "close": "closing the fingers until touch",
               "lift": "following my lift", "transport": "following my carry", "settle_place": "settling at the place pose (robot only)",
               "open": "opening the hand (robot only)", "retreat": "retreat (robot only)"}
HUMAN_CROP_TOP = 40           # px of overlay.mp4 (720 x 960) holding perception's debug text
STILLS = (("approach", 0.3), ("grasp", 0.75), ("place", 0.5))     # (phase, fraction of the phase's robot time); approach 0.3 = while
                                                                   # the robot follows my approach path
GIF_SETTINGS = ((270, 3, 128), (240, 3, 128), (240, 4, 128), (200, 4, 128))   # (panel height, frame step, palette colours)
GIF_MAX_BYTES = 5_000_000
MP4_PANEL_H = 720

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
BG, FG, DIM = (22, 22, 26), (240, 240, 240), (165, 165, 172)
PILL_TODO, PILL_DONE, PILL_NOW = ((48, 48, 56), (125, 125, 135)), ((70, 70, 80), (205, 205, 210)), ((242, 172, 44), (20, 20, 20))


# ---------------------------------------------------------------------------------------------------------------- phases
def run_lengths(labels: list[str]) -> list[dict]:
    out = []
    for j, p in enumerate(labels):
        if out and out[-1]["phase"] == p:
            out[-1]["last_step"] = j
        else:
            out.append(dict(phase=p, first_step=j, last_step=j))
    for r in out:
        r.update(steps=r["last_step"] - r["first_step"] + 1, t_start_s=r["first_step"] / FPS, t_end_s=(r["last_step"] + 1) / FPS)
    return out


def cmd_phases(args) -> None:
    from sim import m4_execute as m4
    from sim.m1_grasp_ladder import peak_rss_mb

    saved = json.loads(SAVED_ROW.read_text())
    t0 = time.perf_counter()
    again = m4.run_one(saved["clip"], saved["variant"], saved["layout"], paths=saved["paths"], diagnostics=(), **m4.m5_kwargs(saved))
    labels = list(m4.worker_task().log.phases)
    raw = m4.row_differences(json.loads(json.dumps(again)), saved)
    skipped = [d for d in raw if d.split(".")[0] in NOT_COMPARED]
    differing = [d for d in raw if d not in skipped]
    video = next(v for v in json.loads(VIDEOS_JSON.read_text())["videos"] if v["run"] == RUN)
    res = dict(
        run=RUN, saved_row=str(SAVED_ROW),
        call=f'sim.m4_execute.run_one("{CLIP}", "{VARIANT}", "{LAYOUT}", paths="{saved["paths"]}", diagnostics=()), then '
             "sim.m4_execute.worker_task().log.phases",
        compared="sim.m4_execute.row_differences(new row, saved row): every field, recursively, except wall_s, worker_peak_rss_mb "
                 "(left out by row_differences) and " + ", ".join(NOT_COMPARED) + " (diagnostics=() drops the collision diagnostic)",
        identical=not differing, fields_differing=differing, not_compared_differences=skipped,
        saved_success=saved["success"], rerun_success=again["success"], saved_category=saved["category"], rerun_category=again["category"],
        control_steps=len(labels), video_control_steps=video["control_steps"], video_frames_in_file=video["frames_in_file"],
        steps_match_video=len(labels) == video["control_steps"] and len(labels) // 2 == video["frames_in_file"],
        phase_runs=run_lengths(labels), wall_s=round(time.perf_counter() - t0, 1), peak_rss_mb=peak_rss_mb())
    OUT.mkdir(parents=True, exist_ok=True)
    PHASES_JSON.write_text(json.dumps(res, indent=2))
    print(f"{RUN}: re-run identical {res['identical']} (differing: {differing}; not compared: {skipped}); success saved "
          f"{res['saved_success']} / re-run {res['rerun_success']}; {len(labels)} steps (video {video['control_steps']}); "
          f"{res['wall_s']} s -> {PHASES_JSON}")
    for r in res["phase_runs"]:
        print(f"  {r['phase']:<16} steps {r['first_step']:4d}-{r['last_step']:4d}  t {r['t_start_s']:6.2f}-{r['t_end_s']:6.2f} s")
    if differing or res["saved_success"] != res["rerun_success"]:
        raise SystemExit(1)


# ---------------------------------------------------------------------------------------------------------------- alignment
def human_events(saved: dict) -> list[dict]:
    """My six knot times (clip time s = t - t[0]) with their sources."""
    from retarget.frames import load_demo
    from retarget.warp import approach_start, event_frames, intervals

    demo = load_demo(CLIP)
    t = demo["t"]
    ev = event_frames(demo)
    a0 = approach_start(demo, ev["g"])
    info = saved["detail"]["path"]["info"]
    assert a0 == info["a0"], (a0, info["a0"])
    v2 = info["v2"]
    seg = {s["name"]: s for s in v2["segments"]}
    j_peak, n_t = v2["j_peak"], seg["transport1"]["m3_intervals"] + seg["transport2"]["m3_intervals"]
    assert seg["transport1"]["m3_intervals"] == j_peak and n_t == intervals(info["k"], t, ev["lift"], ev["place"]), (j_peak, n_t)
    t_peak = t[ev["lift"]] + (t[ev["place"]] - t[ev["lift"]]) * j_peak / n_t
    out = [dict(s=float(t[a0] - t[0]), frame=a0, source=f"retarget/warp.py approach_start (a0 rule) = odometry index {a0}; "
                                                         f"= saved row detail.path.info.a0")]
    for name in ("contact", "lift"):
        out.append(dict(s=float(demo[f"t_{name}"] - t[0]), frame=ev[name], source=f"{DEMO} t_{name} - t[0] (nearest index {ev[name]})"))
    out.append(dict(s=float(t_peak - t[0]), frame=int(np.argmin(np.abs(t - t_peak))),
                    source=f"t[lift {ev['lift']}] + (t[place {ev['place']}] - t[lift]) * j_peak / n_t, j_peak = {j_peak}, n_t = {n_t} "
                           "(saved row detail.path.info.v2: j_peak, transport1 + transport2 m3_intervals; n_t = warp.intervals(k, t, lift, place))"))
    for name in ("place", "release"):
        out.append(dict(s=float(demo[f"t_{name}"] - t[0]), frame=ev[name], source=f"{DEMO} t_{name} - t[0] (nearest index {ev[name]})"))
    return out


def robot_events(phases: dict) -> tuple[list[dict], list[str]]:
    runs = phases["phase_runs"]
    assert tuple(r["phase"] for r in runs) == ROBOT_SEQUENCE, [r["phase"] for r in runs]
    first = {r["phase"]: r["first_step"] for r in runs}
    labels = [r["phase"] for r in runs for _ in range(r["steps"])]
    return [dict(tau=first[p] / FPS, step=first[p], source=f'first "{p}" step ({PHASES_JSON} phase_runs)') for _, _, p in KNOTS], labels


def warp(tau: float, k_tau: np.ndarray, k_s: np.ndarray) -> tuple[float, float]:
    """My clip time at robot time tau, and the slope there (piecewise linear between knots, slope 1 outside)."""
    if tau <= k_tau[0]:
        return float(k_s[0] + tau - k_tau[0]), 1.0
    if tau >= k_tau[-1]:
        return float(k_s[-1] + tau - k_tau[-1]), 1.0
    j = int(np.searchsorted(k_tau, tau, side="right")) - 1
    slope = (k_s[j + 1] - k_s[j]) / (k_tau[j + 1] - k_tau[j])
    return float(k_s[j] + slope * (tau - k_tau[j])), float(slope)


def timeline(n_frames: int, k_tau, k_s, t_img: np.ndarray, labels: list[str], i_prelude_end: int) -> list[dict]:
    """Per output frame f: robot time, my time, my image index, the caption phase and what the robot is doing."""
    first_pregrasp = labels.index("pregrasp")
    rows = []
    for f in range(n_frames):
        tau = (f + 1) / VIDEO_FPS
        s, slope = warp(tau, k_tau, k_s)
        i = 1 + int(np.argmin(np.abs(t_img[1:] - s)))        # nearest image frame (index 0 has no image)
        held = s < t_img[1] or s > t_img[-1]
        step = 2 * f + 1
        doing = labels[step]
        if doing == "pregrasp" and step <= first_pregrasp + i_prelude_end:
            doing = "prelude"
        phase = int(np.sum(np.asarray(k_tau) < tau)) - 1      # -1 = before the first knot
        rows.append(dict(f=f, tau=tau, s=s, slope=slope, held=held, i=i, step=step, doing=doing, phase=phase))
    return rows


# ---------------------------------------------------------------------------------------------------------------- drawing
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), max(6, int(size)))


def fitted(text: str, size: int, max_w: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    """The largest font <= size whose rendering of text is at most max_w wide."""
    while size > 6 and font(size, bold).getlength(text) > max_w:
        size -= 1
    return font(size, bold)


class Layout:
    """Pixel layout for a panel height H; ``text`` scales the fonts relative to H (the GIF uses larger relative text)."""

    def __init__(self, H: int, text: float = 1.0):
        u = H / 720
        self.H = H
        self.pad = max(6, round(12 * u))
        self.wh = round(H * 720 / (960 - HUMAN_CROP_TOP))
        self.wr = round(H * 1280 / 720)
        self.label_px, self.sub_px, self.pill_px, self.cap_px = (round(v * u * text) for v in (30, 19, 22, 17))
        self.header = self.pad + self.label_px + round(0.45 * self.pad) + self.sub_px + self.pad
        self.pill_h = round(self.pill_px * 1.7)
        self.footer = self.pad + self.pill_h + round(0.6 * self.pad) + self.cap_px + self.pad
        w = self.pad + self.wh + self.pad + self.wr + self.pad
        h = self.header + H + self.footer
        self.W, self.Htot = w + w % 2, h + h % 2
        self.xh, self.xr, self.y = self.pad, 2 * self.pad + self.wh, self.header


def compose(lay: Layout, human: np.ndarray, robot: np.ndarray, row: dict) -> Image.Image:
    img = Image.new("RGB", (lay.W, lay.Htot), BG)
    hum = Image.fromarray(human[HUMAN_CROP_TOP:]).resize((lay.wh, lay.H), Image.LANCZOS)
    rob = Image.fromarray(robot)
    if rob.size != (lay.wr, lay.H):
        rob = rob.resize((lay.wr, lay.H), Image.LANCZOS)
    img.paste(hum, (lay.xh, lay.y))
    img.paste(rob, (lay.xr, lay.y))
    d = ImageDraw.Draw(img)

    # headers: label + sub-label (clock, speed / what the robot does)
    y1, y2 = lay.pad, lay.pad + lay.label_px + round(0.45 * lay.pad)
    labels = ((lay.xh, lay.wh, "my hand (iPhone, 45° view)"), (lay.xr, lay.wr, "Unitree G1 + HERO in MuJoCo"))
    f_label = min((fitted(label, lay.label_px, w, bold=True) for _, w, label in labels), key=lambda f: f.size)   # same size for both
    for x, _, label in labels:
        d.text((x, y1), label, font=f_label, fill=FG)
    speed = "held" if row["held"] else ("real time" if row["slope"] == 1.0 else f"time-warped, {row['slope']:.2f}x speed")
    hsub = f"t = {row['s_img']:.2f} s   {speed}"
    d.text((lay.xh, y2), hsub, font=fitted(hsub, lay.sub_px, lay.wh), fill=DIM)
    rsub = f"t = {row['tau']:5.2f} s   real time   ·   {ROBOT_DOING[row['doing']]}"
    d.text((lay.xr, y2), rsub, font=fitted(rsub, lay.sub_px, lay.wr), fill=DIM)

    # footer: phase strip, then the caption line
    f_pill = font(lay.pill_px, bold=True)
    names = [k[0] for k in KNOTS]
    gap, padx = round(0.8 * lay.pad), round(0.9 * lay.pill_px)
    widths = [f_pill.getlength(n) + 2 * padx for n in names]
    x = (lay.W - (sum(widths) + gap * (len(names) - 1))) / 2
    yp = lay.y + lay.H + lay.pad
    for k, (n, w) in enumerate(zip(names, widths)):
        fill, ink = PILL_NOW if k == row["phase"] else (PILL_DONE if k < row["phase"] else PILL_TODO)
        d.rounded_rectangle((x, yp, x + w, yp + lay.pill_h), radius=lay.pill_h // 2, fill=fill)
        d.text((x + w / 2, yp + lay.pill_h / 2), n, font=f_pill, fill=ink, anchor="mm")
        x += w + gap
    cap = "robot: real time   ·   my hand: time-warped (piecewise linear) so both reach each phase change at the same moment"
    f_cap = fitted(cap, lay.cap_px, lay.W - 2 * lay.pad)
    d.text((lay.W / 2, yp + lay.pill_h + round(0.6 * lay.pad)), cap, font=f_cap, fill=DIM, anchor="ma")
    return img


def frames(rows: list[dict], lay: Layout, step: int = 1):
    """Composed frames for output frames 0, step, 2 step, ... (decodes both videos forward once)."""
    hum_it, rob_it = iter(imageio.get_reader(HUMAN_VIDEO)), iter(imageio.get_reader(ROBOT_VIDEO))
    hk, himg = -1, None
    for row in rows:
        robot = next(rob_it)
        if row["f"] % step:
            continue
        k = row["i"] - 1                      # overlay.mp4 frame k = odometry index k + 1
        assert k >= hk, "my frame index went backwards"
        while hk < k:
            himg, hk = next(hum_it), hk + 1
        yield row, compose(lay, himg, robot, row)


def ffmpeg_writer(out: Path, size: tuple[int, int], fps: str, args: list[str]) -> subprocess.Popen:
    return subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
                             "-r", fps, "-i", "-", "-threads", "4", *args, str(out)], stdin=subprocess.PIPE)


def write(proc: subprocess.Popen, what: str) -> None:
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {what}")


def probe(path: Path) -> dict:
    s = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                   "stream=nb_read_frames,width,height,r_frame_rate", "-of", "json", str(path)],
                                  check=True, capture_output=True, text=True).stdout)["streams"][0]
    return dict(file=str(path), frames=int(s["nb_read_frames"]), width=int(s["width"]), height=int(s["height"]), fps=s["r_frame_rate"],
                size_bytes=path.stat().st_size)


# ---------------------------------------------------------------------------------------------------------------- render
def cmd_render(args) -> None:
    t0 = time.perf_counter()
    phases = json.loads(PHASES_JSON.read_text())
    if not (phases["identical"] and phases["steps_match_video"] and phases["rerun_success"]):
        raise SystemExit(f"{PHASES_JSON}: the re-run is not identical to the saved row / the video; run `phases` again")
    saved = json.loads(SAVED_ROW.read_text())
    hum, (rob, labels) = human_events(saved), robot_events(phases)
    k_tau, k_s = np.array([r["tau"] for r in rob]), np.array([h["s"] for h in hum])
    assert np.all(np.diff(k_tau) > 0) and np.all(np.diff(k_s) > 0), (k_tau, k_s)

    with np.load(DEMO, allow_pickle=False) as d:
        t_img = d["t"] - d["t"][0]
    n_frames = phases["video_frames_in_file"]
    rows = timeline(n_frames, k_tau, k_s, t_img, labels, saved["detail"]["path"]["i_prelude_end"])
    for r in rows:
        r["s_img"] = float(t_img[r["i"]])
    assert all(b["i"] >= a["i"] for a, b in zip(rows, rows[1:]))

    # stills: a fraction into a phase (by robot time)
    bounds = list(k_tau) + [n_frames / VIDEO_FPS]
    names = [k[0] for k in KNOTS]
    still_frames = {}
    for name, frac in STILLS:
        j = names.index(name)
        tau = bounds[j] + frac * (bounds[j + 1] - bounds[j])
        still_frames[int(round(tau * VIDEO_FPS)) - 1] = name

    OUT.mkdir(parents=True, exist_ok=True)
    lay = Layout(MP4_PANEL_H)
    mp4 = OUT / "hand_to_robot.mp4"
    proc = ffmpeg_writer(mp4, (lay.W, lay.Htot), str(VIDEO_FPS), ["-c:v", "libx264", "-crf", "20", "-preset", "medium",
                                                                   "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    n_written, stills = 0, []
    for row, img in frames(rows, lay):
        proc.stdin.write(img.tobytes())
        n_written += 1
        if row["f"] in still_frames:
            p = OUT / f"hand_to_robot_{still_frames[row['f']]}.png"
            img.save(p)
            stills.append(dict(phase=still_frames[row["f"]], file=str(p), output_frame=row["f"], robot_t_s=row["tau"],
                               my_t_s=row["s_img"], my_odometry_index=row["i"]))
    write(proc, mp4)
    assert n_written == n_frames
    mp4_info = probe(mp4)

    gif, tried = OUT / "hand_to_robot.gif", []
    for panel_h, step, colors in GIF_SETTINGS:
        glay = Layout(panel_h, text=1.5)
        fps = f"{VIDEO_FPS}/{step}"
        proc = ffmpeg_writer(gif, (glay.W, glay.Htot), fps, [
            "-vf", f"split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
            "-loop", "0"])
        for _, img in frames(rows, glay, step):
            proc.stdin.write(img.tobytes())
        write(proc, gif)
        tried.append(dict(panel_height=panel_h, frame_step=step, colors=colors, fps=VIDEO_FPS / step, width=glay.W, height=glay.Htot,
                          size_bytes=gif.stat().st_size))
        print(f"GIF panel {panel_h} px, every {step} frames, {colors} colours: {tried[-1]['size_bytes'] / 1e6:.2f} MB", flush=True)
        if tried[-1]["size_bytes"] < GIF_MAX_BYTES:
            break
    else:
        gif.unlink()
        raise SystemExit(f"no GIF setting under {GIF_MAX_BYTES} bytes: {tried}")

    # the record
    events = []
    for (name, h_name, r_name), h, r in zip(KNOTS, hum, rob):
        events.append(dict(phase_starts=name, human_event=h_name, human_t_s=round(h["s"], 4), human_odometry_index=h["frame"],
                           human_source=h["source"], robot_event=f'start of "{r_name}"', robot_t_s=r["tau"], robot_step=r["step"],
                           robot_source=r["source"]))
    segments = [dict(segment=f"{names[j]} -> {names[j + 1]}", human_s=round(k_s[j + 1] - k_s[j], 4), robot_s=round(k_tau[j + 1] - k_tau[j], 4),
                     human_speed=round((k_s[j + 1] - k_s[j]) / (k_tau[j + 1] - k_tau[j]), 4)) for j in range(len(KNOTS) - 1)]
    held = [r for r in rows if r["held"]]
    res = dict(
        description="side-by-side my hand (left, time-warped) -> robot (right, real time); see sim/hand_to_robot_video.py docstring",
        inputs=dict(human_video=str(HUMAN_VIDEO), human_video_crop=f"top {HUMAN_CROP_TOP} px removed (perception's debug text)",
                    robot_video=str(ROBOT_VIDEO), robot_video_note="results/m4b/videos/videos.json: re-rendered run identical to the saved row",
                    demo=str(DEMO), saved_row=str(SAVED_ROW), robot_phases=str(PHASES_JSON)),
        rerun_comparison={k: phases[k] for k in ("call", "compared", "identical", "fields_differing", "not_compared_differences",
                                                 "saved_success", "rerun_success", "control_steps", "video_control_steps", "steps_match_video")},
        clocks=dict(robot="tau = control steps since the first logged step / 50 s; robot video frame f (= output frame f) shows the state "
                          "at the end of step 2f + 1, tau = (f + 1) / 25 s",
                    human="s = t[i] - t[0] (odometry timestamps); overlay.mp4 frame k = odometry index k + 1"),
        warp="human time piecewise linear in robot time between the knots below; nearest image frame shown (no blending)",
        outside_knots="before the first knot and after the last: both at real time (slope 1); my video holds its first image before it "
                      "starts and its last image after it ends",
        events=events, segments=segments,
        before_first_knot=dict(robot_s=float(k_tau[0]), human_s=float(k_s[0]), human_held_first_image_until_robot_t_s=round(float(k_tau[0] - k_s[0] + t_img[1]), 4)),
        after_last_knot=dict(robot_s_to_run_end=round(phases["control_steps"] / FPS - k_tau[-1], 4),
                             robot_s_to_last_video_frame=round(rows[-1]["tau"] - k_tau[-1], 4), human_s=round(float(t_img[-1] - k_s[-1]), 4),
                             human_holds_last_image_from_robot_t_s=round(float(k_tau[-1] + t_img[-1] - k_s[-1]), 4)),
        frames_with_held_human_image=len(held),
        human_index_per_output_frame=[r["i"] for r in rows],
        caption_phase_rule="the phase of the robot step that produced the frame: last knot with tau_k < tau_f (none before the first knot)",
        outputs=dict(mp4=dict(**mp4_info, panel_height=lay.H, layout=dict(human_panel=[lay.wh, lay.H], robot_panel=[lay.wr, lay.H])),
                     gif=dict(file=str(gif), chosen=tried[-1], tried=tried, max_bytes=GIF_MAX_BYTES, frames=probe(gif)["frames"],
                              real_time_robot=True),
                     stills=stills),
        tool=subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0],
        wall_s=round(time.perf_counter() - t0, 1))
    (OUT / "hand_to_robot.json").write_text(json.dumps(res, indent=2))
    print(f"{mp4} {mp4_info['size_bytes'] / 1e6:.2f} MB, {gif} {tried[-1]['size_bytes'] / 1e6:.2f} MB, {len(stills)} stills; "
          f"{res['wall_s']} s")
    for e in events:
        print(f"  {e['phase_starts']:<8} human {e['human_t_s']:6.3f} s  <-  robot {e['robot_t_s']:6.2f} s ({e['robot_event']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("phases", help="re-run the saved row once (no video), save its per-step phases and the row comparison")
    sub.add_parser("render", help="build the mp4, the GIF, the stills and hand_to_robot.json")
    args = ap.parse_args()
    {"phases": cmd_phases, "render": cmd_render}[args.cmd](args)


if __name__ == "__main__":
    main()
