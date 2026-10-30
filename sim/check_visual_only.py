"""Proof that the visual polish (sim/visuals.py) changed no physics.

It compiles the scene twice, in two separate processes so the two ``sim`` packages never clash:
  before = the base commit's ``sim/`` and ``config/`` (``--base``, default 0eca38c; taken with ``git archive`` into tmp/),
  after  = this working tree,
once for every cup model (cylinder, mug, mug_body) x contact model (legacy, grasp). Each compile dumps the model's arrays and
the two sets are compared element for element (``np.array_equal``, no tolerance):
  - every body: body_mass, body_inertia, body_pos, body_quat (the four the spec names), plus body_ipos, body_iquat, body_subtreemass;
  - every NON-visual geom: geom_size, geom_pos, geom_quat, geom_friction, geom_contype, geom_conaffinity, geom_condim, geom_solref,
    geom_solimp (the spec's list), plus every other geom_* array except the two that are the colour (geom_rgba, geom_matid);
  - every field of ``opt`` (the solver options: timestep, cone, impratio, noslip, tolerance, ...);
  - extras, not in the spec's list: joints, dofs (dof_length included), sites (all but the colour), sensors, actuators, equalities,
    explicit contact pairs (by geom name), meshes, and the mass-related ``stat`` values;
  - dynamics: both models simulated for 1 s (50 control steps, HERO's PD toward the default pose, the cup falling to the table);
    qpos, qvel and the contact count must be bit-identical.
Geoms (and sites) are matched by order and checked by (name, body) sequence. Anything the polish adds (named "*_visual", only in
"after") is excluded from the comparison and listed; an added geom would have to have contype = conaffinity = 0, and the cup body's
mass / inertia must not move (they are in the comparison). The white mug interior is added as a site, so no geom is added at all.

Run from the repo root (needs the hoi-sim env; nothing renders):
  env -u PYTHONPATH conda run --no-capture-output -n hoi-sim python -m sim.check_visual_only [--base 0eca38c]
Writes results/visual_polish/check_visual_only.json and exits 1 if anything differs.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "0eca38c"
OUT_DIR = Path("results/visual_polish")
CUP_MODELS = ("cylinder", "mug", "mug_body")
CONTACT_MODELS = ("legacy", "grasp")
TABLE_HEIGHT = 0.74
DYNAMICS_STEPS = 50

BODY_SPEC = ("body_mass", "body_inertia", "body_pos", "body_quat")
GEOM_SPEC = ("geom_size", "geom_pos", "geom_quat", "geom_friction", "geom_contype", "geom_conaffinity", "geom_condim",
             "geom_solref", "geom_solimp")
GEOM_VISUAL = ("geom_rgba", "geom_matid")              # the only geom arrays that are colour, not physics
SITE_VISUAL = ("site_rgba", "site_matid")
BODY_EXTRA = ("body_ipos", "body_iquat", "body_subtreemass")
EXTRA_PREFIXES = ("jnt_", "dof_", "sensor_", "actuator_", "eq_", "tendon_", "mesh_", "exclude_")
EXTRA_SCALARS = ("nq", "nv", "nu", "nbody", "njnt", "nsensor", "nmesh", "npair")
PAIR_FIELDS = ("pair_dim", "pair_signature", "pair_solref", "pair_solreffriction", "pair_solimp", "pair_margin", "pair_gap", "pair_friction")
STAT_PHYSICS = ("meaninertia", "meanmass")


# ---------------------------------------------------------------------------------------------------------------- base tree
def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=True).stdout.strip()


def materialise_base(commit: str) -> tuple[Path, str]:
    """The commit's ``sim/`` and ``config/`` extracted to tmp/base_<short>/ (read-only use). Returns the tree and the full hash."""
    full = git("rev-parse", "--verify", f"{commit}^{{commit}}")
    dest = REPO / "tmp" / f"base_{full[:10]}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    archive = subprocess.run(["git", "-C", str(REPO), "archive", full, "sim", "config"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive, check=True)
    return dest, full


# ---------------------------------------------------------------------------------------------------------------- the worker
def dump_model(root: str, out: str) -> None:
    """Worker process: build every TableScene of ``root``'s ``sim`` and save its arrays to ``out`` (an .npz)."""
    sys.path.insert(0, root)
    import mujoco
    from hero_isaacsim.paths import CHECKPOINTS_ROOT
    from sim2sim.policy_hero_export import HeroExportPolicy

    from sim import scene as S

    assert Path(S.__file__).resolve().is_relative_to(Path(root).resolve()), f"imported {S.__file__}, not a sim under {root}"
    assert tuple(S.CUP_MODELS) == CUP_MODELS and tuple(S.CONTACT_MODELS) == CONTACT_MODELS, "the check's model lists are out of date"
    policy = HeroExportPolicy(CHECKPOINTS_ROOT / "example", stand_flag=0.0)
    name = lambda m, kind, i: mujoco.mj_id2name(m, kind, int(i)) or ""
    saved = {}
    for cup in CUP_MODELS:
        for contact in CONTACT_MODELS:
            scene = S.TableScene(TABLE_HEIGHT, policy.kp, policy.kd, policy.effort_limit, policy.default_dof_pos, cup_model=cup,
                                 contact_model=contact)
            m = scene.model
            put = lambda field, value: saved.__setitem__(f"{cup}|{contact}|{field}", np.asarray(value))
            for field in dir(m):
                if field.startswith(("body_", "geom_", "site_", "pair_") + EXTRA_PREFIXES) and isinstance(getattr(m, field), np.ndarray):
                    put(field, getattr(m, field))
            put("qpos0", m.qpos0)
            put("qpos_spring", m.qpos_spring)
            for field in EXTRA_SCALARS:
                put(field, getattr(m, field))
            put("ngeom", m.ngeom)
            put("nsite", m.nsite)
            put("counts_visual", [m.ncam, m.nlight, m.ntex, m.nmat])
            for field in STAT_PHYSICS:
                put(f"stat.{field}", getattr(m.stat, field))
            for field in dir(m.opt):
                value = getattr(m.opt, field)
                if not field.startswith("_") and not callable(value):
                    put(f"opt.{field}", value)
            put("geom_names", [name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)])
            put("geom_body_names", [name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) for g in range(m.ngeom)])
            put("site_names", [name(m, mujoco.mjtObj.mjOBJ_SITE, g) for g in range(m.nsite)])
            put("site_body_names", [name(m, mujoco.mjtObj.mjOBJ_BODY, m.site_bodyid[g]) for g in range(m.nsite)])
            put("body_names", [name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)])
            put("pair_geom_names", [(name(m, mujoco.mjtObj.mjOBJ_GEOM, a), name(m, mujoco.mjtObj.mjOBJ_GEOM, b))
                                    for a, b in zip(m.pair_geom1, m.pair_geom2)])
            # dynamics: 1 s of control steps on the default pose
            target = np.asarray(policy.default_dof_pos, dtype=float).reshape(29)
            for _ in range(DYNAMICS_STEPS):
                scene.step(target)
            put("dyn_qpos", scene.data.qpos)
            put("dyn_qvel", scene.data.qvel)
            put("dyn_ncon", scene.data.ncon)
            print(f"  dumped {cup} / {contact}: {m.ngeom} geoms", file=sys.stderr, flush=True)
    np.savez_compressed(out, **saved)


def run_worker(root: Path, out: Path) -> float:
    t0 = time.perf_counter()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--dump", str(root), str(out)], check=True, env=env, cwd=root)
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------------------------------------------- the comparison
def same(a: np.ndarray, b: np.ndarray) -> bool:
    equal_nan = a.dtype.kind == "f"
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=equal_nan))


def compare(before, after, key: str) -> dict:
    """One (cup, contact) combination: every comparison as {name: dict(passed, n, ...)}, plus the geoms / sites only 'after' has."""
    b, a = (lambda f, d=before: d[f"{key}|{f}"]), (lambda f, d=after: d[f"{key}|{f}"])
    results: dict[str, dict] = {}

    def check(name: str, x: np.ndarray, y: np.ndarray, group: str):
        passed = same(x, y)
        row = dict(passed=passed, group=group, n_elements=int(x.size), shape_before=list(x.shape), shape_after=list(y.shape))
        if not passed and x.shape == y.shape and x.dtype.kind in "fiub":
            row["max_abs_diff"] = float(np.nanmax(np.abs(x.astype(float) - y.astype(float))))
        results[name] = row

    # geoms and sites: match the kept ones by order, verified by (name, body) sequence; the rest are "added" (named *_visual)
    kept_of, added_of = {}, {}
    for kind, skip_visual in (("geom", GEOM_VISUAL), ("site", SITE_VISUAL)):
        names_b, bodies_b, names_a, bodies_a = b(f"{kind}_names"), b(f"{kind}_body_names"), a(f"{kind}_names"), a(f"{kind}_body_names")
        added = [i for i, n in enumerate(names_a) if n.endswith("_visual") and n not in set(names_b)]
        kept = [i for i in range(len(names_a)) if i not in added]
        sequence_ok = [(names_a[i], bodies_a[i]) for i in kept] == list(zip(names_b, bodies_b))
        results[f"{kind}_match_by_order"] = dict(passed=bool(sequence_ok), group=f"{kind}s", n_elements=len(kept),
                                                 note=f"(name, body) sequence of the kept 'after' {kind}s equals 'before'")
        kept_of[kind], added_of[kind] = kept, added
        if kind == "geom":
            # a new geom is visual only if it never collides (contype = conaffinity = 0); its mass shows in body_mass, compared below
            results["added_geoms_visual_only"] = dict(
                passed=all(a("geom_contype")[i] == 0 and a("geom_conaffinity")[i] == 0 for i in added), group="geoms", n_elements=len(added),
                note="contype = conaffinity = 0 for every geom only 'after' has")
        if sequence_ok:
            fields = [f.split("|", 2)[2] for f in before.files if f.startswith(f"{key}|{kind}_")]
            fields = [f for f in fields if f not in skip_visual and not f.endswith(("_names", "_body_names"))]
            fields = [f for f in GEOM_SPEC if kind == "geom"] + sorted(set(fields) - set(GEOM_SPEC))     # the spec's list first
            for field in fields:
                check(field, b(field), a(field)[kept], f"{kind}s (spec)" if field in GEOM_SPEC else f"{kind}s (extra)")
    added_info = []
    for kind in ("geom", "site"):
        names_a, bodies_a = a(f"{kind}_names"), a(f"{kind}_body_names")
        for i in added_of[kind]:
            added_info.append(dict(kind=kind, name=str(names_a[i]), body=str(bodies_a[i]), size=a(f"{kind}_size")[i].tolist(),
                                   pos=a(f"{kind}_pos")[i].tolist(), collides=(bool(a("geom_contype")[i] or a("geom_conaffinity")[i])
                                                                               if kind == "geom" else False)))
    added = added_of["geom"]
    bodies_a = a("geom_body_names")
    # bodies
    check("body_names", b("body_names"), a("body_names"), "bodies")
    for field in BODY_SPEC:
        check(field, b(field), a(field), "bodies (spec)")
    for field in BODY_EXTRA:
        check(field, b(field), a(field), "bodies (extra)")
    per_body = {n: sum(1 for g in added if bodies_a[g] == n) for n in set(bodies_a[g] for g in added)}   # empty: no geom is added
    expected_geomnum = b("body_geomnum") + np.array([per_body.get(n, 0) for n in b("body_names")])
    check("body_geomnum (+ added geoms)", expected_geomnum, a("body_geomnum"), "bodies (extra)")
    other_body = sorted(f.split("|", 2)[2] for f in before.files if f.startswith(f"{key}|body_"))
    skip = set(BODY_SPEC) | set(BODY_EXTRA) | {"body_names", "body_geomnum", "body_geomadr"}     # geomadr = start index into geom_*: shifts with ngeom
    bvh = lambda f: "bvh" in f
    for field in other_body:
        if field in skip or bvh(field) or field.startswith("body_names"):
            continue
        check(field, b(field), a(field), "bodies (extra)")
    # the solver options
    for field in sorted(f.split("|", 2)[2] for f in before.files if f.startswith(f"{key}|opt.")):
        check(field, b(field), a(field), "opt")
    # extras
    for prefix in EXTRA_PREFIXES:
        for field in sorted(f.split("|", 2)[2] for f in before.files if f.startswith(f"{key}|{prefix}")):
            check(field, b(field), a(field), "extra: joints / sites / sensors / ...")
    for field in ("qpos0", "qpos_spring", *EXTRA_SCALARS, *(f"stat.{s}" for s in STAT_PHYSICS)):
        check(field, b(field), a(field), "extra: joints / sites / sensors / ...")
    # explicit contact pairs: by geom NAME, since geom ids shift when a geom is added in the middle
    check("pair_geom_names", b("pair_geom_names"), a("pair_geom_names"), "extra: contact pairs")
    for field in PAIR_FIELDS:
        check(field, b(field), a(field), "extra: contact pairs")
    # dynamics
    for field in ("dyn_qpos", "dyn_qvel", "dyn_ncon"):
        check(field, b(field), a(field), f"dynamics ({DYNAMICS_STEPS} control steps = {DYNAMICS_STEPS * 0.02:.0f} s)")
    counts = dict(geoms=(int(b("ngeom")), int(a("ngeom"))), sites=(int(b("nsite")), int(a("nsite"))),
                  cameras_lights_textures_materials_before=b("counts_visual").tolist(),
                  cameras_lights_textures_materials_after=a("counts_visual").tolist())
    return dict(results=results, added=added_info, counts=counts, passed=all(r["passed"] for r in results.values()))


def print_table(combos: dict[str, dict]) -> None:
    keys = list(combos)
    col = max(len(k) for k in keys) + 2
    rows = {}
    for k, c in combos.items():
        for name, r in c["results"].items():
            rows.setdefault(r["group"], {}).setdefault(name, {})[k] = r["passed"]
    print("\nPASS / FAIL per field, before (base commit) vs after (this tree), one column per cup / contact model")
    print(" " * 38 + "".join(f"{k:>{col}}" for k in keys))
    for group, fields in rows.items():
        print(f"-- {group}")
        shown = list(fields.items())
        collapse = group.startswith(("opt", "extra", "geoms (extra)", "sites (extra)", "bodies (extra)")) and len(shown) > 12
        for name, per in shown:
            if collapse and all(per.values()) and name not in ("pair_geom_names", "body_geomnum (+ added geoms)"):
                continue
            print(f"   {name:<35}" + "".join(f"{'PASS' if per.get(k) else 'FAIL':>{col}}" for k in keys))
        if collapse:
            n_pass = sum(all(p.values()) for p in fields.values())
            print(f"   ({n_pass} of {len(shown)} fields identical in every combination; only the failures are listed above)")
    added = {}
    for k, c in combos.items():
        for g in c["added"]:
            added.setdefault((g["kind"], g["name"], g["body"], tuple(np.round(g["size"], 4)), g["collides"]), []).append(k)
    for (kind, name, body, size, collides), where in added.items():
        print(f"added visual {kind}: {name} on body {body}, size {list(size)}, collides: {collides}  [{', '.join(where)}]")
    if not any(kind == "geom" for kind, *_ in added):
        print("no geom is added: every geom array is compared in full")
    print(f"\ngeoms before / after: {sorted({c['counts']['geoms'] for c in combos.values()})}, sites {sorted({c['counts']['sites'] for c in combos.values()})}; "
          f"(cameras, lights, textures, materials) before {combos[keys[0]]['counts']['cameras_lights_textures_materials_before']}, "
          f"after {combos[keys[0]]['counts']['cameras_lights_textures_materials_after']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", default=DEFAULT_BASE, help="the commit to compare against (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--dump", nargs=2, metavar=("ROOT", "OUT_NPZ"), help="internal: worker mode")
    args = ap.parse_args()
    if args.dump:
        dump_model(*args.dump)
        return 0
    t0 = time.perf_counter()
    base_root, base_full = materialise_base(args.base)
    scratch = REPO / "tmp" / "visual_check"
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"before = {base_full[:10]} ({base_root}); after = {REPO} (HEAD {git('rev-parse', '--short', 'HEAD')}"
          f"{', with uncommitted changes' if git('status', '--short', '--', 'sim', 'config') else ''})", flush=True)
    t_before = run_worker(base_root, scratch / "before.npz")        # one process at a time
    t_after = run_worker(REPO, scratch / "after.npz")
    before, after = np.load(scratch / "before.npz"), np.load(scratch / "after.npz")
    combos = {f"{cup}/{contact}": compare(before, after, f"{cup}|{contact}") for cup in CUP_MODELS for contact in CONTACT_MODELS}
    print_table(combos)
    ok = all(c["passed"] for c in combos.values())
    failures = [f"{k}: {n}" for k, c in combos.items() for n, r in c["results"].items() if not r["passed"]]
    n_checks = sum(len(c["results"]) for c in combos.values())
    out = dict(
        question="Did the visual polish change any physics? Every non-visual geom, site and body array, every solver option, and 1 s of "
                 "dynamics, compiled before (base commit) and after, for every cup model x contact model",
        base_commit=base_full, after=dict(head=git("rev-parse", "HEAD"), sim_config_dirty=bool(git("status", "--short", "--", "sim", "config"))),
        method="two worker processes (sim/check_visual_only.py --dump), arrays compared with np.array_equal, no tolerance; geoms matched by "
               "order (checked by (name, body) sequence); geoms / sites only in 'after' (named *_visual) are the visual-only additions: "
               "excluded from the comparison and listed (a geom would have to have contype = conaffinity = 0)",
        spec_fields=dict(bodies=list(BODY_SPEC), geoms=list(GEOM_SPEC), opt="every field of mjOption", geom_arrays_left_out=list(GEOM_VISUAL)),
        combos={k: dict(passed=c["passed"], added=c["added"], counts=c["counts"], results=c["results"]) for k, c in combos.items()},
        n_comparisons=n_checks, failures=failures, all_pass=ok,
        runtime_s=round(time.perf_counter() - t0, 1), worker_s=dict(before=round(t_before, 1), after=round(t_after, 1)),
        peak_rss_mb_parent=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        peak_rss_mb_workers=round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "check_visual_only.json").write_text(json.dumps(out, indent=2))
    print(f"\n{'PASS' if ok else 'FAIL'}: {n_checks - len(failures)}/{n_checks} comparisons identical "
          f"({len(combos)} combinations); wrote {args.out / 'check_visual_only.json'} ({out['runtime_s']} s)")
    for f in failures:
        print("  differs:", f)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
