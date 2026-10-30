"""VISUAL ONLY: colours, textures, lights, sky, the render camera and the video resolution of the teach-g1 scene.

Nothing in this file may change what the simulation does. It sets geom ``rgba`` and ``material``, adds textures, materials, lights
and a camera, sets the offscreen framebuffer size, and adds ONE new element: a white disc on the mug's top face (the real mug
is dark green outside and white inside). It is a SITE, not a geom. A site is MuJoCo's marker for drawing: it never collides and
has no mass. A geom with
contype = conaffinity = 0 and mass 0 would also do, but any geom enters the cup body's bounding sphere, and with it the compiled
``dof_length`` of the cup's free joint (0.0640 -> 0.0863 m for the cylinder cup in the check's first run); a site leaves every
geom, body and dof array identical. Positions, sizes, masses, inertias, friction, solref / solimp, condim, contype / conaffinity
of existing geoms and every solver option are set in sim/scene.py and are not touched here. ``sim/check_visual_only.py`` proves
it: it compiles the scene before (the commit this file was added on top of) and after, with every cup and contact model, and
asserts that every physics array is identical.

Used by ``TableScene._add_scene`` (the spec edits) and by ``sim.m1_reach_test.VideoWriter`` (the camera and the resolution).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

# ---- video ----------------------------------------------------------------------------------------------------------------
VIDEO_WIDTH, VIDEO_HEIGHT = 1280, 720           # was 640 x 480; the offscreen framebuffer is set to this size in the model

# ---- render camera: on the robot's right, slightly in front of the hand, 0.75 m from the grasp so the right hand and the mug fill
# the frame. Fixed: it frames the cup, the coaster and the palm of all 50 evaluation layouts (checked by projecting them). ----
# Free-camera convention (mujoco.MjvCamera): the camera looks at ``lookat`` from ``distance`` away; azimuth 90 = on the robot's
# right (-y) looking left, 180 = in front of the robot looking back at it; elevation < 0 = looking down.
CAMERA_NAME = "grasp_right"
CAMERA = dict(lookat=(0.45, -0.15, 0.85), distance=0.75, azimuth=108.0, elevation=-20.0)

# ---- colours (rgba) --------------------------------------------------------------------------------------------------------
MUG_GREEN = (0.04, 0.28, 0.16, 1.0)            # my mug: glossy dark green
MUG_INTERIOR = (0.93, 0.93, 0.90, 1.0)          # ... with a white interior
COASTER_CREAM = (0.94, 0.89, 0.76, 1.0)         # light beige / cream resin
FLOOR_LIGHT, FLOOR_DARK = (0.80, 0.80, 0.79), (0.74, 0.74, 0.73)       # neutral grey, low-contrast checker
SKY_TOP, SKY_BOTTOM = (0.74, 0.77, 0.81), (0.92, 0.92, 0.91)
HAZE = (0.86, 0.87, 0.88, 1.0)                  # distant floor fades into this
WOOD_TINT, WOOD_LEG_TINT = (1.0, 1.0, 1.0, 1.0), (0.62, 0.55, 0.50, 1.0)
INTERIOR_RIM_M = 0.004                          # cosmetic: width of the green rim around the white disc (not a measurement)
INTERIOR_DISC_HALF_THICKNESS_M = 0.0003
N_PLANKS = 12                                   # wood texture: planks per texture tile


def wood_texture(size: int = 512, seed: int = 3) -> np.ndarray:
    """A seamless wood-plank texture (uint8, size x size x 3), planks running along the texture's u axis (columns).

    Procedural and seeded, so the repo needs no image file: ``N_PLANKS`` planks side by side with their own tone and ring phase,
    a ring pattern bent a little by smooth periodic noise (integer cycles per tile, so the tile repeats without a seam), fine grain
    streaks along the plank, and a dark joint line between planks.
    """
    rng = np.random.default_rng(seed)
    u = np.linspace(0.0, 1.0, size, endpoint=False)[None, :]      # along the grain
    v = np.linspace(0.0, 1.0, size, endpoint=False)[:, None]      # across the grain

    def periodic_noise(cu: int, cv: int, terms: int) -> np.ndarray:
        """Smooth noise in [-1, 1] with 0..cu cycles along u and 1..cv across v per tile."""
        out = np.zeros((size, size))
        for _ in range(terms):
            fu, fv = rng.integers(0, cu + 1), rng.integers(1, cv + 1)
            out += rng.uniform(0.4, 1.0) * np.cos(2 * np.pi * (fu * u + fv * v) + rng.uniform(0, 2 * np.pi))
        return out / np.abs(out).max()

    plank = np.floor(v * N_PLANKS).astype(int)[:, 0]               # plank index per row
    local_v = v * N_PLANKS - plank[:, None]                         # 0..1 across one plank
    ring_phase = rng.uniform(0.0, 1.0, N_PLANKS)[plank][:, None]    # each plank has its own ring pattern
    tone = rng.uniform(-0.08, 0.08, N_PLANKS)[plank][:, None]       # ... and its own brightness
    rings = 3.0 * local_v + ring_phase + 0.30 * periodic_noise(2, 3, 5) + 0.10 * periodic_noise(6, 8, 6)
    band = (0.5 + 0.5 * np.sin(2 * np.pi * rings)) ** 1.5          # 0..1 ring profile
    streak = 0.5 + 0.5 * periodic_noise(3, 120, 24)                 # long thin streaks along the grain
    shade = np.clip(0.50 * band + 0.35 * streak + tone + 0.10, 0.0, 1.0)
    light, dark = np.array([0.72, 0.57, 0.40]), np.array([0.50, 0.36, 0.24])
    rgb = dark + (light - dark) * shade[..., None]
    joint = np.clip(1.0 - np.minimum(local_v, 1.0 - local_v) / 0.015, 0.0, 1.0)         # dark line where two planks meet
    rgb *= 1.0 - 0.5 * joint[..., None]
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def camera_pose(lookat, distance: float, azimuth: float, elevation: float) -> tuple[np.ndarray, np.ndarray]:
    """Camera position and ``xyaxes`` (image x = right, image y = up) of mujoco.MjvCamera's free-camera convention."""
    az, el = math.radians(azimuth), math.radians(elevation)
    forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.asarray(lookat, float) - distance * forward, np.concatenate([right, up])


def free_camera() -> mujoco.MjvCamera:
    """The same view as the named camera, as a free camera (for models that do not contain it, e.g. the pre-polish scene)."""
    cam = mujoco.MjvCamera()
    cam.lookat[:] = CAMERA["lookat"]
    cam.distance, cam.azimuth, cam.elevation = CAMERA["distance"], CAMERA["azimuth"], CAMERA["elevation"]
    return cam


def add_visuals(spec: mujoco.MjSpec, cup_geoms: list, *, table_geoms: list, leg_geoms: list, coaster_geom) -> None:
    """Apply every visual of the scene to the spec, before it compiles. ``cup_geoms[0]`` is the body cylinder ("cup_geom").

    Static-geom colours are set here and not at their ``add_geom`` calls in sim/scene.py, so all of the look is in this file.
    """
    # --- textures and materials
    rgb = int(mujoco.mjtTextureRole.mjTEXROLE_RGB)
    with_texture = lambda mat, tex: setattr(mat, "textures", [tex if i == rgb else t for i, t in enumerate(mat.textures)])

    spec.add_texture(name="floor_grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=list(FLOOR_LIGHT), rgb2=list(FLOOR_DARK), width=512, height=512)
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=list(SKY_TOP), rgb2=list(SKY_BOTTOM), width=128, height=768)
    wood = wood_texture()
    spec.add_texture(name="wood", type=mujoco.mjtTexture.mjTEXTURE_2D, width=wood.shape[1], height=wood.shape[0], nchannel=3)
    spec.texture("wood").data = wood.tobytes()

    floor_mat = spec.add_material(name="floor_mat", texrepeat=[4, 4], texuniform=True, specular=0.05, shininess=0.1, reflectance=0.0)
    with_texture(floor_mat, "floor_grid")
    wood_mat = spec.add_material(name="wood_mat", texrepeat=[1.0, 1.0], texuniform=True, rgba=list(WOOD_TINT), specular=0.15, shininess=0.25)
    with_texture(wood_mat, "wood")
    leg_mat = spec.add_material(name="wood_leg_mat", texrepeat=[1.0, 1.0], texuniform=True, rgba=list(WOOD_LEG_TINT), specular=0.1, shininess=0.2)
    with_texture(leg_mat, "wood")
    spec.add_material(name="mug_green_mat", rgba=list(MUG_GREEN), specular=0.9, shininess=0.95)       # glossy glaze
    spec.add_material(name="mug_white_mat", rgba=list(MUG_INTERIOR), specular=0.6, shininess=0.8)
    spec.add_material(name="coaster_mat", rgba=list(COASTER_CREAM), specular=0.1, shininess=0.1)

    # --- existing geoms: colour only
    next(g for g in spec.geoms if g.name == "floor").material = "floor_mat"
    for g in table_geoms:
        g.material, g.rgba = "wood_mat", list(WOOD_TINT)
    for g in leg_geoms:
        g.material, g.rgba = "wood_leg_mat", list(WOOD_LEG_TINT)
    coaster_geom.material, coaster_geom.rgba = "coaster_mat", list(COASTER_CREAM)
    for g in cup_geoms:
        g.material, g.rgba = "mug_green_mat", list(MUG_GREEN)

    # --- the one new element: the mug's white interior, a thin disc (a site) just above the body cylinder's top face. It collides
    # with nothing and has no mass; everything else about the cup is untouched.
    body_geom = cup_geoms[0]
    radius, half_height = float(body_geom.size[0]), float(body_geom.size[1])
    body_geom.parent.add_site(name="cup_interior_visual", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                              size=[radius - INTERIOR_RIM_M, INTERIOR_DISC_HALF_THICKNESS_M, 0],
                              pos=[0, 0, half_height + INTERIOR_DISC_HALF_THICKNESS_M], material="mug_white_mat", rgba=list(MUG_INTERIOR))

    # --- lights: a soft key light that casts the shadows, and a shadow-less fill from the other side; brighter ambient
    directional = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    spec.worldbody.add_light(name="key", type=directional, pos=[1.6, -1.2, 3.0], dir=[-0.45, 0.35, -0.82], diffuse=[0.60, 0.59, 0.56],
                             specular=[0.25, 0.25, 0.25], castshadow=True)
    spec.worldbody.add_light(name="fill", type=directional, pos=[-0.5, 1.5, 2.2], dir=[0.3, -0.6, -0.75], diffuse=[0.22, 0.23, 0.26],
                             specular=[0.05, 0.05, 0.05], castshadow=False)
    spec.visual.headlight.ambient = [0.34, 0.34, 0.34]
    spec.visual.headlight.diffuse = [0.28, 0.28, 0.28]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]
    spec.visual.rgba.haze = list(HAZE)
    spec.visual.rgba.fog = list(HAZE)               # the far floor fades into this (also hides the checker's moire at the horizon)
    spec.visual.map.fogstart, spec.visual.map.fogend = 2.5, 7.0       # in units of the model's extent
    spec.visual.quality.shadowsize = 4096
    spec.visual.quality.offsamples = 8

    # --- the named render camera, and the offscreen framebuffer big enough for the video
    pos, xyaxes = camera_pose(**CAMERA)
    spec.worldbody.add_camera(name=CAMERA_NAME, mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED, pos=pos.tolist(), xyaxes=xyaxes.tolist(), fovy=45.0)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = VIDEO_WIDTH, VIDEO_HEIGHT
