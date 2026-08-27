# Interactive Articulation Refinement UI

The workflow's automatic joint prediction (step 5) is a VLM actor/critic loop:
it usually gets joint type and rough placement right, but axes can be slightly
off, pivots can sit off the true hinge line, limits are often too wide or too
narrow, and effort/velocity are hard-coded placeholders (`5`/`5`). Dynamics
are estimated separately by step 5b (`estimate_physics.py`) and sometimes need
a human touch too. The segmentation step already has an interactive correction
UI (`postprocess_segmentation/interactive_ui.py`); this is the equivalent for
the *articulation result*.

The UI (package [`simfoundry/refine_articulation/`](../simfoundry/refine_articulation/))
is a Dash web app that loads an object's published result
(`<root_dir>/<scene>/<object>/results/{mobility.urdf, meshes/}`) and lets you
refine, per joint, with a live 3D preview of the articulated motion:

- **Joint type** — revolute / prismatic / continuous / fixed (e.g. demote a
  hallucinated joint to fixed, or promote a missed one).
- **Axis** — numeric editing in the parent-link frame, flip, and one-click
  alignment to a world axis (converted through the kinematic chain); the
  world-frame direction is always shown alongside.
- **Pivot / origin** — numeric editing or *pick a point on the mesh* in the 3D
  view. The child link's visual **and** collision origins are compensated
  automatically so the part's rest pose (q=0) never jumps — the same invariant
  the workflow's own pivot relocation (`translate_link` in odio_urdf)
  maintains. Joints parented to the edited joint's child are compensated too,
  so multi-level subtrees stay put.
- **Limits** — lower/upper (radians for revolute with a live degree readout,
  meters for prismatic), effort, velocity. The 3D view draws the sweep arc /
  travel segment and translucent "ghost" poses at both limits.
- **Test motion** — a per-joint slider (with animate toggle) drives forward
  kinematics so you can watch the joint move through its range.
- **Dynamics** — per-joint damping and friction, plus per-part mass, surface
  friction, and joint damping. The physics-estimation step's values (step 5b)
  appear as baselines; the parts table shows them as placeholders and blank
  inputs keep them.

Multiple objects of a scene are handled in one session via the object
dropdown.

## Running it

As part of the workflow (opens after step 5 publishes results):

```yaml
# in your generated_<scene>.yaml / template
s6_refine_articulation:
  enabled: true
  port: 8060
  open_browser: true
  max_faces_per_link: 40000   # display-only decimation threshold
```

Post-hoc, on already-published results (no articulation step is re-run):

```bash
cd simfoundry
python run_articulation_refinement.py --config-name=generated_<scene>
# subset of objects:
python run_articulation_refinement.py --config-name=generated_<scene> '+refine_objects=[toaster_oven]'
```

The server binds `127.0.0.1` (ports are probed upward from `port`); use SSH
port forwarding on remote machines.

## What gets written on save

All writes go to the object's `results/` directory:

| File | Meaning |
| --- | --- |
| `mobility.urdf` | Updated in place — this is what downstream consumers read. |
| `mobility_original.urdf` | One-time pristine backup of the pre-refinement URDF. |
| `mobility_refined_<N>.urdf` | Versioned copy of every save (session history). |
| `physics_overrides.json` | Per-part `{mass_kg, friction, joint_damping}` and per-joint `{damping, friction}` user overrides. |
| `refinement_log.json` | Timestamped log of each save (changed joints, warnings). |

Saving validates the URDF first: every revolute/prismatic joint must keep a
`<limit>` with `lower < upper` (downstream urdfpy-based consumers hard-fail
without one), continuous joints must not carry one, and movable axes are
normalized on write.

## How dynamics flow downstream (SimFoundry pipeline)

The articulation pipeline is the single source of dynamics for articulated
objects:

1. **Step 5b** (`estimate_physics.py`, `s5b_estimate_physics` config) makes
   one VLM call per object and writes per-joint damping/friction into
   `mobility.urdf` (`<dynamics>`) plus per-part mass/surface friction into
   `results/physics_properties.json`.
2. **This UI's edits** land in `results/physics_overrides.json` (and the
   URDF), and always take precedence over the step-5b estimates.
3. **SimFoundry's sim-ready stage (stage 10)** consumes both: it preserves the
   URDF's `<dynamics>`, builds `<inertial>` from the pipeline's `mass_kg`, and
   feeds surface friction to physics stabilization — falling back to its own
   legacy VLM estimate only when `physics_properties.json` is absent (results
   produced before step 5b existed).

Other caveats:

- **Geometry and limits survive as-is.** Stage 10 preserves joint types,
  axes, origins, and the full `<limit>` element (effort/velocity included).
- **Limit signs feed a heuristic.** Stage 10 infers each joint's "openable
  direction" from whichever limit is closer to zero; flipping limit signs
  flips that direction in simulation.
- **q=0 is the as-scanned rest pose** (a scanned-open door has q=0 = open).
  Downstream metadata is computed at q=0, so the UI warns when edited limits
  exclude zero.
- **Re-articulation overwrites refinements.** If stage 8b re-runs an object
  (e.g. the canonical mesh orientation changed), the object's output directory
  is rebuilt and refinements are lost — `refinement_log.json` and the
  versioned copies exist so you can tell it happened; re-apply on the fresh
  result.
