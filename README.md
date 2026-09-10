# Ultra Isaac Lab scene

A compact Isaac Lab tabletop scene containing the Ultra bimanual robot, a table,
a cube, and a plate. The scene is procedural and vectorizable, making it a clean
starting point for a manipulation environment.

## Layout

```text
assets/robots/ultra/       Source-contract metadata (generated USD stays local)
scripts/                   Runnable Isaac Lab entry points
source/ultra_scene/        Scene package
source/ultra_scene/robots/ Robot configurations and controllers
tests/                     Offline USD integrity checks
tools/                     Asset conversion utilities
third_party/SimFoundry/    Existing third-party checkout
```

## Run the scene

Create the pinned Python 3.12 environment and install its dependencies:

```bash
uv sync
```

The robot USD is deliberately excluded from Git. Before running, either place
the flattened asset at `assets/robots/ultra/ultra.usd`, regenerate it as
described below, or point `ULTRA_USD_PATH` at a compatible Ultra USD.

The first sync can take time because Isaac Sim is large. Open the interactive
viewer through the locked environment:

```bash
uv run python scripts/run_isaaclab_scene.py --viz kit
```

For a finite headless check:

```bash
uv run python scripts/run_isaaclab_scene.py --viz none --steps 250
```

To render over SSH without opening a desktop window, save a screenshot or MP4:

```bash
uv run python scripts/run_isaaclab_scene.py --viz none --screenshot outputs/ultra_scene.png
uv run python scripts/run_isaaclab_scene.py --viz none --video outputs/ultra_scene.mp4 --steps 150
```

The runner enables Isaac Lab's offscreen camera renderer automatically when
either output option is present. Screenshots show the configured reset pose;
the simulation then runs 5 steps by default as a basic physics check. A video
runs 150 steps unless `--steps` is provided. Video frames are captured at 10
FPS to match the scene's render interval.

Clone the scene for vectorized development with `--num_envs`, for example:

```bash
uv run python scripts/run_isaaclab_scene.py --viz none --num_envs 16 --steps 250
```

The runner uses Isaac Lab's PhysX backend and its default scene configuration.
Ultra retains the source environment's 32 position and 4 velocity articulation
solver iterations. The lockfile uses NVIDIA's official Isaac Lab and Isaac Sim packages;
it does not depend on the Cobalt fork or its phone/crowdsourcing bridge.

## Quest 3 teleoperation

Install the versioned teleop extra from the committed lockfile (Linux x86-64):

```bash
uv sync --locked --extra teleop
```

Python 3.12, Isaac Lab `3.0.0b2.post1`, and Isaac Teleop `1.4.142` are pinned.
All resolved Python dependencies and wheel hashes are in `uv.lock`. Use
`--locked --extra teleop` on subsequent runs too: plain `uv sync` omits the
optional teleop runtime. This integration targets the installed beta APIs,
not the changing Isaac Lab `develop` examples. Do not install packages manually
into `.venv`; declare changes in `pyproject.toml` and regenerate `uv.lock`.
The manifest explicitly overrides Isaac Sim's `websockets==12` pin with
`15.0.1`, because CloudXR's WSS proxy requires the newer `websockets.asyncio`
API. This affects the base environment too and is recorded in the lockfile.

The host also needs an NVIDIA GPU/driver supported by the pinned Isaac Sim,
Vulkan (`libvulkan1`), and `libbsd0`. Those OS dependencies are not managed by uv.
The first live run asks for NVIDIA's separate CloudXR license acceptance and
stores the answer in `~/.cloudxr/run/eula_accepted`. This is a per-user runtime
setting, not a dependency or a file to commit.

Start teleoperation from a terminal on the workstation:

```bash
uv run --locked --extra teleop python scripts/teleop_ultra.py --viz kit
```

To collect demonstrations, choose a **new** dataset filename:

```bash
uv run --locked --extra teleop python scripts/teleop_ultra.py \
  --viz kit --dataset datasets/ultra_001.hdf5 \
  --task "Place the cube on the plate"
```

The runner enables XR automatically. In the Kit XR panel select OpenXR /
System OpenXR Runtime and click Start XR if it is not already running.
`--viz none` skips the desktop viewer and uses the headset's immersive view.
There are no camera sensors or camera-feed panels in this version.

In the Quest browser, open the matching [CloudXR 1.4 web client](https://nvidia.github.io/IsaacTeleop/client/release-1.4.x),
enter the workstation's LAN IP, follow its certificate link to
`https://<host-ip>:48322/`, accept the certificate for your workstation, return
to the client, and Connect. No custom APK or sideloading is needed.
The headset and workstation must be able to communicate over the LAN. If a
firewall blocks connections, allow LAN traffic to TCP 49100, TCP 48322, and
UDP 47998. The runner does not change firewall rules.

| Input | Behavior |
| --- | --- |
| Left/right grip squeeze | Hold to move that arm; release to clutch out |
| Left/right index trigger | Close that gripper, while its clutch is held |
| Left thumbstick | Move the shared body support left/right and forward/back |
| Right thumbstick | Yaw the shared body support and move it up/down |
| X (left primary) | Mark success, finish episode, reset scene |
| Y (left secondary) | Mark aborted, finish episode, reset scene |
| B (right secondary) | Pause/resume both arms |
| Right stick click | Recalibrate at the current wrist poses, preserving jaw commands |
| A (right primary) | Reserved by Isaac Lab for XR anchor rotation |
| CloudXR Start/Stop/Reset | Enable, pause, or abort/reset through the native control channel |

Release both grips once after connecting. Each subsequent grip press anchors
that controller's **translation** to its current robot target, so physical hand
placement does not teleport the wrist. Wrist orientation is absolute: while
clutched, it directly uses the controller's world orientation, and re-gripping
does not create a new clutch rotation offset. A fixed 90-degree controller-local
Z tool alignment removes the Quest-to-gripper wrist-roll mismatch; override it
with `--tool_rotation_offset X Y Z` (degrees), using `0 0 -90` if the roll is
opposite on your runtime. Tracking loss or pause requires releasing
the affected grip before moving again. Orientations use XYZW quaternions.
Translation scale defaults to 1.0 (`--scale`);
the two controllers solve only their corresponding seven-joint arms. The
thumbsticks solve the shared body link through Ultra's six torso joints, so the
entire bimanual assembly moves with it. Use `--anchor_pos X Y Z` to place the
Quest user's physical origin in the simulation (default `0 1.8 0`) and
`--anchor_yaw DEGREES` to turn the view about world Z. Ultra faces world -Y;
if the scene is behind you, restart with `--anchor_yaw 180`. Increasing anchor
Y moves the user farther behind Ultra, and changing anchor Z raises/lowers the
virtual floor relative to the user. This anchor is an initial placement, not a
camera rigidly attached to the robot.

Control runs at 25 Hz over the existing 50 Hz physics. Damped least-squares IK
uses one solve per palm and a separate solve for the shared body support.
Controller translation maps 1:1 relative to the clutch pose; controller
orientation maps absolutely in simulation world, with no clutch rotation offset.
There is no artificial Cartesian or angular target-rate limit. Joint updates
obey Ultra's authored velocity and position limits, and jaw travel is bounded to
0.08 m/s. Wrist positions stay in the shared-body frame, so translating the body
carries both arms rather than making arm IK counteract that motion. Because wrist
orientation is absolute, yawing the body makes the arms compensate to preserve
the controller-requested world orientation. IK is local and is **not collision-aware**;
avoid driving through the table or the other arm. Unreachable targets may need
a clutch/recalibration. These are simulation controls, not a real-robot driver.

An episode starts on the first engaged clutch. Success is an **operator label**
(X), not an implemented task predicate. Y or the headset reset records an
aborted attempt; the default 120-second simulated episode limit records a timeout
(`--episode_seconds`). Closing the app leaves the current attempt interrupted.
All attempts are retained with explicit labels; train behavior cloning only on
the desired labels. X/Y reset the scene immediately, so release the grips before
starting the next attempt.

### Demonstration format

The local HDF5 schema is versioned; it is not a claim of compatibility with
Isaac Lab Mimic or LeRobot's dataset schema. Each `data/demo_XXXXXX` contains:

- `obs` and `next_obs`: all 24 joint positions/velocities, both palm poses,
  cube pose and plate pose, paired before/after the action;
- `actions`: the executed 22 joint-position targets, ordered torso[6], left
  arm[7], left jaw[1], right arm[7], right jaw[1]; torso/arms use radians and
  jaws use **metres** (0 closed, 0.045 open);
- `quest`: the two world-frame controller records, including validity and buttons;
- `eef_targets`: positions in the shared-body frame and absolute world-frame
  orientations, plus wall-clock and simulated timestamps;
- `initial_state`, `status`, `success`, and `num_samples`.

Dataset metadata includes joint names, action units, quaternion convention,
tool rotation offset, task text, seed, control rate, package versions, and hashes of the robot asset
and lockfile. Transitions are flushed incrementally, and existing dataset paths
are refused. No images are recorded yet.

### Replay demonstrations

List attempts without launching Isaac Sim:

```bash
uv run --locked python scripts/replay_ultra.py datasets/ultra_002.hdf5 --list
```

Replay an attempt at recorded speed in the desktop viewer:

```bash
uv run --locked python scripts/replay_ultra.py datasets/ultra_002.hdf5 \
  --episode 1 --viz kit --hold-seconds 10
```

`--speed 2` replays at twice recorded speed. `--start-step` and `--end-step`
select a smaller interval. Replay restores the recorded robot, cube, and plate
initial state before applying the exact saved 22-joint action sequence.

### Verification without a headset

```bash
uv run --locked --extra teleop python -m unittest discover -s tests -p test_teleop.py -v
uv run --locked --extra teleop python scripts/teleop_ultra.py \
  --smoke --viz none --steps 50 --dataset /tmp/ultra_smoke_new.hdf5
```

The smoke run drives small upward motions through both IK chains and records
them with `status=synthetic` and `success=false`. It checks physics/control and
the recorder; only connecting the real Quest can validate tracking alignment,
buttons, network latency, and headset comfort. Smoke datasets are not training
demonstrations.

## Ultra fidelity

The robot is ported from `~/real2sim2real_ws` without changing its articulation
hierarchy or geometry. The configuration preserves:

- the 24-DOF articulation and original link/joint names;
- the 22 independently controlled joints and two authored jaw followers;
- the source table-placement rule (base 0.11 m below the tabletop and 0.98 m
  behind its edge) with `-90` degrees yaw;
- the left and right arm neutral configurations;
- the source PD gains, effort limits, and gripper range; and
- the source environment's 50 Hz physics timestep.

[`source_spec.yaml`](assets/robots/ultra/source_spec.yaml) and
[`neutral_pose.json`](assets/robots/ultra/neutral_pose.json) are snapshots of the
source robot contract. `ULTRA_USD_PATH` can point the scene at the original or
another compatible USD; otherwise runs use the ignored local flattened asset.

The table, cube, plate, lighting, and ground plane are intentionally simple and
are independent from the robot configuration.

## Rebuild the standalone asset

Generate or refresh the local standalone USD with:

```bash
/home/lairlab/miniforge3/envs/simfoundry-editor/bin/python \
  tools/convert_ultra_asset.py
```

The converter flattens referenced layers into one file while preserving the
source articulation hierarchy, physics relationships, cameras, collision
geometry, and materials. It never modifies the source asset. Generated USD files
and other large simulation/training artifacts are ignored and must be distributed
through external artifact storage rather than Git.

Run the offline asset checks with a Python environment that provides OpenUSD:

```bash
/home/lairlab/miniforge3/envs/simfoundry-editor/bin/python \
  -m unittest tests/test_ultra_asset.py -v
```
