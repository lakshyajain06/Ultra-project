# Ultra Isaac Lab scene

A compact Isaac Lab tabletop scene containing the Ultra bimanual robot, a table,
a cube, and a plate. The scene is procedural and vectorizable, making it a clean
starting point for a manipulation environment.

## Layout

```text
assets/robots/ultra/ Source-contract metadata (generated USD stays local)
src/sim/             Isaac Lab task environments, robots, and teleoperation
src/data/            Shared demonstration schema, recording, and datasets
src/learning/        Simulator-independent models, inference, and training
scripts/             Data collection, training, and evaluation entry points
scripts/debug/       Scene and manager-environment smoke-test runners
tests/               Offline simulation, data, and learning tests
tools/               Asset conversion utilities
third_party/         Third-party source checkouts
```

The source domains have one-way dependencies: `data` is standalone, `learning`
may consume `data`, and `sim` does not import either. Scripts compose simulation
and learned-policy inference when a workflow needs both.

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
uv run python scripts/debug/run_isaaclab_scene.py --viz kit
```

For a finite headless check:

```bash
uv run python scripts/debug/run_isaaclab_scene.py --viz none --steps 250
```

To render over SSH without opening a desktop window, save a screenshot or MP4:

```bash
uv run python scripts/debug/run_isaaclab_scene.py --viz none --screenshot outputs/ultra_scene.png
uv run python scripts/debug/run_isaaclab_scene.py --viz none --video outputs/ultra_scene.mp4 --steps 150
```

The runner enables Isaac Lab's offscreen camera renderer automatically when
either output option is present. Screenshots show the configured reset pose;
the simulation then runs 5 steps by default as a basic physics check. A video
runs 150 steps unless `--steps` is provided. Video frames are captured at 10
FPS to match the scene's render interval.

Clone the scene for vectorized development with `--num_envs`, for example:

```bash
uv run python scripts/debug/run_isaaclab_scene.py --viz none --num_envs 16 --steps 250
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

The runner enables XR automatically. With `--viz none`, it also starts the AR
profile programmatically, so no desktop XR-panel interaction is required. With
the Kit viewer, use the XR panel to select OpenXR / System OpenXR Runtime and
start or stop XR manually. `--viz none` skips the desktop viewer and uses the
headset's immersive view.
When `--dataset` is present, the collector enables the authored Ultra head
camera (`zed_left`) and both wrist cameras and records synchronized RGB frames.
The default resolution is 320x240; use `--camera_width` and `--camera_height`
to change it. These dataset cameras are independent of the headset view. At the
default resolution, budget roughly 1 GB per recorded minute depending on image
content; HDF5 datasets remain excluded from Git.

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
does not create a new clutch rotation offset. On initial valid tracking, each
controller learns one fixed controller-to-tool transform that preserves the
current robot wrist orientation. Hold the controllers comfortably before
connecting; right-stick click recalibrates this transform without changing the
absolute mapping. `--tool_rotation_offset X Y Z` can instead provide an explicit
fixed local-axis transform in degrees. Tracking loss or pause requires releasing
the affected grip before moving again. Orientations use XYZW quaternions.
Translation scale defaults to 1.0 (`--scale`);
the two controllers solve only their corresponding seven-joint arms. The
thumbsticks solve the shared body link through Ultra's six torso joints, so the
entire bimanual assembly moves with it. The Quest view is dynamically anchored
to Ultra's `zed_left` head camera, following its position and full orientation.
Use `--anchor_pos X Y Z` for a world-axis offset from that camera (default
`0 0 0`) and `--anchor_yaw DEGREES` for a local yaw offset. The dynamic anchor
remains attached as Ultra's head moves. OpenXR reports the headset relative to
its floor origin, so `--headset_height` compensates that height when aligning
the user's eyes to the robot camera (default 1.65 m).

Pass `--randomize_cube_position` to sample a new cube position on every reset.
The default `--cube_position_range 0.08 0.08` applies independent ±8 cm X/Y
offsets around the authored start position; sampling is deterministic from
`--seed`, and the setting is stored in dataset metadata.

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

- `obs` and `next_obs`: the canonical 22-D `proprio` state, raw full-articulation
  joint positions/velocities/accelerations/torques/targets, per-link world
  poses/velocities, both palm poses, cube pose and plate pose, plus
  synchronized `head_rgb`, `left_wrist_rgb`, and `right_wrist_rgb` uint8 images
  paired before/after the action;
- `actions`: the executed 22 joint-position targets, ordered torso[6], left
  arm[7], left jaw[1], right arm[7], right jaw[1]; torso/arms use radians and
  jaws use **metres** (0 closed, 0.045 open);
- `quest`: the two world-frame controller records, including validity and buttons;
- `eef_targets`: positions in the shared-body frame and absolute world-frame
  orientations, plus wall-clock and simulated timestamps;
- `controller_rotation_offsets`: the per-step fixed controller-to-tool XYZW
  calibration used for reproducible absolute wrist targets;
- `initial_state/proprio`, `status`, `success`, and `num_samples`.

New recordings use schema version 6 and store `proprio` directly, in the same
22-D layout consumed by the manager environment and ACT pipeline. Older
schema-4/5 recordings remain readable; the training loader reconstructs their
state from the raw joint and body-relative end-effector fields.
The raw stream also includes full-articulation joint accelerations, applied
torques, commanded targets, per-link world poses and velocities, object state,
Quest packets, Cartesian targets, and camera frames for future reprocessing.

Dataset metadata includes joint names, action units, quaternion convention,
tool rotation offset, task text, seed, control rate, package versions, and hashes
of the robot asset and lockfile. RGB arrays use HWC layout and fast lossless HDF5
LZF compression. Transitions are flushed incrementally, and existing dataset
paths are refused.

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

### Review and correct episode labels

Start the browser annotator on localhost:

```bash
uv run --locked python scripts/annotate_data.py datasets/ultra_002.hdf5 --port 8000
```

Open `http://127.0.0.1:8000`, select an episode, play or scrub its synchronized
head and wrist frames, choose the corrected status, add optional notes, and
save. Corrections update the episode attributes in the HDF5 file directly.
`original_status` preserves the first recorded label, while `status`, `success`,
`annotation_notes`, and `annotated_utc` reflect the review. Do not collect,
replay, or train from the same file while the annotator is running.

For access from another machine on a trusted LAN, bind all interfaces:

```bash
uv run --locked python scripts/annotate_data.py datasets/ultra_002.hdf5 \
  --host 0.0.0.0 --port 8000
```

The server has no authentication; do not expose it to the public internet.

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

## Manager-based RL evaluation environment

`UltraCubePlateEnvCfg` provides a vectorized Isaac Lab `ManagerBasedRLEnv` for
policy evaluation and RL fine-tuning. It preserves the demonstrations' absolute
22-joint action order, 25 Hz control rate, and 50 Hz physics rate. Actions are
unscaled absolute position targets in the same units as teleop (radians for the
torso/arms and metres for the jaws), clipped to the authored joint limits. A
future ACT controller can therefore emit the same target vector stored in
`actions` without an environment-side representation change.

The primary deployable observation is the 22-value `proprio` vector
`[torso_q(6), left_eef_pose_body(7), left_gripper_q(1),
right_eef_pose_body(7), right_gripper_q(1)]`. Consequently, slices `0:6`,
`6:13`, `13`, `14:21`, and `21` have stable meanings. Each end-effector pose is
XYZ plus XYZW quaternion relative to the shared `fr30_6` body; quaternions are
normalized by the frame transform and canonicalized to non-negative W so the
same orientation cannot jump between `q` and `-q`.

This task-space state is a compact fit for ACT and the teleop data: torso angles
retain the shared body's configuration, body-relative wrist poses remove
irrelevant world translation and move with that shared body, and the two jaw
openings retain the grasp state. The slice boundaries also align with the
teleop action's torso/left/right segments even though actions remain absolute
joint targets. Every value is hardware-measurable or available through forward
kinematics and can be derived directly from recorded `joint_pos` and
`eef_pose_body`; cube/plate truth is deliberately excluded. Arm joint angles,
passive jaw followers, and joint velocities are not part of the default policy
input. ACT's observation/action history captures short-term motion, while this
smaller representation avoids redundant passive and kinematic coordinates.

The manager environment exposes only deployable `proprio` and any requested RGB
terms to `policy`. Cube/plate truth is used by reward and termination managers,
not exposed as a policy observation.

Success is deliberately stricter than center overlap. The cube footprint must
fit within the circular plate in the plate's local frame, its center must be at
the face-on-face resting height, the plate must face upward, cube/plate relative
linear and angular motion must be small, and each gripper must be open or clear
of the cube. These conditions must hold for five consecutive 25 Hz control
steps (0.2 seconds). This stable geometric support test is deterministic across
sleeping/contact-reporting states while still rejecting fly-throughs, edge
overhangs, carried cubes, and toppled plates. Episodes also end when the cube
falls below the table or after 120 simulated seconds.

Run a short headless smoke test:

```bash
uv run --locked python scripts/debug/run_ultra_rl_env.py --viz none --num_envs 4 --steps 10

# Also exercise the stable-placement termination with a known-good placement.
uv run --locked python scripts/debug/run_ultra_rl_env.py --viz none --steps 10 --check-success
```

Use `--vision` to enable float32 HWC `head_rgb`, `left_wrist_rgb`, and
`right_wrist_rgb` policy observations in the same 0..255 value range and at the
same default 320x240 resolution as the HDF5 data. Each attached robot camera is
independently optional:

```bash
# Head and right wrist only, at a smaller resolution.
uv run --locked python scripts/debug/run_ultra_rl_env.py --viz none --vision \
  --cameras head_rgb right_wrist_rgb --camera-width 160 --camera-height 120
```

In Python, pass `enabled_cameras=("head_rgb",)` to `UltraCubePlateEnvCfg`, or
call `configure_cameras(cfg, streams, width, height)` before constructing the
environment. An empty stream tuple disables all cameras. RTX cameras are
substantially more expensive than state observations, so choose `num_envs`
accordingly.
## Policy behavior-cloning training

The default `model=act trainer=act` configuration selects a compact Action
Chunking with Transformers policy and its `ACTTrainer`. Hydra chooses the
trainer class; the generic entry point only composes the experiment and runs
that class. The model-family registry is used separately to reconstruct models
from checkpoints during inference. The
HDF5 training pipeline runs outside Isaac Sim and uses
only dependencies already present in the project. By default, each sample uses
the 22D task-centric `proprio` vector, the head/right-wrist RGB frames, and
predicts the next 25 absolute 22-joint targets. The state order is six torso
joint angles, left body-relative EEF xyz + XYZW quaternion, left gripper opening,
right body-relative EEF xyz + XYZW quaternion, and right gripper opening. This
mirrors the action's torso/left/right grouping while omitting arm angles and
velocities that are less useful than task-space wrist state for this task.
The loader uses `observation_joint_names` metadata to select torso and gripper
positions from legacy 24-DOF recordings and combines them with
`obs/eef_pose_body`; the two passive jaw followers are never included.
State and action statistics are fitted on the training episodes only and stored
inside every checkpoint. Training uses one Hydra YAML configuration with
`dataset`, `model`, `train`, and `wandb` sections; command-line values use
Hydra's `section.key=value` override syntax.

Train on one or more collections:

```bash
uv run python scripts/train_policy.py \
  'dataset.paths=[datasets/ultra_001.hdf5,datasets/ultra_002.hdf5]' \
  train.epochs=100 train.batch_size=16
```

By default each run creates a timestamped directory such as
`outputs/training/20260914_153012_-0400`. It contains `config.yaml` with the
fully composed Hydra configuration, `manifest.json`, `latest.pt`, `best.pt`,
and the run's tracking files. Set `output=outputs/policy/place_cube` when a
specific directory is preferred.

Only `success` episodes are selected by default. Status selection is explicit;
for exploratory training on a collection containing only aborted episodes use,
for example, `dataset.statuses=[aborted]`. Splitting is deterministic and performed by
episode, preventing transitions from one demonstration leaking across train and
validation sets. Use `train.seed`, `dataset.validation_fraction`,
`dataset.chunk_size`, `dataset.state_keys`, and `dataset.camera_names` to configure the input contract. The optional
legacy 44D controlled q/qdot state remains available with
`dataset.state_keys=[joint_pos,joint_vel]`. Cameras can be removed independently,
including all of them (`dataset.camera_names=[]`), without changing the dataset
format. Run with `--help` to inspect the fully composed configuration tree.
Configuration files live under `src/learning/training/conf`; Hydra groups make
presets composable, for example
`model=act trainer=act train=debug wandb=offline`. Model
parameters are overridden below the selected model, such as
`model.parameters.hidden_dim=128`; ACT's KL weight is similarly configured as
`model.training.kl_weight`. Dataset and model implementations validate their
resolved YAML mappings when they construct runtime objects.
Adding another policy family requires its model and trainer implementations,
matching Hydra files under `training/conf/model` and `training/conf/trainer`,
and a checkpoint-loading entry in `learning.models.registry`; the generic
training and evaluation entry points do not change.
Select exact demonstrations with `dataset.episode_keys`. Short keys are accepted
with or without the `data/` prefix and must resolve uniquely across all selected
files:

```bash
uv run python scripts/train_policy.py \
  'dataset.paths=[datasets/ultra_001.hdf5]' \
  'dataset.episode_keys=[demo_000003]' \
  dataset.validation_fraction=0
```

Resume an interrupted run with the same architecture and data arguments:

```bash
uv run python scripts/train_policy.py 'dataset.paths=[datasets/ultra_001.hdf5]' \
  output=outputs/policy/place_cube resume=outputs/policy/place_cube/latest.pt
```

`latest.pt` and `best.pt` are written at epoch boundaries and contain model and optimizer state, model/training
configuration, normalization, the exact episode split, counters, and random
number generator state. File size/mtime fingerprints prevent accidentally
resuming after an input dataset changed. `manifest.json` makes the split easy to inspect.
Console JSON reports normalized MAE, aggregate MAE in the recorded action units,
separate joint-target MAE in radians and jaw-target MAE in metres, KL loss, and
total loss. The best checkpoint is selected using validation MAE in action units
(or training MAE when no validation episodes are requested).

Training is tracked with Weights & Biases by default under the `ultra-policy`
project:

```bash
uv run python scripts/train_policy.py \
  'dataset.paths=[datasets/ultra_001.hdf5]' wandb.name=place-cube-baseline
```

The run records the complete train/dataset/model configuration and namespaced
train and validation metrics at each epoch. Logged diagnostics include
reconstruction and weighted-KL losses, physical and normalized errors,
pre-clipping gradient norm, learning rate, throughput, epoch timing, dataset
sizes, and model parameter counts. `wandb.entity`,
`wandb.group`, `wandb.tags`, and `wandb.mode=offline` are also supported.
When `resume=` is used with the same output directory, tracking resumes the
saved W&B run ID. Use `wandb.mode=disabled` for an intentionally untracked
run.

For manager-environment integration, pass the same observation dictionary to
the simulator-independent inference wrapper:

```python
from learning.inference import PolicyInference

policy = PolicyInference.from_checkpoint("outputs/policy/place_cube/best.pt", device="cuda")
action_chunk = policy.predict(observation)  # accepts manager [1,22] proprio and [1,H,W,3] RGB
action = action_chunk[0]
```

The runtime is deliberately separate from the controller: it emits the same
ordered absolute torso/arm/jaw targets recorded during teleoperation. The
evaluator selects how those chunks are executed through its Hydra `inference`
config group.
The manager environment's vision policy dictionary can be passed directly for
one environment: `proprio` is `[1,22]` in the task-centric order above, and
camera tensors are `[1,H,W,3]`. Unbatched `[22]` and `[H,W,3]` values are
accepted too. For a legacy observation with named 24-DOF `joint_pos` and
`eef_pose_body`, call
`policy.predict(observation, observation_joint_names)` so the same
metadata-driven selection is applied. Batched manager observations produce
batched action chunks with shape `[N, chunk_size, 22]`.
The wrapper currently stages device-backed observations through CPU NumPy
before normalized inference. This is a simple, reliable integration path, but
high-throughput vectorized deployment should use an on-device batched adapter
to avoid GPU-to-CPU-to-GPU camera copies.

Evaluate a checkpoint for complete episodes in the manager-based environment:

```bash
uv run --locked python scripts/evaluate_policy.py outputs/policy/place_cube/best.pt \
  --viz none --episodes 100 --num-envs 16
```

Recording is enabled by default. Each run creates a timestamped directory such
as `outputs/evaluation/20260914_153012_-0400` and writes one synchronized 2x2
MP4 per episode containing third-person, head, left-wrist, and right-wrist
views. Override the destination when needed:

```bash
uv run --locked python scripts/evaluate_policy.py outputs/policy/place_cube/best.pt \
  --viz none --episodes 10 --record-dir outputs/policy/place_cube/evaluation
```

The layout is third-person/head on the top row and left/right wrist on the
bottom row. Existing episode files are never overwritten. Pass `--no-record`
for metrics-only evaluation without video capture; policy cameras remain enabled
if the checkpoint requires them.

Parallel evaluation environments are spaced 25 metres apart by default so
neighboring robot clones do not contaminate the policy or recorded camera
views. Use `--env-spacing N` to override this distance. Evaluation uses the
same grid ground plane as teleoperation, keeping the visual background aligned
with the training data.

Isaac Sim's detailed startup and shutdown output is written to `isaac.log` in
the same timestamped evaluation directory. The console shows only checkpoint,
startup, environment, recording, and episode progress plus result JSON. Pass
`--verbose` or `--info` to show Isaac's logs directly while debugging startup.

The evaluator selects the state or vision environment from the cameras stored
in the checkpoint, uses the manager's success/drop/timeout terms, and prints
one JSON record per episode followed by an aggregate `EVALUATION` record.
Select execution with the Hydra configs under
`src/learning/training/conf/inference/`, or override a checkpoint's saved
setting with `--inference MODE`:

| Mode | Execution |
| --- | --- |
| `receding` | Predict every step and execute only the first action. |
| `chunk` | Predict once, then execute `steps` actions from that prediction without averaging. The default config uses 25 steps. |
| `temporal_ensemble` | Predict every step and average overlapping predictions for the current action, weighting a prediction by `exp(-decay * age)`; larger decay favors newer chunks. |

New checkpoints retain their Hydra inference setting. Older checkpoints use
the current default group (`chunk`). `--chunk-steps N` remains a shorthand for
open-loop chunk execution, and `--temporal-decay N` overrides the ensemble
weighting. For example:

```bash
uv run --locked python scripts/evaluate_policy.py outputs/policy/place_cube/best.pt \
  --inference temporal_ensemble --temporal-decay 0.01 --episodes 10 --headless
```

Episodes time out
after 200 policy steps (8 seconds at 25 Hz) by default; override that with
`--max-steps N`. This default includes margin over the longest successful
demonstration in `ultra_full.hdf5` (173 steps; success mean 145.2). Environments
run in parallel and are assigned new episode IDs as they finish. With recording
enabled, each active environment writes its own episode video.

To evaluate a checkpoint trained on randomized cube starts against the same
independent uniform ±8 cm XY distribution used by teleoperation, enable reset
randomization explicitly:

```bash
uv run --locked python scripts/evaluate_policy.py outputs/training/RUN/best.pt \
  --inference chunk --randomize-cube-position --cube-position-range 0.08 0.08 \
  --seed 0 --episodes 10 --headless
```

Each episode result includes `cube_start_xy` in local scene coordinates. The
cube's height and orientation, plate pose, and robot reset are unchanged.
Without `--randomize-cube-position`, evaluation still uses the fixed cube
start even if the checkpoint was trained on randomized demonstrations.

Run the CPU-only synthetic-data tests with:

```bash
uv run python -m unittest tests.test_act_training -v
```

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
