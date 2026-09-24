"""Serve a local browser UI for reviewing and correcting episode labels."""

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import shutil
import subprocess
import threading
from urllib.parse import unquote, urlparse

import h5py
import numpy as np

CAMERAS = (
    "head_rgb", "left_wrist_rgb", "right_wrist_rgb",
    "task_rgb", "left_controlled_wrist_rgb", "right_controlled_wrist_rgb",
    "robot_left_head_rgb", "robot_left_left_wrist_rgb", "robot_left_right_wrist_rgb",
    "robot_right_head_rgb", "robot_right_left_wrist_rgb", "robot_right_right_wrist_rgb",
)
FACING_DUAL_CAMERAS = (
    "robot_left_head_rgb",
    "robot_left_left_wrist_rgb",
    "robot_left_right_wrist_rgb",
    "robot_right_head_rgb",
    "robot_right_left_wrist_rgb",
    "robot_right_right_wrist_rgb",
)
LABELS = ("success", "aborted", "timeout", "interrupted", "rejected", "synthetic")

HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Ultra dataset annotator</title>
<style>
:root{color-scheme:dark;font:15px system-ui;background:#12151a;color:#e8edf2}*{box-sizing:border-box}
body{margin:0;display:grid;grid-template-columns:300px 1fr;height:100vh}aside{padding:18px;border-right:1px solid #343a43;overflow:auto}
main{padding:18px;overflow:auto}h1{font-size:19px;margin:0 0 6px}h2{font-size:13px;text-transform:uppercase;letter-spacing:.08em;color:#9ca7b5;margin:18px 0 8px}.muted{color:#9ca7b5;font-size:13px;white-space:pre-line}.episode{width:100%;text-align:left;padding:10px;margin:4px 0;border:1px solid #343a43;border-radius:7px;background:#1b2027;color:inherit;cursor:pointer}.episode.active{border-color:#5aa7ff;background:#202c39}.episode b{display:block}.episode span{font-size:12px;color:#aab4c0}
video{display:block;width:min(100%,960px);max-height:70vh;background:#050608;margin:16px auto;border-radius:8px}button,select,textarea{font:inherit}.primary{background:#2388ed;color:white;border:0;border-radius:6px;padding:8px 14px;cursor:pointer}.panel{display:grid;grid-template-columns:180px 1fr auto;gap:10px;align-items:start;background:#1b2027;padding:14px;border-radius:8px}select,textarea{background:#101318;color:inherit;border:1px solid #414854;border-radius:5px;padding:8px}textarea{min-height:70px;resize:vertical}.saved{color:#70d99b;padding:8px}
.filter{display:grid;gap:5px;margin:14px 0}.filter select{width:100%}.stats{display:grid;gap:4px}.stat{display:grid;grid-template-columns:1fr auto;gap:8px;padding:5px 7px;border-radius:5px;background:#1b2027;font-size:12px}.stat span:last-child{color:#aab4c0;text-align:right}.stat.total{border-top:1px solid #414854;margin-top:3px;font-weight:600}.empty{padding:12px 4px;color:#9ca7b5;font-size:13px}
@media(max-width:900px){body{grid-template-columns:1fr;height:auto}aside{border-right:0;border-bottom:1px solid #343a43}.panel{grid-template-columns:1fr}}
</style></head><body>
<aside><h1>Ultra annotator</h1><div id="dataset" class="muted"></div>
<label class="filter">Filter by label<select id="filter"></select></label>
<h2>Label statistics</h2><div id="stats" class="stats"></div>
<h2 id="episode-heading">Episodes</h2><div id="episodes"></div></aside>
<main><h1 id="title">Select an episode</h1><div id="details" class="muted"></div>
<video id="video" controls playsinline></video>
<div class="panel"><select id="status"></select><textarea id="notes" placeholder="Notes about this episode or label correction"></textarea><button id="save" class="primary">Save correction</button></div><div id="saved" class="saved"></div>
<script>
const labels=['success','aborted','timeout','interrupted','rejected','synthetic'];
let info,current=null,videoUrl=null,selection=0;
const $=id=>document.getElementById(id); labels.forEach(x=>$('status').add(new Option(x,x)));$('filter').add(new Option('All labels',''));
async function api(url,options){const r=await fetch(url,options);if(!r.ok)throw Error(await r.text());return r.headers.get('content-type')?.includes('json')?r.json():r.arrayBuffer()}
function visibleEpisodes(){const label=$('filter').value;return info.episodes.filter(ep=>!label||ep.effective_status===label)}
function renderStats(){const stats=new Map();info.episodes.forEach(ep=>{const row=stats.get(ep.effective_status)||{episodes:0,samples:0};row.episodes++;row.samples+=ep.samples;stats.set(ep.effective_status,row)});const box=$('stats');box.innerHTML='';const order=[...labels,...[...stats.keys()].filter(x=>!labels.includes(x)).sort()];order.forEach(label=>{const row=stats.get(label)||{episodes:0,samples:0};const div=document.createElement('div');div.className='stat';div.innerHTML=`<span>${label}</span><span>${row.episodes} eps · ${row.samples.toLocaleString()} samples</span>`;box.appendChild(div)});const total=document.createElement('div');total.className='stat total';total.innerHTML=`<span>Total</span><span>${info.episodes.length} eps · ${info.episodes.reduce((n,ep)=>n+ep.samples,0).toLocaleString()} samples</span>`;box.appendChild(total)}
function renderList(){const episodes=visibleEpisodes(),box=$('episodes');box.innerHTML='';$('episode-heading').textContent=`Episodes (${episodes.length}/${info.episodes.length})`;if(!episodes.length){box.innerHTML='<div class="empty">No episodes match this label.</div>';return}episodes.forEach(ep=>{const b=document.createElement('button');b.className='episode'+(current?.name===ep.name?' active':'');b.innerHTML=`<b>${ep.name}</b><span>${ep.effective_status}${ep.annotated?' • corrected':''} · ${ep.samples} frames · ${ep.seconds.toFixed(1)}s</span>`;b.onclick=()=>select(ep);box.appendChild(b)})}
async function select(ep){const selected=++selection;current=null;$('video').pause();$('title').textContent=`Encoding ${ep.name}…`;$('details').textContent='Building an in-memory review video; no file is written';try{const loaded=await api(`/api/episode/${encodeURIComponent(ep.name)}`);const response=await fetch(`/api/video/${encodeURIComponent(ep.name)}`);if(!response.ok)throw Error(await response.text());const url=URL.createObjectURL(await response.blob());if(selected!==selection){URL.revokeObjectURL(url);return}if(videoUrl)URL.revokeObjectURL(videoUrl);videoUrl=url;$('video').src=url;current=ep;$('title').textContent=ep.name;$('details').textContent=`recorded: ${ep.recorded_status} · effective: ${ep.effective_status} · ${ep.samples} samples @ ${info.control_hz} Hz · ${(loaded.video_bytes/1048576).toFixed(1)} MiB in memory`;$('status').value=ep.effective_status;$('notes').value=ep.notes||'';renderList()}catch(error){if(selected!==selection)return;$('title').textContent=`Unable to load ${ep.name}`;$('details').textContent=error.message;console.error(error)}}
$('filter').onchange=renderList;
$('save').onclick=async()=>{if(!current)return;const result=await api(`/api/annotation/${encodeURIComponent(current.name)}`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({status:$('status').value,notes:$('notes').value})});Object.assign(current,result.episode);$('saved').textContent=`Saved to ${result.path}`;renderStats();renderList();setTimeout(()=>$('saved').textContent='',2500)};
api('/api/dataset').then(x=>{info=x;$('dataset').textContent=`${x.dataset}\n${x.task}`;const statuses=[...new Set(x.episodes.map(ep=>ep.effective_status))];[...labels,...statuses.filter(s=>!labels.includes(s)).sort()].forEach(label=>$('filter').add(new Option(label,label)));renderStats();renderList();if(x.episodes.length)select(x.episodes[0])}).catch(e=>document.body.textContent=e);
</script></main></body></html>"""


def episode_cameras(demo, metadata):
    """Return recorded RGB streams, preferring the order stored by the recorder."""
    obs = demo.get("obs")
    if obs is None:
        return []
    configured = metadata.get("camera_streams", ())
    if not isinstance(configured, (list, tuple)):
        configured = ()
    # Recorder metadata is the display contract. This lets upgraded datasets
    # retain archived camera arrays without adding them to the review mosaic.
    candidates = list(configured) if configured else list(CAMERAS) + sorted(obs.keys())
    cameras = []
    for camera in candidates:
        if not isinstance(camera, str) or camera in cameras or camera not in obs:
            continue
        dataset = obs[camera]
        if isinstance(dataset, h5py.Dataset) and len(dataset.shape) == 4 and dataset.shape[-1] == 3:
            cameras.append(camera)
    return cameras


def label_statistics(episodes):
    """Count episodes, samples, and duration for each effective label."""
    counts = {
        label: {"label": label, "episodes": 0, "samples": 0, "seconds": 0.0}
        for label in LABELS
    }
    for episode in episodes:
        label = episode["effective_status"]
        row = counts.setdefault(
            label, {"label": label, "episodes": 0, "samples": 0, "seconds": 0.0}
        )
        row["episodes"] += 1
        row["samples"] += episode["samples"]
        row["seconds"] += episode["seconds"]
    ordered_labels = [*LABELS, *sorted(set(counts).difference(LABELS))]
    return {
        "labels": [counts[label] for label in ordered_labels],
        "total": {
            "episodes": sum(row["episodes"] for row in counts.values()),
            "samples": sum(row["samples"] for row in counts.values()),
            "seconds": sum(row["seconds"] for row in counts.values()),
        },
    }


def compose_mosaic_frame(camera_frames, index, camera_names=None):
    """Arrange one timestep, including the two-triangle facing-dual layout."""
    height, width = camera_frames[0].shape[1:3]
    names = tuple(camera_names or ())
    if len(camera_frames) == 6 and set(names) == set(FACING_DUAL_CAMERAS):
        by_name = dict(zip(names, camera_frames))
        mosaic = np.zeros((height * 2, width * 4, 3), dtype=np.uint8)
        # Each robot is a triangle: centered head above its two wrist views.
        placements = {
            "robot_left_head_rgb": (0, width // 2),
            "robot_left_left_wrist_rgb": (height, 0),
            "robot_left_right_wrist_rgb": (height, width),
            "robot_right_head_rgb": (0, 2 * width + width // 2),
            "robot_right_left_wrist_rgb": (height, 2 * width),
            "robot_right_right_wrist_rgb": (height, 3 * width),
        }
        for name, (top, left) in placements.items():
            mosaic[top:top + height, left:left + width] = by_name[name][index]
        return mosaic
    if len(camera_frames) == 1:
        rows, columns = 1, 1
    elif len(camera_frames) == 2:
        rows, columns = 1, 2
    elif len(camera_frames) == 3:
        rows, columns = 2, 2
    else:
        columns = math.ceil(math.sqrt(len(camera_frames)))
        rows = math.ceil(len(camera_frames) / columns)
    mosaic = np.zeros((height * rows, width * columns, 3), dtype=np.uint8)
    if len(camera_frames) == 3:
        placements = ((0, width // 2), (height, 0), (height, width))
    else:
        placements = tuple(
            (height * (camera // columns), width * (camera % columns))
            for camera in range(len(camera_frames))
        )
    for frames, (top, left) in zip(camera_frames, placements):
        mosaic[top:top + height, left:left + width] = frames[index]
    return mosaic


def encode_mosaic_video(camera_frames, frame_rate, camera_names=None):
    """Encode synchronized RGB arrays as a fragmented MP4 held entirely in RAM."""
    if not camera_frames or frame_rate <= 0:
        raise ValueError("At least one camera and a positive frame rate are required")
    shapes = [frames.shape for frames in camera_frames]
    if any(len(shape) != 4 or shape[-1] != 3 for shape in shapes):
        raise ValueError(f"Camera arrays must be [T,H,W,3], got {shapes}")
    if len({shape[:3] for shape in shapes}) != 1:
        raise ValueError(f"Camera arrays must have matching T/H/W dimensions, got {shapes}")
    count, height, width, _ = shapes[0]
    if not count:
        raise ValueError("Cannot encode an empty episode")
    first_mosaic = compose_mosaic_frame(camera_frames, 0, camera_names)
    video_height, video_width = first_mosaic.shape[:2]
    video_height += video_height % 2
    video_width += video_width % 2
    command = [
        shutil.which("ffmpeg") or "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{video_width}x{video_height}", "-framerate", str(frame_rate),
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "25",
        "-pix_fmt", "yuv420p", "-movflags", "frag_keyframe+empty_moov+default_base_moof",
        "-f", "mp4", "pipe:1",
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = [], []
    readers = [
        threading.Thread(target=lambda: output.append(process.stdout.read()), daemon=True),
        threading.Thread(target=lambda: errors.append(process.stderr.read()), daemon=True),
    ]
    for reader in readers:
        reader.start()
    try:
        for index in range(count):
            mosaic = compose_mosaic_frame(camera_frames, index, camera_names)
            if mosaic.shape[:2] != (video_height, video_width):
                padded = np.zeros((video_height, video_width, 3), dtype=np.uint8)
                padded[:mosaic.shape[0], :mosaic.shape[1]] = mosaic
                mosaic = padded
            process.stdin.write(np.ascontiguousarray(mosaic).tobytes())
    except BrokenPipeError:
        pass
    finally:
        process.stdin.close()
    return_code = process.wait()
    for reader in readers:
        reader.join()
    if return_code:
        message = errors[0].decode(errors="replace").strip() if errors else "unknown ffmpeg error"
        raise RuntimeError(f"ffmpeg failed: {message}")
    return output[0]


class Annotator:
    def __init__(self, dataset):
        self.dataset = Path(dataset).resolve()
        self.io_lock = threading.Lock()
        self.cached_episode = None
        self.cached_video = b""

    def summary(self):
        with self.io_lock, h5py.File(self.dataset, "r") as handle:
            metadata = json.loads(handle.attrs.get("metadata", "{}"))
            control_hz = float(metadata.get("control_hz", 25))
            episodes = []
            for name, demo in handle["data"].items():
                recorded = str(demo.attrs.get("status", "unknown"))
                samples = len(demo.get("actions", ()))
                original = str(demo.attrs.get("original_status", recorded))
                episodes.append({
                    "name": name, "recorded_status": original,
                    "effective_status": recorded,
                    "notes": str(demo.attrs.get("annotation_notes", "")),
                    "annotated": "annotated_utc" in demo.attrs,
                    "samples": samples, "seconds": samples / control_hz,
                    "cameras": episode_cameras(demo, metadata),
                })
        return {
            "dataset": str(self.dataset), "task": metadata.get("task", ""),
            "control_hz": control_hz, "episodes": episodes,
            "statistics": label_statistics(episodes),
        }

    def load_episode(self, episode):
        """Replace the bounded cache with one in-memory mosaic MP4."""
        with self.io_lock:
            if self.cached_episode != episode:
                with h5py.File(self.dataset, "r") as handle:
                    path = f"data/{episode}"
                    if path not in handle:
                        raise ValueError("Unknown episode")
                    demo = handle[path]
                    metadata = json.loads(handle.attrs.get("metadata", "{}"))
                    cameras = episode_cameras(demo, metadata)
                    if not cameras:
                        raise ValueError("Episode has no recorded cameras")
                    frames = [np.asarray(demo[f"obs/{camera}"], dtype=np.uint8) for camera in cameras]
                    control_hz = float(metadata.get("control_hz", 25))
                video = encode_mosaic_video(frames, control_hz, cameras)
                self.cached_episode = episode
                self.cached_video = video
            return {
                "episode": self.cached_episode,
                "video_bytes": len(self.cached_video),
            }

    def video(self, episode):
        self.load_episode(episode)
        with self.io_lock:
            return self.cached_video

    def annotate(self, episode, status, notes):
        if status not in LABELS:
            raise ValueError(f"status must be one of {LABELS}")
        with self.io_lock, h5py.File(self.dataset, "r+") as handle:
            path = f"data/{episode}"
            if path not in handle:
                raise ValueError("Unknown episode")
            demo = handle[path]
            if "original_status" not in demo.attrs:
                demo.attrs["original_status"] = str(demo.attrs.get("status", "unknown"))
            demo.attrs["status"] = status
            demo.attrs["success"] = status == "success"
            demo.attrs["annotation_notes"] = str(notes).strip()
            demo.attrs["annotated_utc"] = datetime.now(timezone.utc).isoformat()
            handle.flush()
        episodes = self.summary()["episodes"]
        return self.dataset, next(item for item in episodes if item["name"] == episode)


def make_handler(annotator):
    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, content_type="application/json"):
            body = body if isinstance(body, bytes) else body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                parts = [unquote(x) for x in urlparse(self.path).path.split("/") if x]
                if not parts:
                    self.send(200, HTML, "text/html; charset=utf-8")
                elif parts == ["api", "dataset"]:
                    self.send(200, json.dumps(annotator.summary()))
                elif len(parts) == 3 and parts[:2] == ["api", "episode"]:
                    self.send(200, json.dumps(annotator.load_episode(parts[2])))
                elif len(parts) == 3 and parts[:2] == ["api", "video"]:
                    self.send(200, annotator.video(parts[2]), "video/mp4")
                else:
                    self.send(404, "not found", "text/plain")
            except (KeyError, OSError, RuntimeError, ValueError) as error:
                self.send(400, str(error), "text/plain")

        def do_PUT(self):
            try:
                parts = [unquote(x) for x in urlparse(self.path).path.split("/") if x]
                if len(parts) != 3 or parts[:2] != ["api", "annotation"]:
                    self.send(404, "not found", "text/plain")
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if size > 65536:
                    raise ValueError("Annotation payload is too large")
                payload = json.loads(self.rfile.read(size))
                path, episode = annotator.annotate(parts[2], payload.get("status"), payload.get("notes", ""))
                self.send(200, json.dumps({"path": str(path), "episode": episode}))
            except (json.JSONDecodeError, KeyError, OSError, ValueError) as error:
                self.send(400, str(error), "text/plain")

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} - {fmt % args}")

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address; use 0.0.0.0 for LAN access")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not args.dataset.is_file():
        parser.error(f"Dataset does not exist: {args.dataset}")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if shutil.which("ffmpeg") is None:
        parser.error("ffmpeg is required to create in-memory review videos")
    annotator = Annotator(args.dataset)
    annotator.summary()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(annotator))
    print(f"Reviewing {args.dataset}")
    print("Corrections are written directly into the HDF5 episode attributes.")
    print(f"Open http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
