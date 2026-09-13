"""Serve a local browser UI for reviewing and correcting episode labels."""

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from urllib.parse import unquote, urlparse

import h5py
import numpy as np

CAMERAS = ("head_rgb", "left_wrist_rgb", "right_wrist_rgb")
LABELS = ("success", "aborted", "timeout", "interrupted", "rejected", "synthetic")

HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Ultra dataset annotator</title>
<style>
:root{color-scheme:dark;font:15px system-ui;background:#12151a;color:#e8edf2}*{box-sizing:border-box}
body{margin:0;display:grid;grid-template-columns:280px 1fr;height:100vh}aside{padding:18px;border-right:1px solid #343a43;overflow:auto}
main{padding:18px;overflow:auto}h1{font-size:19px;margin:0 0 6px}.muted{color:#9ca7b5;font-size:13px}.episode{width:100%;text-align:left;padding:10px;margin:4px 0;border:1px solid #343a43;border-radius:7px;background:#1b2027;color:inherit;cursor:pointer}.episode.active{border-color:#5aa7ff;background:#202c39}.episode b{display:block}.episode span{font-size:12px;color:#aab4c0}
.views{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:10px;margin-top:16px}.view{background:#090b0e;border-radius:8px;overflow:hidden}.view label{display:block;padding:7px 10px;background:#20252c}.view canvas{display:block;width:100%;aspect-ratio:4/3;object-fit:contain}
.transport{display:grid;grid-template-columns:auto auto 1fr auto;gap:10px;align-items:center;margin:16px 0}.transport input{width:100%}button,select,textarea{font:inherit}.primary{background:#2388ed;color:white;border:0;border-radius:6px;padding:8px 14px;cursor:pointer}.panel{display:grid;grid-template-columns:180px 1fr auto;gap:10px;align-items:start;background:#1b2027;padding:14px;border-radius:8px}select,textarea{background:#101318;color:inherit;border:1px solid #414854;border-radius:5px;padding:8px}textarea{min-height:70px;resize:vertical}.saved{color:#70d99b;padding:8px}
@media(max-width:900px){body{grid-template-columns:1fr;height:auto}aside{border-right:0;border-bottom:1px solid #343a43}.views{grid-template-columns:1fr}.panel{grid-template-columns:1fr}}
</style></head><body>
<aside><h1>Ultra annotator</h1><div id="dataset" class="muted"></div><div id="episodes"></div></aside>
<main><h1 id="title">Select an episode</h1><div id="details" class="muted"></div>
<div class="views" id="views"></div>
<div class="transport"><button id="play" class="primary">Play</button><button id="prev">−1</button><input id="frame" type="range" min="0" max="0" value="0"><span id="counter">0 / 0</span></div>
<div class="panel"><select id="status"></select><textarea id="notes" placeholder="Notes about this episode or label correction"></textarea><button id="save" class="primary">Save correction</button></div><div id="saved" class="saved"></div>
<script>
const cameras=['head_rgb','left_wrist_rgb','right_wrist_rgb'], labels=['success','aborted','timeout','interrupted','rejected','synthetic'];
let info,current=null,frame=0,timer=null;
const $=id=>document.getElementById(id); labels.forEach(x=>$('status').add(new Option(x,x)));
async function api(url,options){const r=await fetch(url,options);if(!r.ok)throw Error(await r.text());return r.headers.get('content-type')?.includes('json')?r.json():r.arrayBuffer()}
function renderList(){const box=$('episodes');box.innerHTML='';info.episodes.forEach(ep=>{const b=document.createElement('button');b.className='episode'+(current?.name===ep.name?' active':'');b.innerHTML=`<b>${ep.name}</b><span>${ep.effective_status}${ep.annotated?' • corrected':''} · ${ep.samples} frames · ${ep.seconds.toFixed(1)}s</span>`;b.onclick=()=>select(ep);box.appendChild(b)})}
function select(ep){stop();current=ep;frame=0;$('title').textContent=ep.name;$('details').textContent=`recorded: ${ep.recorded_status} · effective: ${ep.effective_status} · ${ep.samples} samples @ ${info.control_hz} Hz`;$('status').value=ep.effective_status;$('notes').value=ep.notes||'';$('frame').max=Math.max(0,ep.samples-1);$('frame').value=0;$('views').innerHTML='';ep.cameras.forEach(cam=>{$('views').insertAdjacentHTML('beforeend',`<div class="view"><label>${cam.replaceAll('_',' ')}</label><canvas id="${cam}"></canvas></div>`)});renderList();draw()}
async function draw(){if(!current||!current.samples)return;$('counter').textContent=`${frame+1} / ${current.samples}`;$('frame').value=frame;await Promise.all(current.cameras.map(async cam=>{const r=await fetch(`/api/frame/${encodeURIComponent(current.name)}/${cam}/${frame}`);if(!r.ok)throw Error(await r.text());const w=+r.headers.get('X-Width'),h=+r.headers.get('X-Height'),rgb=new Uint8ClampedArray(await r.arrayBuffer()),rgba=new Uint8ClampedArray(w*h*4);for(let i=0,j=0;i<rgb.length;i+=3,j+=4){rgba[j]=rgb[i];rgba[j+1]=rgb[i+1];rgba[j+2]=rgb[i+2];rgba[j+3]=255}const c=$(cam);c.width=w;c.height=h;c.getContext('2d').putImageData(new ImageData(rgba,w,h),0,0)}))}
function stop(){if(timer)clearInterval(timer);timer=null;$('play').textContent='Play'}
function play(){if(timer){stop();return}$('play').textContent='Pause';timer=setInterval(()=>{if(frame>=current.samples-1){stop();return}frame++;draw()},1000/info.control_hz)}
$('play').onclick=play;$('prev').onclick=()=>{stop();frame=Math.max(0,frame-1);draw()};$('frame').oninput=e=>{stop();frame=+e.target.value;draw()};
$('save').onclick=async()=>{if(!current)return;const result=await api(`/api/annotation/${encodeURIComponent(current.name)}`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({status:$('status').value,notes:$('notes').value})});Object.assign(current,result.episode);$('saved').textContent=`Saved to ${result.path}`;renderList();setTimeout(()=>$('saved').textContent='',2500)};
api('/api/dataset').then(x=>{info=x;$('dataset').textContent=`${x.dataset}\n${x.task}`;renderList();if(x.episodes.length)select(x.episodes[0])}).catch(e=>document.body.textContent=e);
</script></main></body></html>"""


class Annotator:
    def __init__(self, dataset):
        self.dataset = Path(dataset).resolve()
        self.io_lock = threading.Lock()

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
                    "cameras": [camera for camera in CAMERAS if f"obs/{camera}" in demo],
                })
        return {
            "dataset": str(self.dataset), "task": metadata.get("task", ""),
            "control_hz": control_hz, "episodes": episodes,
        }

    def frame(self, episode, camera, index):
        if camera not in CAMERAS:
            raise ValueError("Unknown camera")
        with self.io_lock, h5py.File(self.dataset, "r") as handle:
            path = f"data/{episode}/obs/{camera}"
            if path not in handle:
                raise ValueError("Unknown episode or camera")
            dataset = handle[path]
            if not 0 <= index < len(dataset):
                raise ValueError("Frame index out of range")
            frame = np.asarray(dataset[index], dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB frame, got {frame.shape}")
        return np.ascontiguousarray(frame)

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
                elif len(parts) == 5 and parts[:2] == ["api", "frame"]:
                    frame = annotator.frame(parts[2], parts[3], int(parts[4]))
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(frame.nbytes))
                    self.send_header("X-Width", str(frame.shape[1]))
                    self.send_header("X-Height", str(frame.shape[0]))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(frame.tobytes())
                else:
                    self.send(404, "not found", "text/plain")
            except (KeyError, OSError, ValueError) as error:
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
