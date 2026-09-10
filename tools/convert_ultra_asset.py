"""Flatten the source Ultra USD into one standalone Isaac Lab asset.

Run with the simfoundry-editor Python (standalone OpenUSD >= 25.05).
The source asset is never modified.
"""
import argparse
import json
from pathlib import Path


def convert(source, output):
    from pxr import Usd, UsdPhysics

    source = Path(source).expanduser().resolve()
    output = Path(output).resolve()
    stage = Usd.Stage.Open(str(source))
    # Keep the source articulation hierarchy and physics frames exactly intact.
    # Flattening only embeds payloads, meshes, materials, and referenced layers.
    for prim in stage.Traverse():
        if prim.IsInstance():
            prim.SetInstanceable(False)
    stage = Usd.Stage.Open(stage.Flatten())
    links = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(output))
    metadata = {
        "source": str(source),
        "links": sorted(p.GetName() for p in links),
        "joints": sorted(p.GetName() for p in joints),
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="~/LAIR-isaacsim/ultra_isaac/usd/ultra_bimanual/ultra_bimanual/ultra_bimanual.usda")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "assets/robots/ultra/ultra.usd"),
    )
    args = parser.parse_args()
    print(json.dumps(convert(args.source, args.output), indent=2))
