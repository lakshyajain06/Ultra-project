"""Check the converted articulation without starting Isaac Sim."""
from pathlib import Path
import unittest

from pxr import Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]


class UltraAssetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage = Usd.Stage.Open(str(ROOT / "assets/robots/ultra/ultra.usd"))

    def test_articulation_tree(self):
        root = self.stage.GetDefaultPrim()
        links = {p.GetPath() for p in self.stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        self.assertEqual(len(links), 25)
        children = []
        joints = [UsdPhysics.Joint(p) for p in self.stage.Traverse() if p.IsA(UsdPhysics.Joint)]
        fixed = [joint for joint in joints if joint.GetPrim().IsA(UsdPhysics.FixedJoint)]
        driven = [joint for joint in joints if not joint.GetPrim().IsA(UsdPhysics.FixedJoint)]
        self.assertEqual(len(fixed), 1)
        self.assertEqual(len(driven), 24)
        for joint in driven:
            parent, child = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            self.assertEqual(len(parent), 1)
            self.assertEqual(len(child), 1)
            self.assertIn(parent[0], links)
            self.assertIn(child[0], links)
            children.extend(child)
        self.assertEqual(len(set(children)), 24)
        root_link = root.GetPath().AppendPath("Geometry/world")
        self.assertEqual(links - set(children), {root_link})
        self.assertTrue(self.stage.GetPrimAtPath(root.GetPath().AppendChild("Geometry")).HasAPI(
            UsdPhysics.ArticulationRootAPI
        ))

    def test_geometry_and_cameras_survived(self):
        prims = list(self.stage.Traverse())
        self.assertGreater(sum(p.IsA(UsdGeom.Mesh) for p in prims), 20)
        self.assertGreater(sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in prims), 20)
        cameras = {p.GetName() for p in prims if p.IsA(UsdGeom.Camera)}
        self.assertTrue({"la_wrist_fisheye", "ra_wrist_fisheye", "zed_left"} <= cameras)
        self.assertFalse(any(p.IsInstance() for p in prims))

    def test_relationships_resolve(self):
        for prim in self.stage.Traverse():
            for rel in prim.GetRelationships():
                for target in rel.GetTargets():
                    self.assertTrue(self.stage.GetObjectAtPath(target), f"{rel.GetPath()} -> {target}")


if __name__ == "__main__":
    unittest.main()
