"""Inspect converted USDs without launching Kit.

Verifies what physics the conversion actually authored — a converter exit code
of 0 does not prove the SDF collision survived, so this reads the stage back.

Usage: python3 inspect_usd.py <file.usd> [<file.usd> ...]
"""
import glob
import os
import sys

# pxr ships inside Isaac Sim's extension cache; import it directly rather than
# booting the whole Kit app (which hijacks stdout).
_ISAACSIM = os.environ.get(
    "ISAACSIM_PATH",
    "/home/pk/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim",
)
for _cand in glob.glob(os.path.join(_ISAACSIM, "extscache", "omni.usd.libs-*")):
    if os.path.isdir(os.path.join(_cand, "pxr")):
        sys.path.insert(0, _cand)
        _bin = os.path.join(_cand, "bin")
        if os.path.isdir(_bin):
            os.environ["LD_LIBRARY_PATH"] = _bin + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        break

from pxr import Usd, UsdGeom  # noqa: E402

TRACKED = (
    "physxCollision:contactOffset",
    "physxCollision:restOffset",
    "physxSDFMeshCollision:sdfResolution",
    "physxSDFMeshCollision:sdfBitsPerSubgridPixel",
    "physxSDFMeshCollision:sdfSubgridResolution",
)
SCALARS = ("physics:mass", "physics:rigidBodyEnabled", "physics:collisionEnabled")

for path in sys.argv[1:]:
    print("=" * 78)
    print("FILE:", os.path.basename(path))
    stage = Usd.Stage.Open(path)
    dp = stage.GetDefaultPrim()
    print("default prim:", dp.GetPath() if dp else None)
    print("up axis:", UsdGeom.GetStageUpAxis(stage),
          "| meters/unit:", UsdGeom.GetStageMetersPerUnit(stage))
    for prim in stage.Traverse():
        schemas = [s for s in prim.GetAppliedSchemas()
                   if any(k in s for k in ("Physics", "Physx", "Collision", "Mass"))]
        tag = f" <{prim.GetTypeName()}>" if prim.GetTypeName() else ""
        print(f"{prim.GetPath()}{tag}" + (f"  APIs={schemas}" if schemas else ""))
        a = prim.GetAttribute("physics:approximation")
        if a and a.HasAuthoredValue():
            print(f"    physics:approximation = {a.Get()}")
        for name in TRACKED + SCALARS:
            at = prim.GetAttribute(name)
            if at and at.HasAuthoredValue():
                print(f"    {name} = {at.Get()}")
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            pts = m.GetPointsAttr().Get()
            fvc = m.GetFaceVertexCountsAttr().Get()
            print(f"    mesh: {len(pts) if pts else 0} points, {len(fvc) if fvc else 0} faces")
