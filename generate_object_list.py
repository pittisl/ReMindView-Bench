import csv
import math
import random
from pathlib import Path
import bpy
import bmesh
from mathutils import Quaternion, Vector
from bpy_extras.object_utils import world_to_camera_view
from view_reconstruct import reconstruct_views_from_csv

# -------------------------- Scene/Camera Utilities -------------------------- #
def _camera_forward(cam_obj: bpy.types.Object) -> Vector:
    # Blender camera looks along -Z in its local space
    return cam_obj.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))

def _scene_bbox_world_corners(objects) -> list[Vector]:
    corners = []
    for o in objects:
        if o.type != 'MESH':
            continue
        # object's local-space bbox (8 points)
        for c in o.bound_box:
            corners.append(o.matrix_world @ Vector(c))
    return corners

def ensure_camera_clip_for_scene(cam_obj: bpy.types.Object,
                                 objects,
                                 margin: float = 1.2,
                                 min_clip_start: float = 1e-3):
    """Auto-extend camera clip range to encompass the whole scene from this camera."""
    if cam_obj.type != 'CAMERA':
        return
    cam_data = cam_obj.data
    cam_loc = cam_obj.matrix_world.translation

    corners = _scene_bbox_world_corners(objects)
    if not corners:
        return

    # Distance to farthest bbox corner
    max_d = 0.0
    for w in corners:
        d = (w - cam_loc).length
        if d > max_d:
            max_d = d

    # extend clip_end if needed; keep clip_start small
    cam_data.clip_start = min(cam_data.clip_start, min_clip_start)
    cam_data.clip_end = max(cam_data.clip_end, max_d * margin)


# -------------------------- Visibility Estimation -------------------------- #
def _prep_target_triangles_world(target_obj: bpy.types.Object, depsgraph):
    """Triangulate evaluated mesh and return (eval_obj, eval_mesh, triangles_world, areas)."""
    eval_obj = target_obj.evaluated_get(depsgraph)
    eval_me = eval_obj.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)

    bm = bmesh.new()
    bm.from_mesh(eval_me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(eval_me)
    bm.free()

    world_mat = target_obj.matrix_world
    verts_world = [world_mat @ v.co for v in eval_me.vertices]

    tris, areas = [], []
    for p in eval_me.polygons:
        if len(p.vertices) != 3:
            continue
        i0, i1, i2 = p.vertices
        a, b, c = verts_world[i0], verts_world[i1], verts_world[i2]
        area = ((b - a).cross(c - a)).length * 0.5
        if area > 0:
            tris.append((a, b, c))
            areas.append(area)

    return eval_obj, eval_me, tris, areas


def _free_eval_mesh(eval_obj: bpy.types.Object, eval_me: bpy.types.Mesh):
    try:
        eval_obj.to_mesh_clear()
    except Exception:
        pass


def _sample_point_on_triangle(a: Vector, b: Vector, c: Vector) -> Vector:
    r1, r2 = random.random(), random.random()
    sr1 = math.sqrt(r1)
    u = 1.0 - sr1
    v = sr1 * (1.0 - r2)
    w = sr1 * r2
    return u * a + v * b + w * c


def _point_visible_from_camera(scene,
                               depsgraph,
                               cam_obj: bpy.types.Object,
                               target_obj: bpy.types.Object,
                               point_world: Vector,
                               eps: float = 1e-4) -> bool:
    cam_loc = cam_obj.matrix_world.translation
    vec = point_world - cam_loc
    dist = vec.length
    if dist <= 1e-8:
        return False
    direction = vec / dist

    # In-front check (avoid rejecting due to clip z): point must be in front of camera
    if direction.dot(_camera_forward(cam_obj)) <= 0:
        return False

    # 2D in-frame check (x,y within [0,1]); don't enforce z in [0,1] here
    co_ndc = world_to_camera_view(scene, cam_obj, point_world)
    if not (0.0 <= co_ndc.x <= 1.0 and 0.0 <= co_ndc.y <= 1.0):
        return False

    # Raycast: is the first hit the target?
    hit, loc, normal, face_idx, hit_obj, _ = scene.ray_cast(depsgraph, cam_loc, direction, distance=max(dist - eps, 0.0))
    if not hit:
        return False

    hit_root = getattr(hit_obj, "original", hit_obj)
    target_root = getattr(target_obj, "original", target_obj)
    return hit_root == target_root


def estimate_visibility_ratio_for_object(scene,
                                         cam_obj,
                                         target_obj,
                                         samples: int = 1000) -> tuple[float, int, int]:
    """Return (ratio, visible_samples, total_samples) for one object from one camera."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    if target_obj.type != 'MESH':
        return 0.0, 0, 0

    eval_obj, eval_me, tris, areas = _prep_target_triangles_world(target_obj, depsgraph)
    try:
        if not tris:
            return 0.0, 0, 0

        cum, total_area = [], 0.0
        for a in areas:
            total_area += a
            cum.append(total_area)

        vis = 0
        for _ in range(samples):
            r = random.random() * total_area
            lo, hi = 0, len(cum) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if cum[mid] < r:
                    lo = mid + 1
                else:
                    hi = mid
            a, b, c = tris[lo]
            p = _sample_point_on_triangle(a, b, c)
            if _point_visible_from_camera(scene, depsgraph, cam_obj, target_obj, p):
                vis += 1
        return (vis / float(samples)), vis, samples
    finally:
        _free_eval_mesh(eval_obj, eval_me)


def save_visibility_csv_all_objects(
    output_csv_path: Path,
    samples_per_object: int = 800,
    include_hidden: bool = False,
    exclude_names_prefix: tuple[str, ...] = ("reconstructed_",),
    min_ratio: float = 0.10,
    max_ratio: float = 1.00,
    auto_extend_clip: bool = True,
    clip_margin: float = 1.2,
):
    """
    For each camera, compute visibility for EVERY mesh object and save only rows
    whose visible_ratio is within [min_ratio, max_ratio].

    Robust for far views: optionally auto-extends camera clip to cover scene,
    and uses an in-front + XY in-frame test (no hard z rejection).
    """
    scene = bpy.context.scene

    # cameras to evaluate (reconstructed cameras start with "camera_")
    cams = [o for o in scene.objects if o.type == 'CAMERA' and o.name.startswith("camera_")]
    if not cams:
        cams = [o for o in scene.objects if o.type == 'CAMERA']

    # candidate mesh objects (filter by visibility unless include_hidden=True)
    def _obj_ok(o: bpy.types.Object) -> bool:
        if o.type != 'MESH':
            return False
        if not include_hidden and not o.visible_get():
            return False
        for pref in exclude_names_prefix:
            if o.name.startswith(pref):
                return False
        return True

    mesh_objects = [o for o in scene.objects if _obj_ok(o)]
    if not mesh_objects:
        print("⚠️ No mesh objects found to evaluate.")
        mesh_objects = []

    rows = []

    for cam in cams:
        kept = 0
        skipped = 0
        # Auto-extend clip range so distant objects aren't culled by clip_end
        if auto_extend_clip:
            ensure_camera_clip_for_scene(cam, mesh_objects, margin=clip_margin)

        print(f"\n=== Visibile object checking for {cam.name} ... ===")
        for obj in mesh_objects:
            if "spawn_asset" in obj.name:
                ratio, vis_cnt, n = estimate_visibility_ratio_for_object(
                    scene, cam, obj, samples=samples_per_object
                )
                if min_ratio <= ratio <= max_ratio:
                    kept += 1
                    rows.append({
                        "camera": cam.name,
                        "object": obj.name,
                        "visible_ratio": f"{ratio:.4f}",
                        "visible_samples": vis_cnt,
                        "total_samples": n
                    })
                else:
                    skipped += 1
        print(f"kept: {kept}  |  skipped: {skipped}")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["camera","object","visible_ratio","visible_samples","total_samples"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Saved filtered per-object visibility report: {output_csv_path}")


# ------------------------------- Orchestrator ------------------------------- #
def run_reconstruct_and_visibility_all(
    csv_path: Path,
    input_blend_path: Path,
    target_level: str,
    vis_csv_out: Path,
    samples_per_object: int = 800,
    include_hidden: bool = False,
    min_ratio: float = 0.10,
    max_ratio: float = 1.00,
    auto_extend_clip: bool = True,
    clip_margin: float = 1.2,
):
    # Reconstruct cameras
    created = reconstruct_views_from_csv(
        csv_path=csv_path,
        input_blend_path=input_blend_path,
        target_level=target_level,
    )

    # Compute & save visibility for all mesh objects per camera (filtered)
    save_visibility_csv_all_objects(
        output_csv_path=vis_csv_out,
        samples_per_object=samples_per_object,
        include_hidden=include_hidden,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        auto_extend_clip=auto_extend_clip,
        clip_margin=clip_margin,
    )

    print(f"🎉 Done. Cameras processed: {len(created)}")


# --------------------------------- __main__ -------------------------------- #
if __name__ == "__main__":
    # python -m generate_object_list --input_dir view_centric_view_frame_outputs --room_dir Bedroom
    import argparse

    parser = argparse.ArgumentParser(
        description="Reconstruct cameras from CSV and compute per-object visibility per frame, saving only objects within an appearance ratio range. Handles far views robustly."
    )
    parser.add_argument("--input_dir", type=str, default="view_centric_view_frame_outputs",)
    parser.add_argument("--room_dir", type=str, default="Bedroom",)
    parser.add_argument("--scene_dir", type=str, default="outputs/indoors",)
    parser.add_argument("--samples", type=int, default=1000,
                        help="Samples per object per camera (increase for accuracy)")
    parser.add_argument("--include_hidden", action="store_true",
                        help="Include objects hidden in the viewport (default off)")
    parser.add_argument("--min_ratio", type=float, default=0.1,
                        help="Minimum appearance ratio (inclusive). Example: 0.10 = 10%%")
    parser.add_argument("--max_ratio", type=float, default=1.00,
                        help="Maximum appearance ratio (inclusive). Example: 1.0 = 100%%")
    parser.add_argument("--no_auto_extend_clip", action="store_true",
                        help="Disable auto-extending camera clip range")
    parser.add_argument("--clip_margin", type=float, default=1.0,
                        help="Multiplier for clip_end relative to farthest scene point")
    args, _ = parser.parse_known_args()

    room_names = {"DiningRoom", "Bedroom", "Kitchen", "LivingRoom", "Bathroom"}

    input_dir = (Path(args.input_dir) / args.room_dir).resolve()
    blender_dir = Path(args.scene_dir).resolve()
    scene_dir_paths = [scene_dir_path for scene_dir_path in sorted(input_dir.iterdir()) if (scene_dir_path.is_dir() and scene_dir_path.stem.split("_")[0] in room_names)]
    for scene_dir in scene_dir_paths:
        object_dir_paths = [object_dir_path for object_dir_path in sorted(scene_dir.iterdir()) if object_dir_path.is_dir()]
        for object_dir in object_dir_paths:
            for level_folder_path in object_dir.rglob("level_*"):
                object_name = level_folder_path.parent.name
                csv_path = level_folder_path.parent / "camera_poses.csv"
                room_name = level_folder_path.parent.parent.stem
                scene_path = blender_dir / room_name / "scene.blend"
                target_level = level_folder_path.stem
                vis_csv_out = level_folder_path / "object_visibility.csv"

                print(f"\n{'#'*10} Processing object {object_name} at {target_level} in {room_name} {10*'#'}")
                if vis_csv_out.exists():
                    print(f"skipping {vis_csv_out} (already exists)")
                    continue

                run_reconstruct_and_visibility_all(
                    csv_path=csv_path,
                    input_blend_path=scene_path,
                    target_level=target_level,
                    vis_csv_out=vis_csv_out,
                    samples_per_object=args.samples,
                    include_hidden=bool(args.include_hidden),
                    min_ratio=float(args.min_ratio),
                    max_ratio=float(args.max_ratio),
                    auto_extend_clip=(not args.no_auto_extend_clip),
                    clip_margin=float(args.clip_margin),
                )