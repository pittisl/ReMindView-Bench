# === CLI + Visibility Aware Camera Placement Script ===
import bpy
import math
import os
import numpy as np
import argparse
import json
import csv
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
from bpy_extras.mesh_utils import edge_loops_from_edges
import bmesh

# === CONFIGURATION DEFAULTS (view-centric only) ===
default_config = {
    "elevation_deg": 15.0,
    "init_distance_factor": 1.2,
    "resolution": 10,
    # full-visibility control for view-centric start view
    "full_visibility_threshold": 0.8,     # ~fully visible (volume ratio)
    # object "front" definition
    "front_axis": "Y",                     # "X" or "Y" (Z is unusual)
    "front_flip": False,                   # True to invert the front direction
    # yaw sweep (constant distance) search params
    "sweep_step_deg": 10,                  # angular step for search
    "sweep_max_deg": 360,                   # max sweep extent
    "samples": 128,
    "resolution_x": 1024,
    "resolution_y": 768
}

# === UTILITY FUNCTIONS ===
def get_world_bounding_box_center_and_size(obj):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(corners, Vector()) / 8
    min_xy = Vector((min(c.x for c in corners), min(c.y for c in corners)))
    max_xy = Vector((max(c.x for c in corners), max(c.y for c in corners)))
    diag_xy = (max_xy - min_xy).length
    return center, diag_xy

def look_at_matrix(camera_pos, target_pos, global_up=Vector((0, 0, 1))):
    forward = (target_pos - camera_pos).normalized()
    if abs(forward.dot(global_up)) > 0.999:
        global_up = Vector((0, 1, 0))
    right = forward.cross(global_up).normalized()
    up = right.cross(forward).normalized()
    rot = Matrix((right, up, -forward)).transposed()
    return Matrix.Translation(camera_pos) @ rot.to_4x4()

def yaw_rotate_keep_pitch(base_matrix, yaw_rad, target_pos, global_up=Vector((0,0,1))):
    """
    Keep camera position fixed and preserve the original pitch (look-down angle)
    while yaw-rotating around world Z. Uses the base 'look-at' as reference.
    """
    pos = base_matrix.to_translation()
    f0 = (target_pos - pos).normalized()   # camera->target (base)
    z = f0.z
    h = max(1e-8, math.sqrt(max(0.0, 1.0 - z*z)))
    dir_xy = Vector((f0.x, f0.y))
    if dir_xy.length < 1e-8:
        dir_xy = Vector((1.0, 0.0))
    dir_xy.normalize()
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    rot_xy = Vector((c*dir_xy.x - s*dir_xy.y, s*dir_xy.x + c*dir_xy.y))
    new_forward = Vector((rot_xy.x * h, rot_xy.y * h, z)).normalized()
    right = new_forward.cross(global_up).normalized()
    up = right.cross(new_forward).normalized()
    R = Matrix((right, up, -new_forward)).transposed()
    return Matrix.Translation(pos) @ R.to_4x4()

def clear_camera_and_data(name):
    obj = bpy.data.objects.get(name)
    if obj:
        if obj.type == 'CAMERA':
            cam_data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if cam_data.users == 0:
                bpy.data.cameras.remove(cam_data)
        else:
            bpy.data.objects.remove(obj, do_unlink=True)

def create_camera(name):
    cam_data = bpy.data.cameras.new(name)
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam_obj)
    return cam_obj

def get_floor_polygon_2d(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    mesh.transform(obj.matrix_world)

    edge_face_counts = {i: 0 for i in range(len(mesh.edges))}
    for poly in mesh.polygons:
        for edge_idx in poly.edge_keys:
            for i, e in enumerate(mesh.edges):
                if set(edge_idx) == set((e.vertices[0], e.vertices[1])):
                    edge_face_counts[i] += 1
                    break

    boundary_edge_indices = [i for i, count in edge_face_counts.items() if count == 1]
    boundary_edges = [mesh.edges[i] for i in boundary_edge_indices]
    loops = edge_loops_from_edges(mesh, boundary_edges)
    largest_loop = max(loops, key=len)
    verts = [mesh.vertices[i].co for i in largest_loop]
    return [Vector((v.x, v.y)) for v in verts]

def is_point_inside_polygon_2d(point, polygon):
    x, y = point.x, point.y
    inside = False
    n = len(polygon)
    for i in range(n):
        xi, yi = polygon[i].x, polygon[i].y
        xj, yj = polygon[(i + 1) % n].x, polygon[(i + 1) % n].y
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
            inside = not inside
    return inside

def is_point_inside_object_bbox(obj, point):
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    min_c = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    max_c = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return all(min_c[i] <= point[i] <= max_c[i] for i in range(3))

def find_object_containing_point(point, object_list):
    for obj in object_list:
        if is_point_inside_object_bbox(obj, point):
            return obj
    return None

def get_all_nonstructural_objects(exclude=("floor", "camera", "light", "dining-room_0", "kitchen_0", "bathroom_0", "living-room_0", "bedroom_0")):
    return [obj for obj in bpy.data.objects
            if obj.type == 'MESH' and not any(k in obj.name.lower() for k in exclude)]

# === TOTAL CLEARANCE CHECK (no threshold) ===
def is_view_blocked(cam_loc, target_loc, target=None):
    """
    Return True if ANY mesh other than `target` intersects the ray from cam->target.
    Closest-hit logic ensures even a tiny occluder blocks the view.
    """
    direction = (target_loc - cam_loc).normalized()
    distance = (target_loc - cam_loc).length
    depsgraph = bpy.context.evaluated_depsgraph_get()
    hit, location, normal, index, hit_obj, _ = bpy.context.scene.ray_cast(
        depsgraph, cam_loc, direction, distance=distance
    )
    return bool(hit and hit_obj.type == 'MESH' and hit_obj != target)

# === Visibility function using actual mesh ===
def is_point_inside_mesh(pt, bvh, max_distance=1000.0):
    """Check if point is inside mesh using odd-even ray cast in +X direction."""
    loc, normal, index, dist = bvh.ray_cast(pt, Vector((1, 0, 0)), max_distance)
    if loc is None:
        return False
    intersections = 0
    origin = pt.copy()
    step = 0.001
    while True:
        hit = bvh.ray_cast(origin, Vector((1, 0, 0)), max_distance)
        if hit[0] is None:
            break
        intersections += 1
        origin = hit[0] + Vector((step, 0, 0))
    return intersections % 2 == 1

def sample_mesh_volume(obj, resolution=100):
    """Sample points inside mesh volume using BVH and odd-even ray method."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    mesh.transform(obj.matrix_world)

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bvh = BVHTree.FromBMesh(bm)

    verts_world = [v.co for v in bm.verts]
    min_c = Vector((min(v.x for v in verts_world),
                    min(v.y for v in verts_world),
                    min(v.z for v in verts_world)))
    max_c = Vector((max(v.x for v in verts_world),
                    max(v.y for v in verts_world),
                    max(v.z for v in verts_world)))

    points = []
    xs = np.linspace(min_c.x, max_c.x, resolution)
    ys = np.linspace(min_c.y, max_c.y, resolution)
    zs = np.linspace(min_c.z, max_c.z, resolution)

    for x in xs:
        for y in ys:
            for z in zs:
                pt = Vector((x, y, z))
                if is_point_inside_mesh(pt, bvh):
                    points.append(pt)

    bm.free()
    eval_obj.to_mesh_clear()
    return points

def compute_mesh_visibility_ratio(camera, obj, resolution=100):
    """Compute fraction of object volume visible in camera frustum (no occlusion)."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene
    proj_mat = camera.calc_matrix_camera(
        depsgraph,
        x=scene.render.resolution_x,
        y=scene.render.resolution_y
    )
    view_mat = camera.matrix_world.inverted()
    world_to_ndc = proj_mat @ view_mat

    points = sample_mesh_volume(obj, resolution)
    total_points = len(points)
    if total_points == 0:
        return 0.0

    visible_points = 0
    for pt in points:
        ndc = world_to_ndc @ pt.to_4d()
        if ndc.w == 0:
            continue
        ndc /= ndc.w
        if ndc.z >= 0 and -1 <= ndc.x <= 1 and -1 <= ndc.y <= 1 and ndc.z <= 1:
            visible_points += 1

    return visible_points / total_points

# ===== HELPERS FOR FRONT-ALIGNED START VIEW & CONSTANT-DISTANCE YAW SWEEP =====
def get_object_front_dir_world(obj, axis="Y"):
    """Return the object's 'front' direction projected to world XY and normalized."""
    axis = axis.upper()
    if axis == "X":
        local_dir = Vector((1, 0, 0))
    elif axis == "Y":
        local_dir = Vector((0, 1, 0))
    else:
        local_dir = Vector((0, 1, 0))
    world_dir = (obj.matrix_world.to_3x3() @ local_dir)
    world_dir.z = 0.0
    if world_dir.length < 1e-8:
        world_dir = Vector((0, 1, 0))
    world_dir.normalize()
    return world_dir

def rot_xy(v, angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return Vector((c*v.x - s*v.y, s*v.x + c*v.y))

def cam_loc_from_direction(target_center, dir_xy_unit, distance, elevation_rad):
    """Compute camera location at a given distance/elevation along a given XY direction from target."""
    xy_offset = math.cos(elevation_rad) * distance
    z_offset = math.sin(elevation_rad) * distance
    cam_xy = target_center.xy + dir_xy_unit.xy * xy_offset
    cam_z = target_center.z + z_offset
    return Vector((cam_xy.x, cam_xy.y, cam_z))

def find_start_view_by_yaw_sweep(
    target, target_center, base_dir_xy, distance, elevation_rad,
    floor_polygon, obstacles, vis_threshold,
    resolution=100, sweep_step_deg=10, sweep_max_deg=360
):
    """
    Keep distance & elevation fixed. Sweep yaw around target to find a facing view
    whose visibility >= vis_threshold. Start from object's 'front' and expand
    symmetrically (0, ±step, ±2*step, ...), up to sweep_max_deg.
    Returns (cam_loc, visibility, yaw_found) or (None, 0.0, None).
    """
    # Normalize base dir in XY
    d0 = base_dir_xy.copy()
    if d0.length < 1e-8:
        d0 = Vector((0, 1, 0))
    d0.z = 0.0
    d0.normalize()

    step = math.radians(max(1e-6, sweep_step_deg))
    max_yaw = math.radians(max(0.0, sweep_max_deg))

    # Generate symmetric yaw offsets: 0, +s, -s, +2s, -2s, ...
    offsets = [0.0]
    k = 1
    while k * step <= max_yaw + 1e-9:
        offsets.append(k * step)
        offsets.append(-k * step)
        k += 1

    # Fixed Z from elevation and distance
    xy_offset = math.cos(elevation_rad) * distance
    z_offset = math.sin(elevation_rad) * distance
    cam_z = target_center.z + z_offset

    for yaw in offsets:
        dir_xy = rot_xy(d0, yaw)
        cam_xy = target_center.xy + dir_xy.xy * xy_offset
        cam_loc = Vector((cam_xy.x, cam_xy.y, cam_z))

        # Floor & obstacle & occluder checks
        if not is_point_inside_polygon_2d(Vector((cam_xy.x, cam_xy.y)), floor_polygon):
            continue
        if find_object_containing_point(cam_loc, obstacles):
            continue
        if is_view_blocked(cam_loc, target_center, target):
            continue

        # Visibility for a look-at camera
        cam_tmp = create_camera("temp_vis_cam")
        cam_tmp.matrix_world = look_at_matrix(cam_loc, target_center)
        vis = compute_mesh_visibility_ratio(cam_tmp, target, resolution=resolution)
        clear_camera_and_data(cam_tmp.name)

        if vis >= vis_threshold:
            return cam_loc, vis, yaw

    return None, 0.0, None

# === RENDER & LOGGING PIPELINE (view-centric only) ===
def render_visibility_pipeline(scene_path, output_root, room_name, config):
    bpy.ops.wm.open_mainfile(filepath=scene_path)

    name_list = {"include": ["spawn_asset"], "exclude": ["carnivore", "ceilinglight", "herbivore", "pointlamp", "window", "door"]}
    distance_levels_pct = [i * 0.1 for i in range(11)]

    target_obj_name_list = [
        obj.name for obj in bpy.data.objects
        if all(k in obj.name.lower() for k in name_list["include"])
        and not any(k in obj.name.lower() for k in name_list["exclude"])
    ]
    floor_obj_name = [
        obj.name for obj in bpy.data.objects
        if "floor" in obj.name.lower() and room_name in obj.name.lower()
    ][0]

    for target_obj_name in target_obj_name_list:
        print(f"\n🔍 Processing {target_obj_name}")
        target = bpy.data.objects.get(target_obj_name)
        floor = bpy.data.objects.get(floor_obj_name)
        base_output_dir = bpy.path.abspath(os.path.join(output_root, f"{target_obj_name}"))
        if not target or not floor or os.path.exists(base_output_dir):
            print(f"⚠️  Skipping {base_output_dir} - already processed")
            continue

        os.makedirs(base_output_dir, exist_ok=True)

        target_center, target_diag_xy = get_world_bounding_box_center_and_size(target)
        floor_center, floor_diag_xy = get_world_bounding_box_center_and_size(floor)
        max_factor = (floor_diag_xy / target_diag_xy)**0.5
        elevation_rad = math.radians(config["elevation_deg"])
        floor_polygon = get_floor_polygon_2d(floor)
        obstacles = get_all_nonstructural_objects()

        log_entry = []

        for level_idx, pct in enumerate(distance_levels_pct):
            distance_factor = (max_factor - config["init_distance_factor"]) * pct
            distance = target_diag_xy * (distance_factor + config["init_distance_factor"])
            level_name = f"level_{int(pct*100)}"
            render_dir = os.path.join(base_output_dir, level_name)
            os.makedirs(render_dir, exist_ok=True)

            # Clear any previously existing cameras (common names)
            for cam in ["camera_0_0", "camera_0_1", "camrig.0", "camera_0", "camera_1", "camera_2", "camera_3"]:
                clear_camera_and_data(cam)

            # 1) Determine object's "front" in world XY
            front_dir = get_object_front_dir_world(
                target,
                axis=config.get("front_axis", "Y"),
            )

            # 2) Constant-distance yaw sweep to find a fully visible facing start view
            cam_loc, vis0, yaw0 = find_start_view_by_yaw_sweep(
                target=target,
                target_center=target_center,
                base_dir_xy=front_dir,
                distance=distance,
                elevation_rad=elevation_rad,
                floor_polygon=floor_polygon,
                obstacles=obstacles,
                vis_threshold=config.get("full_visibility_threshold", 0.8),
                resolution=max(config.get("resolution", 10), 20),
                sweep_step_deg=config.get("sweep_step_deg", 10),
                sweep_max_deg=config.get("sweep_max_deg", 360)
            )
            if cam_loc is None:
                print(f"⚠️  No constant-distance view meets visibility for {target_obj_name} at {level_name}")
                continue

            # 3) Base orientation (facing target with preserved pitch)
            cam_base = create_camera("camera_0")
            base_matrix = look_at_matrix(cam_loc, target_center)
            cam_base.matrix_world = base_matrix

            # 4) Render 4 orientations (0°, 90°, 180°, 270°), same position & pitch
            yaw_steps = [0.0, 0.5*math.pi, math.pi, 1.5*math.pi]
            for i, yaw in enumerate(yaw_steps):
                view_name = f"{i}"
                cam_name = f"camera_{i}"
                if i == 0:
                    cam = cam_base
                else:
                    cam = create_camera(cam_name)
                cam.matrix_world = yaw_rotate_keep_pitch(base_matrix, yaw, target_center)
                cam.scale = Vector((1, 1, 1))

                bpy.context.scene.camera = cam
                bpy.context.scene.render.engine = 'CYCLES'
                bpy.context.scene.cycles.device = 'CPU'
                bpy.context.scene.cycles.samples = config["samples"]
                bpy.context.scene.cycles.use_denoising = True
                bpy.context.scene.render.resolution_x = config["resolution_x"]
                bpy.context.scene.render.resolution_y = config["resolution_y"]
                bpy.context.scene.render.resolution_percentage = 100
                bpy.context.scene.use_nodes = False
                bpy.context.scene.render.image_settings.file_format = 'PNG'
                bpy.context.scene.render.filepath = os.path.join(render_dir, f"frame_{view_name}.png")
                try:
                    bpy.ops.render.render(write_still=True)
                except Exception as e:
                    print(f"❌ Render failed: {e}")

                cam_rot_quat = cam.matrix_world.to_quaternion()
                log_entry.append({
                    "level": level_name,
                    "view": view_name,
                    "pose": list(cam_loc),             # same position for all 4
                    "rot": list(cam_rot_quat),
                    # visibility only meaningful for the facing view (0)
                    "visibility": round(vis0 if i == 0 else -1.0, 4)
                })

        # Write CSV once per target
        csv_path = os.path.join(base_output_dir, "camera_poses.csv")
        with open(csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["level", "view", "x", "y", "z", "rot_w", "rot_x", "rot_y", "rot_z", "visibility"])
            for view in log_entry:
                pose = view["pose"]
                rotation = view["rot"]
                writer.writerow([
                    view["level"], view["view"],
                    *pose, *rotation,
                    view["visibility"]
                ])

# === MAIN ENTRY POINT ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", required=True, help="Path to .blend file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--room_type", default="dining-room", help="Room name keyword")
    parser.add_argument("--config", default=None, type=str, help="Path to JSON config file")
    args = parser.parse_args()

    my_cwd = os.getcwd()
    scene_path = os.path.join(my_cwd, args.scene_path)
    output_dir = os.path.join(my_cwd, args.output_dir)
    config = default_config.copy()

    if args.config:
        with open(args.config) as f:
            user_config = json.load(f)
        config.update(user_config)

    render_visibility_pipeline(scene_path, output_dir, args.room_type, config)

if __name__ == "__main__":
    main()
