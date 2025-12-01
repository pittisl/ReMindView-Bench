import bpy
import math
from mathutils import Vector
from typing import Tuple
from pathlib import Path
import json


def orientation_object_name_filtering(object_list, orientation_object_name_list):
    return list(set(object_list) & set(orientation_object_name_list))


def object_name_mapping(object_blender_name, object_name_dict, room_name_path):
    object_blender_name = str(room_name_path).split("_", maxsplit=2)[0] + "_object_" + str(room_name_path).split("_", maxsplit=2)[-1] + "/" + str(object_blender_name)
    object_natural_name = object_name_dict.get(object_blender_name, object_blender_name.split("/")[-1].split("Factory")[0])
    object_natural_name = object_natural_name.lower()
    return object_natural_name


def load_json(json_path):
    json_path = Path(json_path)
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            json_file_content = json.load(f)
        return json_file_content
    else:
        print("[Error]: json path does not exist")


def object_view_visible_dict_generation(
    ref_obj_name: str,
    candidate_names: list[str],
    turn_direction: str | None = None,
    *,
    xy_iou_vertical_thresh: float = 0.05,
    treat_any_xy_intersection_as_vertical: bool = True,
    aabb_pad: float = 0.0,
) -> dict:
    """
    Classify candidate objects around 'ref_obj_name' into 5 classes on the XY plane using
    positions + reference forward direction (no visibility):
      - front: (-45, 45)
      - right: [45, 135)
      - back:  (-180, -135] U [135, 180)
      - left:  [-135, -45]
      - vertical: XY AABBs (with padding) intersect or IoU >= xy_iou_vertical_thresh

    Forward:
      - CAMERA: -Z
      - Others: local +X (projected to XY)

    Args:
        ref_obj_name: name of the reference object.
        candidate_names: list of object names to classify (ref is skipped if included).
        turn_direction: optional virtual turn of the reference frame:
                        {"left": +90, "right": -90, "back": 180} or None.
        xy_iou_vertical_thresh: IoU threshold for vertical classification (if not using any-intersection).
        treat_any_xy_intersection_as_vertical: if True, any XY intersection => 'vertical'.
        aabb_pad: uniform padding (world units) applied to XY AABBs before intersection tests.

    Returns:
        dict: {"front": [...], "right": [...], "back": [...], "left": [...], "vertical": [...]}
    """
    turn_to_degree = {"left": 90.0, "right": -90.0, "back": 180.0, None: 0.0}

    ref_obj = bpy.data.objects.get(ref_obj_name)
    if ref_obj is None:
        raise KeyError(f"Reference object not found: {ref_obj_name}")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    ref_eval = ref_obj.evaluated_get(depsgraph)

    # Reference world position and forward
    p_ref = ref_eval.matrix_world.translation
    q_ref = ref_eval.matrix_world.to_quaternion()

    if ref_eval.type == "CAMERA":
        fwd_ws = q_ref @ Vector((0.0, 0.0, -1.0))
    else:
        fwd_ws = q_ref @ Vector((1.0, 0.0, 0.0))

    fwd_xy = Vector((fwd_ws.x, fwd_ws.y))
    if fwd_xy.length < 1e-8:
        fwd_xy = Vector((0.0, 1.0))  # fallback if forward is vertical

    # Optional virtual turn
    extra = turn_to_degree.get(turn_direction, 0.0)

    # Precompute ref XY AABB (padded)
    from typing import Tuple  # for type hints only
    def _world_aabb_local(obj) -> tuple[float, float, float, float, float, float]:
        eo = obj.evaluated_get(depsgraph)
        mw = eo.matrix_world
        xs, ys, zs = [], [], []
        for corner in eo.bound_box:
            c = mw @ Vector(corner)
            xs.append(c.x); ys.append(c.y); zs.append(c.z)
        return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))

    def _pad_xy(a: tuple[float, float, float, float], pad: float) -> tuple[float, float, float, float]:
        (x0, x1, y0, y1) = a
        return (x0 - pad, x1 + pad, y0 - pad, y1 + pad)

    def _xy_intersection(a: tuple[float, float, float, float],
                         b: tuple[float, float, float, float]) -> float:
        (ax0, ax1, ay0, ay1) = a
        (bx0, bx1, by0, by1) = b
        iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        ih = max(0.0, min(ay1, by1) - max(ay0, by0))
        return iw * ih

    def _xy_area(a: tuple[float, float, float, float]) -> float:
        (x0, x1, y0, y1) = a
        return max(0.0, x1 - x0) * max(0.0, y1 - y0)

    def _xy_iou(a: tuple[float, float, float, float],
                b: tuple[float, float, float, float]) -> float:
        inter = _xy_intersection(a, b)
        if inter <= 0.0:
            return 0.0
        union = _xy_area(a) + _xy_area(b) - inter
        return inter / union if union > 0.0 else 0.0

    ax0, ax1, ay0, ay1, _, _ = _world_aabb_local(ref_obj)
    ref_xy = _pad_xy((ax0, ax1, ay0, ay1), aabb_pad)

    buckets = {"front": [], "right": [], "back": [], "left": [], "vertical": []}

    for name in candidate_names:
        if name == ref_obj_name:
            continue
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue

        obj_eval = obj.evaluated_get(depsgraph)
        p_obj = obj_eval.matrix_world.translation

        # ---------- vertical check via XY AABB overlap ----------
        bx0, bx1, by0, by1, _, _ = _world_aabb_local(obj)
        obj_xy = _pad_xy((bx0, bx1, by0, by1), aabb_pad)
        inter_area = _xy_intersection(ref_xy, obj_xy)
        if inter_area > 0.0:
            if treat_any_xy_intersection_as_vertical or _xy_iou(ref_xy, obj_xy) >= xy_iou_vertical_thresh:
                buckets["vertical"].append(name)
                continue

        # ---------- angular sectoring on XY ----------
        rel_xy = Vector((p_obj.x - p_ref.x, p_obj.y - p_ref.y))
        if rel_xy.length == 0.0:
            # Same XY center; most likely vertical relation -> classify as vertical
            buckets["vertical"].append(name)
            continue

        ang_rel = math.degrees(math.atan2(rel_xy.y, rel_xy.x))
        ang_fwd = math.degrees(math.atan2(fwd_xy.y, fwd_xy.x))

        # Relative angle in (-180, 180]
        rel_deg = (ang_rel - ang_fwd + 180.0) % 360.0 - 180.0
        rel_deg = -rel_deg
        rel_deg += extra
        if rel_deg > 180.0:
            rel_deg -= 360.0
        if rel_deg < -180.0:
            rel_deg += 360.0

        # 4-way sectoring (90° each)
        if -60.0 <= rel_deg <= 60.0:
            buckets["front"].append(name)
        if 30.0 <= rel_deg <= 150.0:
            buckets["right"].append(name)
        if rel_deg >= 120.0 or rel_deg <= -120.0:
            buckets["back"].append(name)
        if -150 <= rel_deg <= -30:
            buckets["left"].append(name)

    return buckets



def _world_aabb(obj) -> Tuple[float, float, float, float, float, float]:
    """
    World-space axis-aligned bounding box (AABB) for a Blender object.
    Returns (minx, maxx, miny, maxy, minz, maxz).
    Uses evaluated depsgraph to honor constraints/parenting.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eo = obj.evaluated_get(depsgraph)
    mw = eo.matrix_world
    xs, ys, zs = [], [], []
    for corner in eo.bound_box:
        c = mw @ Vector(corner)
        xs.append(c.x); ys.append(c.y); zs.append(c.z)
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def _xy_intersection(a: Tuple[float, float, float, float],
                     b: Tuple[float, float, float, float]) -> float:
    """
    Intersection area of two XY AABBs (minx,maxx,miny,maxy).
    """
    (ax0, ax1, ay0, ay1) = a
    (bx0, bx1, by0, by1) = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    return iw * ih


def _xy_area(a: Tuple[float, float, float, float]) -> float:
    (x0, x1, y0, y1) = a
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _xy_iou(a: Tuple[float, float, float, float],
            b: Tuple[float, float, float, float]) -> float:
    inter = _xy_intersection(a, b)
    if inter <= 0.0:
        return 0.0
    union = _xy_area(a) + _xy_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def _pad_xy(a: Tuple[float, float, float, float], pad: float) -> Tuple[float, float, float, float]:
    """
    Uniformly grows/shrinks an XY AABB by 'pad' in world units.
    """
    (x0, x1, y0, y1) = a
    return (x0 - pad, x1 + pad, y0 - pad, y1 + pad)


def direction_between_objects(
    obj1_name: str,
    obj2_name: str,
    obj3_name: str = None,
    turn_direction=None,
    *,
    xy_iou_vertical_thresh: float = 0.05,
    treat_any_xy_intersection_as_vertical: bool = True,

    aabb_pad: float = 0.0
) -> str:
    """
    8-way direction of obj2 relative to obj1 on the XY plane, but using obj3's orientation
    as the reference frame.

    - Positions: uses obj1 and obj2 world positions.
    - Orientation: uses obj3 world orientation (CAMERA -> -Z forward; others -> +X forward).
      If obj3_name is None, falls back to obj1's orientation (backward compatible).

    Vertical-only exclusion:
      Projects both objects' world AABBs onto the XY plane and checks overlap.
      If their XY AABBs intersect (or IoU exceeds 'xy_iou_vertical_thresh'),
      returns 'vertical' to indicate top/bottom/inside relationships.

    Returns one of:
      'front', 'front-right', 'right', 'back-right',
      'back', 'back-left', 'left', 'front-left',
      'vertical' (excluded), or 'undefined'.
    """
    turn_to_degree = {"left": 90, "right": -90, "back": 180}

    obj1 = bpy.data.objects.get(obj1_name)
    obj2 = bpy.data.objects.get(obj2_name)
    if obj1 is None or obj2 is None:
        raise KeyError(f"Object not found: {obj1_name} or {obj2_name}")

    obj3 = None
    if obj3_name is not None:
        obj3 = bpy.data.objects.get(obj3_name)
        if obj3 is None:
            raise KeyError(f"Object not found: {obj3_name}")

    (ax0, ax1, ay0, ay1, _, _) = _world_aabb(obj1)
    (bx0, bx1, by0, by1, _, _) = _world_aabb(obj2)

    a_xy = _pad_xy((ax0, ax1, ay0, ay1), aabb_pad)
    b_xy = _pad_xy((bx0, bx1, by0, by1), aabb_pad)

    inter_area = _xy_intersection(a_xy, b_xy)
    if inter_area > 0.0:
        if treat_any_xy_intersection_as_vertical:
            return "vertical"
        if _xy_iou(a_xy, b_xy) >= xy_iou_vertical_thresh:
            return "vertical"

    depsgraph = bpy.context.evaluated_depsgraph_get()
    o1 = obj1.evaluated_get(depsgraph)
    o2 = obj2.evaluated_get(depsgraph)

    p1 = o1.matrix_world.translation
    p2 = o2.matrix_world.translation

    rel_xy = Vector((p2.x - p1.x, p2.y - p1.y))
    if rel_xy.length == 0:
        return "vertical"

    o_ref = obj3.evaluated_get(depsgraph) if obj3 is not None else o1
    q_ref = o_ref.matrix_world.to_quaternion()

    ref_type = obj3.type if obj3 is not None else obj1.type
    if ref_type == "CAMERA":
        fwd_ws = (q_ref @ Vector((0.0, 0.0, -1.0)))  
    else:
        fwd_ws = q_ref @ Vector((1.0, 0.0, 0.0))

    fwd_xy = Vector((fwd_ws.x, fwd_ws.y))
    if fwd_xy.length < 1e-8:
        fwd_xy = Vector((0.0, 1.0))

    up_ws   = Vector((0.0, 0.0, 1.0))
    right_ws = up_ws.cross(fwd_ws)
    right_xy = Vector((right_ws.x, right_ws.y))
    if right_xy.length < 1e-8:
        x_ws = q_ref @ Vector((1.0, 0.0, 0.0))
        right_xy = Vector((x_ws.x, x_ws.y))
        if right_xy.length < 1e-8:
            right_xy = Vector((1.0, 0.0))

    ang_rel = math.degrees(math.atan2(rel_xy.y,  rel_xy.x))
    ang_fwd = math.degrees(math.atan2(fwd_xy.y, fwd_xy.x))
    rel_deg = (ang_rel - ang_fwd + 180.0) % 360.0 - 180.0
    rel_deg = -rel_deg

    if turn_direction:
        rel_deg += turn_to_degree[turn_direction]
        if rel_deg > 180:   rel_deg -= 360
        if rel_deg < -180:  rel_deg += 360

    if -22.5 <= rel_deg < 22.5:
        return "front"
    elif 22.5 <= rel_deg < 67.5:
        return "front-right"
    elif 67.5 <= rel_deg < 112.5:
        return "right"
    elif 112.5 <= rel_deg < 157.5:
        return "back-right"
    elif rel_deg >= 157.5 or rel_deg < -157.5:
        return "back"
    elif -157.5 <= rel_deg < -112.5:
        return "back-left"
    elif -112.5 <= rel_deg < -67.5:
        return "left"
    elif -67.5 <= rel_deg < -22.5:
        return "front-left"
    return "undefined"


def distance_between_objects(object_1: str, object_2: str) -> float:
    """
    Return the Euclidean distance between two objects (including cameras)
    in the currently loaded Blender scene.
    """
    
    object_1_bpy = bpy.data.objects.get(object_1)
    object_2_bpy = bpy.data.objects.get(object_2)
    
    if object_1_bpy is None:
        raise KeyError(f"Object not found: {object_1_bpy}")
    if object_2_bpy is None:
        raise KeyError(f"Object not found: {object_2_bpy}")
    
    if "camera" in object_1:
        object_1_loc = object_1_bpy.location
    else:
        object_1_loc = object_1_bpy.matrix_world.translation

    if "camera" in object_2:
        object_2_loc = object_2_bpy.location
    else:
        object_2_loc = object_2_bpy.matrix_world.translation
    return (object_1_loc - object_2_loc).length