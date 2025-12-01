import pandas as pd
import itertools
from pathlib import Path
import random
from collections import defaultdict
from copy import deepcopy
from view_reconstruct import reconstruct_views_from_csv
from utils_ground_truth_generation import load_json, distance_between_objects,direction_between_objects, object_view_visible_dict_generation, object_name_mapping, orientation_object_name_filtering
import argparse


def view_view_qa_generation(in_image_dir_path, in_blender_dir_path, out_csv_path, query_template, visible_ratio=0.1):
    in_image_dir_path = Path(in_image_dir_path)
    in_blender_dir_path = Path(in_blender_dir_path)
    scene_complexity = in_image_dir_path.name.split("_")[0]
    data_rows = []
    clockwise_directions_name2idx = {"0": 0, "2": 1, "1": 2, "3": 3}
    clockwise_directions_idx2name = {v: k for k, v in clockwise_directions_name2idx.items()}
    movement_choices = ["go opposite", "go left and go forward", "go right and go forward"]
    direction_choices = ["opposite", "left", "right"]
    for object_folder in in_image_dir_path.rglob("*spawn_asset*"):
        room_name = object_folder.parent.name
        csv_path = object_folder / "camera_poses.csv"
        input_blend_path = (in_blender_dir_path / (scene_complexity + "_indoors") / room_name / "scene.blend").absolute()
        for level_folder in object_folder.rglob("level_*"):
            if not level_folder.is_dir():
                continue
            else:
                print(level_folder)
                target_level = level_folder.name
                visibility_file_path = level_folder / "object_visibility.csv"
                visibility_df = pd.read_csv(visibility_file_path, index_col="camera")
                visibility_df = visibility_df[["object", "visible_ratio"]]
                camera_names = reconstruct_views_from_csv(
                    csv_path=csv_path,
                    input_blend_path=input_blend_path,
                    target_level=target_level,
                    )
                camera_visibility_dict = {
                    cam: dict(zip(g["object"], g["visible_ratio"]))
                    for cam, g in visibility_df.groupby(level=0)
                    if cam in camera_names
                    }
                camera_names = list(camera_visibility_dict.keys())
                object_list = list()
                for _, object_dict in camera_visibility_dict.items():
                    object_list.extend([obj for obj, ratio in object_dict.items() if ratio >= visible_ratio])
                    object_list = list(set(object_list))
                image_index_list = sorted([clockwise_directions_name2idx[image_path.stem.split("_")[-1]] for image_path in level_folder.glob("*.png")])
                if len(image_index_list) >= 3:
                    all_combinations = list(itertools.combinations(image_index_list, 2))
                    for combination in all_combinations:
                        image_idx = combination[0]
                        
                        # generate view-view relative distance QA
                        if (combination[1] - image_idx) % 2 == 0:
                            query = query_template["view-view"]["relative_distance"][1].replace("<frame_num_1>", clockwise_directions_idx2name[image_idx])
                            ground_truth = "frame" + clockwise_directions_idx2name[combination[1]]
                            choices = ",".join(["frame" + clockwise_directions_idx2name[idx] for idx in image_index_list if idx != image_idx])
                            data_rows.append({
                                "folder_path": str(level_folder),
                                "query_type": "view-view|relative_distance|1",
                                "query": query,
                                "ground_truth": ground_truth,
                                "choices": choices,
                                "cross_frame": True,
                                "object_num": len(object_list)
                            })
                        if (combination[1] - image_idx) % 2 == 1 and ((image_idx + 2) % 4 in image_index_list):
                            query = query_template["view-view"]["relative_distance"][0].replace("<frame_num_1>", clockwise_directions_idx2name[image_idx])
                            ground_truth = "frame" + clockwise_directions_idx2name[combination[1]]
                            choices = ",".join(["frame" + clockwise_directions_idx2name[combination[1]], "frame" + clockwise_directions_idx2name[(image_idx + 2) % 4]])
                            data_rows.append({
                                "folder_path": str(level_folder),
                                "query_type": "view-view|relative_distance|0",
                                "query": query,
                                "ground_truth": ground_truth,
                                "choices": choices,
                                "cross_frame": True,
                                "object_num": len(object_list)
                            })  

                        # generate view-view relative direction QA
                        query = query_template["view-view"]["relative_direction"][0].replace("<frame_num_1>", clockwise_directions_idx2name[image_idx]).replace("<frame_num_2>", clockwise_directions_idx2name[combination[1]])
                        choices = ",".join(movement_choices)
                        if (combination[1] - image_idx) % 2 == 0:
                            ground_truth = movement_choices[0]
                            data_rows.append({
                                "folder_path": str(level_folder),
                                "query_type": "view-view|relative_direction|0",
                                "query": query,
                                "ground_truth": ground_truth,
                                "choices": choices,
                                "cross_frame": True,
                                "object_num": len(object_list)
                            })
                        elif combination[1] != 3 or image_idx != 0:
                            ground_truth = movement_choices[1]
                            data_rows.append({
                                "folder_path": str(level_folder),
                                "query_type": "view-view|relative_direction|0",
                                "query": query,
                                "ground_truth": ground_truth,
                                "choices": choices,
                                "cross_frame": True,
                                "object_num": len(object_list)
                            })
                        else:
                            ground_truth = movement_choices[2]
                            data_rows.append({
                                "folder_path": str(level_folder),
                                "query_type": "view-view|relative_direction|0",
                                "query": query,
                                "ground_truth": ground_truth,
                                "choices": choices,
                                "cross_frame": True,
                                "object_num": len(object_list)
                            })
                        if (combination[1] - image_idx) % 2 == 0:
                            query = query_template["view-view"]["relative_direction"][1].replace("<frame_num_1>", clockwise_directions_idx2name[image_idx]).replace("<direction>", direction_choices[0])
                            ground_truth = "frame" + clockwise_directions_idx2name[combination[1]]
                            choices = ",".join(["frame" + clockwise_directions_idx2name[idx] for idx in image_index_list if idx != image_idx])
                            data_rows.append({
                                "folder_path": str(level_folder),
                                "query_type": "view-view|relative_direction|1",
                                "query": query,
                                "ground_truth": ground_truth,
                                "choices": choices,
                                "cross_frame": True,
                                "object_num": len(object_list)
                            })
                        elif combination[1] != 3 or image_idx != 0:
                            query = query_template["view-view"]["relative_direction"][1].replace("<frame_num_1>", clockwise_directions_idx2name[image_idx]).replace("<direction>", direction_choices[1])
                            ground_truth = "frame" +  clockwise_directions_idx2name[combination[1]]
                            choices = ",".join(["frame" + clockwise_directions_idx2name[idx] for idx in image_index_list if idx != image_idx])
                            data_rows.append({
                                "folder_path": str(level_folder),
                                "query_type": "view-view|relative_direction|1",
                                "query": query,
                                "ground_truth": ground_truth,
                                "choices": choices,
                                "cross_frame": True,
                                "object_num": len(object_list)
                            })
                        else:
                            query = query_template["view-view"]["relative_direction"][1].replace("<frame_num_1>", clockwise_directions_idx2name[image_idx]).replace("<direction>", direction_choices[2])
                            ground_truth = "frame" + clockwise_directions_idx2name[combination[1]]
                            choices = ",".join(["frame" + clockwise_directions_idx2name[idx] for idx in image_index_list if idx != image_idx])
                            data_rows.append({
                                "folder_path": str(level_folder),
                                "query_type": "view-view|relative_direction|1",
                                "query": query,
                                "ground_truth": ground_truth,
                                "choices": choices,
                                "cross_frame": True,
                                "object_num": len(object_list)
                            })

    df = pd.DataFrame(data_rows)
    df.to_csv(out_csv_path, index=False)



def view_object_qa_generation(in_image_dir_path, in_blender_dir_path, out_csv_path, query_template, object_rename_dir_path, visible_ratio=0.1, min_object_list_len=3, sample_num=2):
    in_blender_dir_path = Path(in_blender_dir_path)
    in_image_dir_path = Path(in_image_dir_path)
    object_rename_dir_path = Path(object_rename_dir_path)
    data_rows = []
    for object_folder in in_image_dir_path.rglob("*spawn_asset*"):
        room_name_path = object_folder.parent
        room_name = object_folder.parent.name
        room_type = object_folder.parent.parent.name
        scene_complexity = in_image_dir_path.name.split("_")[0]
        object_rename_file_path = object_rename_dir_path / (scene_complexity + "_indoors") / room_type / (room_type + ".json")
        object_rename_dict = load_json(object_rename_file_path)
        object_orientation_file_path = object_rename_dir_path / (scene_complexity + "_indoors") / room_type / ("orientation.json")
        object_orientation_dict = load_json(object_orientation_file_path)
        orientation_object_name_list = [object_name_path.split("/")[-1] for object_name_path in object_orientation_dict.keys()]
        csv_path = object_folder / "camera_poses.csv"
        input_blend_path = (in_blender_dir_path / (scene_complexity + "_indoors") / room_name / "scene.blend").absolute()
        for target_level_folder in object_folder.rglob("level_*"):
            print(target_level_folder)
            target_level = target_level_folder.name
            visibility_file_path = target_level_folder / "object_visibility.csv"
            visibility_df = pd.read_csv(visibility_file_path, index_col="camera")
            visibility_df = visibility_df[["object", "visible_ratio"]]
            camera_names = reconstruct_views_from_csv(
                csv_path=csv_path,
                input_blend_path=input_blend_path,
                target_level=target_level,
                )
            camera_visibility_dict = {
                cam: dict(zip(g["object"], g["visible_ratio"]))
                for cam, g in visibility_df.groupby(level=0)
                if cam in camera_names
                }
            camera_names = list(camera_visibility_dict.keys())
            object_list = list()
            for _, object_dict in camera_visibility_dict.items():
                object_list.extend([obj for obj, ratio in object_dict.items() if ratio >= visible_ratio])
                object_list = list(set(object_list))
            orientation_object_list = orientation_object_name_filtering(object_list, orientation_object_name_list)

            # relative distance qa generation
            if len(object_list) >= min_object_list_len:
                camera_object_distance_ascend_dict = {}
                for camera_name in camera_names:
                    camera_object_distance_ascend_dict[camera_name] = sorted(object_list, key=lambda o: distance_between_objects(camera_name, o))
                object_object_distance_ascend_dict = {}
                for object_name in object_list:
                    object_object_distance_ascend_dict[object_name] = sorted(object_list, key=lambda o: distance_between_objects(object_name, o))
                    object_object_distance_ascend_dict[object_name].remove(object_name)
            
            if len(camera_names) >= sample_num:
                sampled_camera_names = random.sample(camera_names, sample_num)
            else:
                sampled_camera_names = camera_names
            for camera_name in sampled_camera_names:
                if len(object_list) >= min_object_list_len:
                    query = query_template["view-object"]["relative_distance"]["non_perspective_changing"][0].replace("<frame_num_1>", camera_name.split("_")[-1])
                    object_distance_ascend_list = camera_object_distance_ascend_dict[camera_name]
                    ground_truth = object_name_mapping(object_distance_ascend_list[0], object_rename_dict, room_name_path)
                    shuffle_object_list = deepcopy(object_list)
                    if len(object_list) < 4:
                        choice_list = [object_name_mapping(object_name, object_rename_dict, room_name_path) for object_name in shuffle_object_list]
                    else:
                        choice_list = random.sample([object_name_mapping(object_name, object_rename_dict, room_name_path) for object_name in object_list], 3)
                        choice_list.append(ground_truth)
                    random.shuffle(choice_list)
                    choice_list = list(set(choice_list))
                    choices = ",".join(choice_list)
                    data_rows.append({
                            "folder_path": str(target_level_folder),
                            "query_type": f"view-object|relative_distance|non_perspective_changing|0",
                            "query": query,
                            "ground_truth": ground_truth,
                            "choices": choices,
                            "cross_frame": ground_truth not in camera_visibility_dict[camera_name],
                            "perspective_changing": False,
                            "object_num": len(object_list)
                        })
                    
            if len(camera_names) >= sample_num:
                sampled_camera_names = random.sample(camera_names, sample_num)
            else:
                sampled_camera_names = camera_names
            for camera_name in sampled_camera_names:
                if len(object_list) >= min_object_list_len:
                    query = query_template["view-object"]["relative_distance"]["non_perspective_changing"][1].replace("<frame_num_1>", camera_name.split("_")[-1])
                    object_distance_ascend_list = camera_object_distance_ascend_dict[camera_name]
                    ground_truth = object_name_mapping(object_distance_ascend_list[-1], object_rename_dict, room_name_path)
                    shuffle_object_list = deepcopy(object_list)
                    if len(object_list) < 4:
                        choice_list = [object_name_mapping(object_name, object_rename_dict, room_name_path) for object_name in shuffle_object_list]
                    else:
                        choice_list = random.sample([object_name_mapping(object_name, object_rename_dict, room_name_path) for object_name in object_list], 3)
                        choice_list.append(ground_truth)
                    choice_list = list(set(choice_list))
                    random.shuffle(choice_list)
                    choices = ",".join(choice_list)
                    data_rows.append({
                            "folder_path": str(target_level_folder),
                            "query_type": "view-object|relative_distance|non_perspective_changing|1",
                            "query": query,
                            "ground_truth": ground_truth,
                            "choices": choices,
                            "cross_frame": ground_truth not in camera_visibility_dict[camera_name],
                            "perspective_changing": False,
                            "object_num": len(object_list)
                        })
                    
            if len(object_list) >= sample_num:
                sampled_object_list = random.sample(object_list, sample_num)  
            else:
                sampled_object_list = object_list      
            for object_name in sampled_object_list:
                if len(object_list) >= min_object_list_len:
                    query = query_template["view-object"]["relative_distance"]["perspective_changing"][0].replace("<object_1>", object_name_mapping(object_name, object_rename_dict, room_name_path))
                    object_distance_ascend_list = object_object_distance_ascend_dict[object_name]
                    ground_truth = object_name_mapping(object_distance_ascend_list[0], object_rename_dict, room_name_path)
                    shuffle_object_list = deepcopy(object_list)
                    if len(object_list) < 4:
                        choice_list = [object_name_mapping(object_name, object_rename_dict, room_name_path) for object_name in shuffle_object_list]
                    else:
                        choice_list = random.sample([object_name_mapping(choice_object_name, object_rename_dict, room_name_path) for choice_object_name in object_list if choice_object_name != object_name], 3)
                        choice_list.append(ground_truth)
                    choice_list = list(set(choice_list))
                    random.shuffle(choice_list)
                    choices = ",".join(choice_list)
                    cross_frame = True
                    for object_visbility_dict in camera_visibility_dict.values():
                        cross_frame = not(object_name in object_visbility_dict and ground_truth in object_visbility_dict)
                        if not cross_frame:
                            break
                    data_rows.append({
                            "folder_path": str(target_level_folder),
                            "query_type": "view-object|relative_distance|perspective_changing|0",
                            "query": query,
                            "ground_truth": ground_truth,
                            "choices": choices,
                            "cross_frame": cross_frame,
                            "perspective_changing": True,
                            "object_num": len(object_list)
                        })
                        
                if len(object_list) >= sample_num:
                    sampled_object_list = random.sample(object_list, sample_num)  
                else:
                    sampled_object_list = object_list         
                for object_name in sampled_object_list:
                    if len(object_list) >= min_object_list_len:
                        query = query_template["view-object"]["relative_distance"]["perspective_changing"][1].replace("<object_1>", object_name_mapping(object_name, object_rename_dict, room_name_path))
                        object_distance_ascend_list = object_object_distance_ascend_dict[object_name]
                        ground_truth = object_name_mapping(object_distance_ascend_list[-1], object_rename_dict, room_name_path)
                        shuffle_object_list = deepcopy(object_list)
                        if len(object_list) < 4:
                            choice_list = [object_name_mapping(object_name, object_rename_dict, room_name_path) for object_name in shuffle_object_list]
                        else:
                            choice_list = random.sample([object_name_mapping(choice_object_name, object_rename_dict, room_name_path) for choice_object_name in object_list if choice_object_name != object_name], 3)
                            choice_list.append(ground_truth)
                        choice_list = list(set(choice_list))
                        random.shuffle(choice_list)
                        choices = ",".join(choice_list)
                        cross_frame = True
                        for object_visbility_dict in camera_visibility_dict.values():
                            cross_frame = not(object_name in object_visbility_dict and ground_truth in object_visbility_dict)
                            if not cross_frame:
                                break
                        data_rows.append({
                                "folder_path": str(target_level_folder),
                                "query_type": "view-object|relative_distance|perspective_changing|1",
                                "query": query,
                                "ground_truth": ground_truth,
                                "choices": choices,
                                "cross_frame": cross_frame,
                                "perspective_changing": True,
                                "object_num": len(object_list)
                            })
                    
            # relative direction qa generation
            camera_direction_object_dict = dict()
            direction_choices = ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"]
            for camera_name in camera_names:
                if len(object_list) >= min_object_list_len:
                    direction_object_dict = defaultdict(list)
                    for object_name in object_list:
                        camera_object_direction = direction_between_objects(camera_name, object_name)
                        if camera_object_direction in direction_choices:
                            direction_object_dict[camera_object_direction].append(object_name)
                    camera_direction_object_dict[camera_name] = dict(direction_object_dict)
                    
            if len(camera_names) >= sample_num:
                sampled_camera_names = random.sample(camera_names, sample_num)
            else:
                sampled_camera_names = camera_names
            for camera_name in sampled_camera_names:
                if len(object_list) >= min_object_list_len:
                    direction = random.choice(list(camera_direction_object_dict[camera_name].keys()))
                    object_name = random.choice(camera_direction_object_dict[camera_name][direction])
                    query = query_template["view-object"]["relative_direction"]["non_perspective_changing"][0].replace("<frame_num_1>", camera_name.split("_")[-1]).replace("<object_1>", object_name_mapping(object_name, object_rename_dict, room_name_path))
                    ground_truth = direction
                    choice_list = [direction for direction in direction_choices if not(set(direction.split("-")) & set(ground_truth.split("-")))]
                    choice_list = random.sample(choice_list, 3)
                    choice_list.append(ground_truth)
                    random.shuffle(choice_list)
                    choices = ",".join(choice_list)
                    data_rows.append({
                            "folder_path": str(target_level_folder),
                            "query_type": "view-object|relative_direction|non_perspective_changing|0",
                            "query": query,
                            "ground_truth": ground_truth,
                            "choices": choices,
                            "cross_frame": object_name not in camera_visibility_dict[camera_name],
                            "perspective_changing": False,
                            "object_num": len(object_list)
                        })
                        
            if len(camera_names) >= sample_num:
                sampled_camera_names = random.sample(camera_names, sample_num)
            else:
                sampled_camera_names = camera_names
            for camera_name in sampled_camera_names:
                if len(object_list) >= min_object_list_len:
                    exist_object_direction_choice = list(camera_direction_object_dict[camera_name].keys())
                    direction = random.choice(exist_object_direction_choice)
                    direction_choice_list = [direction_choice for direction_choice in exist_object_direction_choice if not(set(direction_choice.split("-")) & set(direction.split("_"))) and direction_choice is not direction]
                    ground_truth_object = random.choice(camera_direction_object_dict[camera_name][direction])
                    direction_object_list = list(set([obj for direction in direction_choice_list for obj in direction_object_dict[direction] if obj != ground_truth_object]))
                    if len(direction_object_list) >= 3:
                        query = query_template["view-object"]["relative_direction"]["non_perspective_changing"][1].replace("<frame_num_1>", camera_name.split("_")[-1]).replace("<direction>", direction)
                        ground_truth = object_name_mapping(ground_truth_object, object_rename_dict, room_name_path)
                        choice_list = random.sample([object_name for object_name in direction_object_list], min(3, len(direction_object_list)))
                        choice_list = [object_name_mapping(object_name, object_rename_dict, room_name_path) for object_name in choice_list]
                        choice_list.append(ground_truth)
                        random.shuffle(choice_list)
                        choices = ",".join(choice_list)
                        data_rows.append({
                            "folder_path": str(target_level_folder),
                            "query_type": "view-object|relative_direction|non_perspective_changing|1",
                            "query": query,
                            "ground_truth": ground_truth,
                            "choices": choices,
                            "cross_frame": ground_truth not in camera_visibility_dict[camera_name],
                            "perspective_changing": False,
                            "object_num": len(object_list)
                        })

            turn_direction_list = ["left", "right", "back"]
            if len(object_list) >= min_object_list_len:
                possible_orientation_object_list = random.sample(orientation_object_list, min(2, len(orientation_object_list)))
                for object_name_1 in possible_orientation_object_list:
                    possible_query_object_list = random.sample([object_name for object_name in object_list if object_name != object_name_1], min(2, len(object_list)))
                    for object_name_2 in possible_query_object_list:
                        query = query_template["view-object"]["relative_direction"]["perspective_changing"][0].replace("<object_1>", object_name_mapping(object_name_1, object_rename_dict, room_name_path)).replace("<object_2>", object_name_mapping(object_name_2, object_rename_dict, room_name_path))
                        ground_truth = direction_between_objects(object_name_1, object_name_2)
                        if ground_truth in direction_choices:
                            choice_list = [direction for direction in direction_choices if not(set(direction.split("-")) & set(ground_truth.split("-")))]
                            choice_list = random.sample(choice_list, 3)
                            choice_list.append(ground_truth)
                            random.shuffle(choice_list)
                            choices = ",".join(choice_list)
                            cross_frame = True
                            for visibility_dict in camera_visibility_dict.values():
                                visibility_list = [object_name for object_name, visibility in visibility_dict.items() if visibility>=visible_ratio]
                                if object_name_1 in visibility_list and object_name_2 in visibility_list:
                                    cross_frame = False
                                    break
                            data_rows.append({
                                        "folder_path": str(target_level_folder),
                                        "query_type": "view-object|relative_direction|perspective_changing|0",
                                        "query": query,
                                        "ground_truth": ground_truth,
                                        "choices": choices,
                                        "cross_frame": cross_frame,
                                        "perspective_changing": True,
                                        "object_num": len(object_list)
                                    })  
                            
                possible_orientation_object_list = random.sample(orientation_object_list, min(2, len(orientation_object_list)))
                for object_name_1 in possible_orientation_object_list:
                    possible_query_object_list = random.sample([object_name for object_name in object_list if object_name != object_name_1], min(2, len(object_list)))
                    for object_name_2 in possible_query_object_list:
                        turn_direction_choice = random.choice(turn_direction_list)
                        query = query_template["view-object"]["relative_direction"]["perspective_changing"][2].replace("<object_1>", object_name_mapping(object_name_1, object_rename_dict, room_name_path)).replace("<object_2>", object_name_mapping(object_name_2, object_rename_dict, room_name_path)).replace("<direction>", turn_direction_choice)
                        ground_truth = direction_between_objects(object_name_1, object_name_2, turn_direction=turn_direction_choice)
                        if ground_truth in direction_choices:
                            choice_list = [direction for direction in direction_choices if not(set(direction.split("-")) & set(ground_truth.split("-")))]
                            choice_list = random.sample(choice_list, 3)
                            choice_list.append(ground_truth)
                            random.shuffle(choice_list)
                            choices = ",".join(choice_list)
                            data_rows.append({
                                        "folder_path": str(target_level_folder),
                                        "query_type": "view-object|relative_direction|perspective_changing|2",
                                        "query": query,
                                        "ground_truth": ground_truth,
                                        "choices": choices,
                                        "cross_frame": cross_frame,
                                        "perspective_changing": True,
                                        "object_num": len(object_list)
                                    })  
                
                object_direction_object_dict = defaultdict(lambda: defaultdict(list))
                for object_name_1 in orientation_object_list:
                    for object_name_2 in object_list:
                        direction = direction_between_objects(object_name_1, object_name_2)
                        if direction in direction_choices:
                            object_direction_object_dict[object_name_1][direction].append(object_name_2)
                query_number = 0
                for object_name_1, direction_object_dict in object_direction_object_dict.items():
                    if len(direction_object_dict) >= 2 & query_number < 3:
                        direction = random.choice(list(direction_object_dict.keys()))
                        object_name_2 = random.choice(direction_object_dict[direction])
                        other_object = []
                        for other_direction, other_object_list in direction_object_dict.items():
                            if not(set(direction.split("-")) & set(other_direction.split("-"))):
                                other_object.extend(other_object_list)
                        other_object = list(set(other_object))
                        if other_object:
                            choice_list = random.sample(other_object, min(3, len(other_object)))
                            choice_list.append(object_name_1)
                            random.shuffle(choice_list)
                            choice_list = [object_name_mapping(choice, object_rename_dict, room_name_path) for choice in choice_list]
                            choices = ",".join(choice_list)
                            query = query_template["view-object"]["relative_direction"]["perspective_changing"][1].replace("<object_1>",object_name_mapping(object_name_1, object_rename_dict, room_name_path)).replace("<direction>", direction)
                            ground_truth = object_name_mapping(object_name_2, object_rename_dict, room_name_path)
                            query_number += 1
                            cross_frame = True
                            for object_visibility_dict in camera_visibility_dict.values():
                                if object_name_1 in object_visibility_dict and object_name_2 in object_visibility_dict:
                                    cross_frame = False
                            data_rows.append({
                                "folder_path": str(target_level_folder),
                                "query_type": "view-object|relative_direction|perspective_changing|1",
                                "query": query,
                                "ground_truth": ground_truth,
                                "choices": choices,
                                "cross_frame": cross_frame,
                                "perspective_changing": True,
                                "object_num": len(object_list)
                            }) 

                turn_direction_choice = random.choice(turn_direction_list)
                object_direction_object_dict = defaultdict(lambda: defaultdict(list))
                for object_name_1 in orientation_object_list:
                    for object_name_2 in object_list:
                        direction = direction_between_objects(object_name_1, object_name_2, turn_direction=turn_direction_choice)
                        if direction in direction_choices:
                            object_direction_object_dict[object_name_1][direction].append(object_name_2)
                query_number = 0
                for object_name_1, direction_object_dict in object_direction_object_dict.items():
                    if len(direction_object_dict) >= 2 & query_number < 3:
                        direction = random.choice(list(direction_object_dict.keys()))
                        object_name_2 = random.choice(direction_object_dict[direction])
                        other_object = []
                        for other_direction, other_object_list in direction_object_dict.items():
                            if not(set(direction.split("-")) & set(other_direction.split("-"))):
                                other_object.extend(other_object_list)
                        other_object = list(set(other_object))
                        if other_object:
                            choice_list = random.sample(other_object, min(3, len(other_object)))
                            choice_list.append(object_name_1)
                            random.shuffle(choice_list)
                            choice_list = [object_name_mapping(choice, object_rename_dict, room_name_path) for choice in choice_list]
                            choices = ",".join(choice_list)
                            query = query_template["view-object"]["relative_direction"]["perspective_changing"][3].replace("<object_1>", object_name_mapping(object_name_1, object_rename_dict, room_name_path)).replace("<direction_2>", direction).replace("<direction_1>", turn_direction_choice)
                            query_number += 1
                            cross_frame = True
                            for object_visibility_dict in camera_visibility_dict.values():
                                if object_name_1 in object_visibility_dict and object_name_2 in object_visibility_dict:
                                    cross_frame = False
                            data_rows.append({
                                "folder_path": str(target_level_folder),
                                "query_type": "view-object|relative_direction|perspective_changing|3",
                                "query": query,
                                "ground_truth": ground_truth,
                                "choices": choices,
                                "cross_frame": cross_frame,
                                "perspective_changing": True,
                                "object_num": len(object_list)
                            }) 
    
    df = pd.DataFrame(data_rows)
    df.to_csv(out_csv_path, index=False)
            

def object_object_qa_generation(in_image_dir_path, in_blender_dir_path, out_csv_path, query_template, object_rename_dir_path, visible_ratio=0.1, min_object_list_len=3, sample_num=2):
    in_blender_dir_path = Path(in_blender_dir_path)
    in_image_dir_path = Path(in_image_dir_path)
    object_rename_dir_path = Path(object_rename_dir_path)
    data_rows = []
    for object_folder in in_image_dir_path.rglob("*spawn_asset*"):
        main_object_name = object_folder.name
        room_name = object_folder.parent.name
        room_name_path = object_folder.parent
        room_type = object_folder.parent.parent.name
        scene_complexity = in_image_dir_path.name.split("_")[0]
        object_rename_file_path = object_rename_dir_path / (scene_complexity + "_indoors") / room_type / (room_type + ".json")
        object_rename_dict = load_json(object_rename_file_path)
        object_orientation_file_path = object_rename_dir_path / (scene_complexity + "_indoors") / room_type / ("orientation.json")
        object_orientation_dict = load_json(object_orientation_file_path)
        orientation_object_name_list = [object_name_path.split("/")[-1] for object_name_path in object_orientation_dict.keys()]
        csv_path = object_folder / "camera_poses.csv"
        scene_complexity = in_image_dir_path.name.split("_")[0]
        input_blend_path = (in_blender_dir_path / (scene_complexity + "_indoors") / room_name / "scene.blend").absolute()
        for target_level_folder in object_folder.rglob("level_*"):
            print(target_level_folder)
            target_level = target_level_folder.name
            visibility_file_path = target_level_folder / "object_visibility.csv"
            visibility_df = pd.read_csv(visibility_file_path, index_col="camera")
            visibility_df = visibility_df[["object", "visible_ratio"]]
            camera_names = reconstruct_views_from_csv(
                csv_path=csv_path,
                input_blend_path=input_blend_path,
                target_level=target_level,
                )
            camera_visibility_dict = {
                cam: dict(zip(g["object"], g["visible_ratio"]))
                for cam, g in visibility_df.groupby(level=0)
                if cam in camera_names
                }
            camera_names = list(camera_visibility_dict.keys())
            object_list = list()
            for _, object_dict in camera_visibility_dict.items():
                object_list.extend([obj for obj, ratio in object_dict.items() if ratio >= visible_ratio])
                object_list = list(set(object_list))
            orientation_object_list = orientation_object_name_filtering(object_list, orientation_object_name_list)
            
            # relative distance qa generation
            if len(object_list) >= min_object_list_len:
                query_objects = random.sample(object_list, sample_num)
                query_object_object_distance_dict = {}
                for object_name in query_objects:
                    query_object_object_distance_dict[object_name] = sorted([other_object_name for other_object_name in object_list if other_object_name is not object_name], key=lambda o: distance_between_objects(object_name, o))
                    query = query_template["object-object"]["relative_distance"]["non_perspective_changing"][0].replace("<object_1>", object_name_mapping(object_name, object_rename_dict, room_name_path))
                    ground_truth_name = query_object_object_distance_dict[object_name][0]
                    ground_truth = object_name_mapping(ground_truth_name, object_rename_dict, room_name_path)
                    if len(object_list) < 4:
                        choice_list = [object_name_mapping(object_name, object_rename_dict, room_name_path) for object_name in object_list]
                    else:
                        choice_list = random.sample([object_name_mapping(object_name, object_rename_dict, room_name_path) for object_name in object_list if object_name != ground_truth_name], 3)
                        choice_list.append(ground_truth)
                    random.shuffle(choice_list)
                    choices = ",".join(choice_list)
                    cross_frame = True
                    for visible_object_dict in camera_visibility_dict.values():
                        if all([choice_object_name in visible_object_dict for choice_object_name in choices.split(",")]):
                            cross_frame = False
                            break
                    data_rows.append({
                            "folder_path": str(target_level_folder),
                            "query_type": "object-object|relative_distance|non_perspective_changing|0",
                            "query": query,
                            "ground_truth": ground_truth,
                            "choices": choices,
                            "cross_frame": cross_frame,
                            "perspective_changing": False,
                            "object_num": len(object_list)
                        }) 
                    
                query_objects = random.sample(object_list, sample_num)
                query_object_object_distance_dict = {}
                for object_name in query_objects:
                    query_object_object_distance_dict[object_name] = sorted([other_object_name for other_object_name in object_list if other_object_name is not object_name], key=lambda o: distance_between_objects(object_name, o))
                    query = query_template["object-object"]["relative_distance"]["non_perspective_changing"][1].replace("<object_1>", object_name_mapping(object_name, object_rename_dict, room_name_path))
                    ground_truth_name = query_object_object_distance_dict[object_name][-1]
                    ground_truth = object_name_mapping(ground_truth_name, object_rename_dict, room_name_path)
                    if len(object_list) < 4:
                        choice_list = [object_name_mapping(object_name, object_rename_dict, room_name_path) for object_name in object_list]
                    else:
                        choice_list = random.sample([object_name_mapping(object_name, object_rename_dict, room_name_path) for object_name in object_list if object_name != ground_truth_name], 3)
                        choice_list.append(ground_truth)
                    random.shuffle(choice_list)
                    choices = ",".join(choice_list)
                    cross_frame = True
                    for visible_object_dict in camera_visibility_dict.values():
                        if all([choice_object_name in visible_object_dict for choice_object_name in choices.split(",")]):
                            cross_frame = False
                            break
                    data_rows.append({
                            "folder_path": str(target_level_folder),
                            "query_type": "object-object|relative_distance|non_perspective_changing|1",
                            "query": query,
                            "ground_truth": ground_truth,
                            "choices": choices,
                            "cross_frame": cross_frame,
                            "perspective_changing": False,
                            "object_num": len(object_list)
                        }) 
        
            # relative direction qa generation
            if main_object_name in object_list:
                direction_choices = ["front", "front-right", "right", "back-right", "back", "back-left", "left", "front-left"]
                camera_direction_object_dict = {}
                for camera_name in camera_names:
                    direction_object_dict = defaultdict(list)
                    for object_name in object_list:
                        direction = direction_between_objects(main_object_name, object_name, camera_name)
                        if direction in direction_choices:
                            direction_object_dict[direction].append(object_name)
                    camera_direction_object_dict[camera_name] = direction_object_dict      

                if len(object_list) >= min_object_list_len:
                    possible_camera_name = [camera_name for camera_name in camera_names if len(camera_direction_object_dict[camera_name])]
                    for camera_name in possible_camera_name:
                        object_direction_dict = camera_direction_object_dict[camera_name]
                        smallest_object_list_len = min(len(v) for v in object_direction_dict.values())
                        directions = [k for k, v in object_direction_dict.items() if len(v) == smallest_object_list_len]
                        direction = random.choice(directions)
                        direction_choice_list = [direction_choice for direction_choice in object_direction_dict.keys() if not(set(direction_choice.split("-")) & set(direction.split("-"))) and direction_choice is not direction]
                        choice_list = []
                        ground_truth_name = random.choice(object_direction_dict[direction])
                        ground_truth = object_name_mapping(ground_truth_name, object_rename_dict, room_name_path)
                        for choice_direction in direction_choice_list:
                            choice_list.extend(object_direction_dict[choice_direction])                    
                        choice_list = list(set(choice_list))
                        choice_list = [object_name for object_name in choice_list if object_name != ground_truth_name]
                        if len(choice_list):
                            choice_list = random.sample(choice_list, min(3, len(choice_list)))
                            choice_list = [object_name_mapping(object_name, object_rename_dict, room_name_path) for object_name in choice_list]
                            choice_list.append(ground_truth)
                            random.shuffle(choice_list)
                            query = query_template["object-object"]["relative_direction"]["non_perspective_changing"][0].replace("<frame_num_1>", camera_name.split("_")[-1]).replace("<object_1>", object_name_mapping(main_object_name, object_rename_dict, room_name_path)).replace("<direction>", direction)
                            choices = ",".join(choice_list)
                            cross_frame = True
                            for object_visibility_dict in camera_visibility_dict.values():
                                if main_object_name in object_visibility_dict and ground_truth in object_visibility_dict:
                                    cross_frame = False
                            data_rows.append({
                                    "folder_path": str(target_level_folder),
                                    "query_type": "object-object|relative_direction|non_perspective_changing|0",
                                    "query": query,
                                    "ground_truth": ground_truth,
                                    "choices": choices,
                                    "cross_frame": cross_frame,
                                    "perspective_changing": False,
                                    "object_num": len(object_list)
                                }) 

                        ground_truth = random.choice(list(camera_direction_object_dict[camera_name]))
                        query_object = random.choice(camera_direction_object_dict[camera_name][ground_truth])
                        choice_list = [direction_choice for direction_choice in direction_choices if not(set(direction_choice.split("-")) & set(ground_truth.split("-")))]
                        choice_list = random.sample(choice_list, min(3, len(choice_list)))
                        choice_list.append(ground_truth)
                        random.shuffle(choice_list)
                        choices = ",".join(choice_list)
                        query = query_template["object-object"]["relative_direction"]["non_perspective_changing"][1].replace("<frame_num_1>", camera_name.split("_")[-1]).replace("<object_1>", object_name_mapping(main_object_name, object_rename_dict, room_name_path)).replace("<object_2>", object_name_mapping(query_object, object_rename_dict, room_name_path))
                        cross_frame = True
                        for object_visibility_dict in camera_visibility_dict.values():
                            if main_object_name in object_visibility_dict and query_object in object_visibility_dict:
                                cross_frame = False
                        data_rows.append({
                                "folder_path": str(target_level_folder),
                                "query_type": "object-object|relative_direction|non_perspective_changing|1",
                                "query": query,
                                "ground_truth": ground_truth,
                                "choices": choices,
                                "cross_frame": cross_frame,
                                "perspective_changing": False,
                                "object_num": len(object_list)
                            }) 
                        
                if main_object_name in orientation_object_list:
                    query_object_dict = object_view_visible_dict_generation(main_object_name, object_list)
                    query_object_list = query_object_dict['front']
                    if len(query_object_list) >= 2:
                        object_direction_object_dict = defaultdict(lambda : defaultdict(list))
                        for object_name_1 in query_object_list:
                            for object_name_2 in query_object_list:
                                if object_name_1 != object_name_2:
                                    direction = direction_between_objects(object_name_1, object_name_2, main_object_name)
                                    if direction in direction_choices:
                                        object_direction_object_dict[object_name_1][direction].append(object_name_2)
                        possible_query_object_list = [object_name for object_name in object_direction_object_dict.keys() if len(object_direction_object_dict[object_name])>=2]
                        for object_name_1 in possible_query_object_list:
                            direction = random.choice(list(object_direction_object_dict[object_name_1].keys()))
                            ground_truth_name = random.choice(object_direction_object_dict[object_name_1][direction])
                            ground_truth = object_name_mapping(ground_truth_name, object_rename_dict, room_name_path)
                            choice_list = []
                            for direct, object_name_list in object_direction_object_dict[object_name_1].items():
                                if direct != direction:
                                    choice_list.extend(object_name_list)
                            choice_list = list(set(choice_list))
                            if len(choice_list):
                                choice_list = [object_name_mapping(object_name, object_rename_dict, room_name_path) for object_name in choice_list]
                                choices = ",".join(choice_list)
                                query = query_template["object-object"]["relative_direction"]["perspective_changing"][0].replace("<object_1>", object_name_mapping(main_object_name, object_rename_dict, room_name_path)).replace("<object_2>", object_name_mapping(object_name_1, object_rename_dict, room_name_path)).replace("<direction>", direction)
                                cross_frame = True
                                for object_visibility_dict in camera_visibility_dict.values():
                                    if object_name_1 in object_visibility_dict and ground_truth in object_visibility_dict and main_object_name in object_visibility_dict:
                                        cross_frame = False
                                data_rows.append({
                                        "folder_path": str(target_level_folder),
                                        "query_type": "object-object|relative_direction|perspective_changing|0",
                                        "query": query,
                                        "ground_truth": ground_truth,
                                        "choices": choices,
                                        "cross_frame": cross_frame,
                                        "perspective_changing": True,
                                        "object_num": len(object_list)
                                    }) 
                            
                        possible_query_object_combinations = list(itertools.combinations(query_object_list, 2))
                        possible_query_object_combinations = random.sample(possible_query_object_combinations, min(sample_num, len(possible_query_object_combinations)))
                        for object_name_1, object_name_2 in possible_query_object_combinations:
                            ground_truth = direction_between_objects(object_name_1, object_name_2, main_object_name)
                            choice_list = [direction_choice for direction_choice in direction_choices if not(set(direction_choice.split("-")) & set(ground_truth.split("-")))]
                            choice_list = random.sample(choice_list, min(3, len(choice_list)))
                            choice_list.append(ground_truth)
                            choices = ",".join(choice_list)
                            query = query_template["object-object"]["relative_direction"]["perspective_changing"][1].replace("<object_1>", object_name_mapping(main_object_name, object_rename_dict, room_name_path)).replace("<object_2>", object_name_mapping(object_name_1, object_rename_dict, room_name_path)).replace("<object_3>", object_name_mapping(object_name_2, object_rename_dict, room_name_path))
                            cross_frame = True
                            for object_visibility_dict in camera_visibility_dict.values():
                                if object_name_1 in object_visibility_dict and object_name_2 in object_visibility_dict and main_object_name in object_visibility_dict:
                                    cross_frame = False
                            data_rows.append({
                                    "folder_path": str(target_level_folder),
                                    "query_type": "object-object|relative_direction|perspective_changing|1",
                                    "query": query,
                                    "ground_truth": ground_truth,
                                    "choices": choices,
                                    "cross_frame": cross_frame,
                                    "perspective_changing": True,
                                    "object_num": len(object_list)
                                })

                    turn_direction_choices = ["right", "left", "back"]
                    turn_direction = random.choice(turn_direction_choices)  
                    query_object_dict = object_view_visible_dict_generation(main_object_name, object_list, turn_direction)
                    query_object_list = query_object_dict['front']
                    if len(query_object_list) >= 2:
                        object_direction_object_dict = defaultdict(lambda : defaultdict(list))
                        for object_name_1 in query_object_list:
                            for object_name_2 in query_object_list:
                                if object_name_1 != object_name_2:
                                    direction = direction_between_objects(object_name_1, object_name_2, main_object_name, turn_direction)
                                    if direction in direction_choices:
                                        object_direction_object_dict[object_name_1][direction].append(object_name_2)
                        possible_query_object_list = [object_name for object_name in object_direction_object_dict.keys() if len(object_direction_object_dict[object_name])>=2]
                        for object_name_1 in possible_query_object_list:
                            direction = random.choice(list(object_direction_object_dict[object_name_1].keys()))
                            ground_truth_name = random.choice(object_direction_object_dict[object_name_1][direction])
                            ground_truth = object_name_mapping(ground_truth_name, object_rename_dict, room_name_path)
                            choice_list = []
                            for direct, object_name_list in object_direction_object_dict[object_name_1].items():
                                if direct != direction:
                                    choice_list.extend(object_name_list)
                            choice_list = list(set(choice_list))
                            if len(choice_list): 
                                choice_list = [object_name_mapping(object_name, object_rename_dict, room_name_path) for object_name in choice_list]
                                choices = ",".join(choice_list)
                                query = query_template["object-object"]["relative_direction"]["perspective_changing"][2].replace("<object_1>", object_name_mapping(main_object_name, object_rename_dict, room_name_path)).replace("<object_2>",object_name_mapping(object_name_1, object_rename_dict, room_name_path)).replace("<direction_1>", turn_direction).replace("<direction_2>", direction)
                                cross_frame = True
                                for object_visibility_dict in camera_visibility_dict.values():
                                    if object_name_1 in object_visibility_dict and ground_truth in object_visibility_dict and main_object_name in object_visibility_dict:
                                        cross_frame = False
                                data_rows.append({
                                        "folder_path": str(target_level_folder),
                                        "query_type": "object-object|relative_direction|perspective_changing|2",
                                        "query": query,
                                        "ground_truth": ground_truth,
                                        "choices": choices,
                                        "cross_frame": cross_frame,
                                        "perspective_changing": True,
                                        "object_num": len(object_list)
                                    }) 
                            
                        possible_query_object_combinations = list(itertools.combinations(query_object_list, 2))
                        possible_query_object_combinations = random.sample(possible_query_object_combinations, min(sample_num, len(possible_query_object_combinations)))
                        for object_name_1, object_name_2 in possible_query_object_combinations:
                            ground_truth = direction_between_objects(object_name_1, object_name_2, main_object_name, turn_direction)
                            choice_list = [direction_choice for direction_choice in direction_choices if not(set(direction_choice.split("-")) & set(ground_truth.split("-")))]
                            choice_list = random.sample(choice_list, min(3, len(choice_list)))
                            choice_list.append(ground_truth)
                            choices = ",".join(choice_list)
                            query = query_template["object-object"]["relative_direction"]["perspective_changing"][3].replace("<object_1>",object_name_mapping(main_object_name, object_rename_dict, room_name_path)).replace("<object_2>", object_name_mapping(object_name_1, object_rename_dict, room_name_path)).replace("<object_3>",object_name_mapping(object_name_2, object_rename_dict, room_name_path)).replace("<direction>", turn_direction)
                            cross_frame = True
                            for object_visibility_dict in camera_visibility_dict.values():
                                if object_name_1 in object_visibility_dict and object_name_2 in object_visibility_dict and main_object_name in object_visibility_dict:
                                    cross_frame = False
                            data_rows.append({
                                    "folder_path": str(target_level_folder),
                                    "query_type": "object-object|relative_direction|perspective_changing|3",
                                    "query": query,
                                    "ground_truth": ground_truth,
                                    "choices": choices,
                                    "cross_frame": cross_frame,
                                    "perspective_changing": True,
                                    "object_num": len(object_list)
                                })

    df = pd.DataFrame(data_rows)
    df.to_csv(out_csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str)
    parser.add_argument("--qa_type", type=str)
    parser.add_argument("--blender_dir_path", type=str, default="outputs")
    parser.add_argument("--rename_dir_path", type=str, default="object_rename")
    parser.add_argument("--query_template_path", type=str, default="query_template.json")
    args = parser.parse_args()
    name_part = args.image_folder.split("_")
    first_part = "/"+ "_".join([name_part[0], name_part[1], name_part[2]])
    second_part = args.qa_type + "_qa.csv"
    qa_file_name = "_".join([first_part, second_part])
    query_template = load_json(args.query_template_path)

    if args.qa_type == "view_view":
        view_view_qa_generation(args.image_folder, args.blender_dir_path, args.image_folder + qa_file_name, query_template)

    if args.qa_type == "view_object":
        view_object_qa_generation(args.image_folder, args.blender_dir_path, args.image_folder + qa_file_name, query_template, args.rename_dir_path)
    
    if args.qa_type == "object_object":
        object_object_qa_generation(args.image_folder, args.blender_dir_path, args.image_folder + qa_file_name, query_template, args.rename_dir_path)


    # template_path = "query_template.json"
    # query_template = load_json(template_path)
    # dense_object_folder = "dense_object_centric_view_frame_outputs_processed"
    # dense_view_folder = "dense_view_centric_view_frame_outputs_processed"
    # rename_dir_path = "object_rename"

    # view_view_qa_generation(dense_object_folder, "outputs", dense_object_folder + "/object_centric_view_view_qa.csv", query_template)
    # view_object_qa_generation(dense_object_folder, "outputs", dense_object_folder + "/object_centric_view_object_qa.csv", query_template, rename_dir_path)
    # object_object_qa_generation(dense_object_folder, "outputs", dense_object_folder + "/object_centric_object_object_qa.csv", query_template, rename_dir_path)

    # view_object_qa_generation(dense_view_folder, "outputs", dense_view_folder + "/view_centric_view_object_qa.csv", query_template, rename_dir_path)
    # object_object_qa_generation(dense_view_folder, "outputs", dense_view_folder + "/view_centric_object_object_qa.csv", query_template, rename_dir_path)