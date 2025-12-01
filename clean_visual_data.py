from pathlib import Path
import pandas as pd
import argparse

def remove_recursive(path: Path):
    for child in path.iterdir():
        if child.is_dir():
            remove_recursive(child)
        else:
            child.unlink()  # delete file
    path.rmdir()

def clean_visual_data(dir_path):
    frame_num_dir = {
        "1_frame": 0,
        "2_frame": 0,
        "3_frame": 0,
        "4_frame": 0,
        }
    dir_path = Path(dir_path)
    camera_set = set("camera_" + str(i) for i in range(4))
    for level_dir in dir_path.rglob("level_*"):
        visible_file_path = level_dir / "object_visibility.csv"
        visibility_df = pd.read_csv(visible_file_path)
        camrea_with_object = set(visibility_df["camera"].unique())
        clear_frame_paths = [level_dir / ("frame_" + camera.split("_")[1] + ".png") for camera in camera_set - camrea_with_object]
        for frame_path in clear_frame_paths:
            frame_path.unlink(missing_ok=True)

        if len(list(level_dir.rglob("*.png"))) == 0:
            remove_recursive(level_dir)
        elif len(list(level_dir.rglob("*.png"))) == 1:
            frame_num_dir["1_frame"] += 1
        elif len(list(level_dir.rglob("*.png"))) == 2:
            frame_num_dir["2_frame"] += 1
        elif len(list(level_dir.rglob("*.png"))) == 3:
            frame_num_dir["3_frame"] += 1
        elif len(list(level_dir.rglob("*.png"))) == 4:
            frame_num_dir["4_frame"] += 1
            
    for object_dir in dir_path.rglob("*spawn_asset*"):
        if len(list(object_dir.rglob("*.png"))) == 0:
            remove_recursive(object_dir)
    frame_num_dir["object_num"] = len(list(dir_path.rglob("*spawn_asset*")))
    return frame_num_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_path", type=str, help="Path to the directory containing visual data.")
    args = parser.parse_args()
    result = clean_visual_data(args.dir_path)
    print(result)