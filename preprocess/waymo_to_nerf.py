from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from waymo_loader import WaymoLoader
import camtools as ct
import numpy as np
import json
import shutil


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    waymo_parent_dir = data_dir / "waymo"
    sequence_id = "1137"
    waymo_parent_dir = waymo_parent_dir / f"waymo_seq{sequence_id}"

    # Load waymo train.
    train_w = WaymoLoader(
        waymo_parent_dir=waymo_parent_dir,
        cameras=["FRONT"],
        lidar=["TOP"],
        train_split_fraction=0.9,
        split="train",
    )
    train_Ks, train_Ts = train_w.load_cameras()
    train_im_paths = train_w.load_image_paths()
    train_lidar2worlds = train_w.load_lidars()
    train_lidar_paths = train_w.load_lidar_paths()
    assert len(train_Ks) == len(train_Ts) == len(train_im_paths)
    # print(f"Num train images: {train_im_paths}")

    # Load waymo test.
    test_w = WaymoLoader(
        waymo_parent_dir=waymo_parent_dir,
        cameras=["FRONT"],
        lidar=["TOP"],
        train_split_fraction=0.9,
        split="test",
    )
    test_Ks, test_Ts = test_w.load_cameras()
    test_im_paths = test_w.load_image_paths()
    test_lidar2worlds = test_w.load_lidars()
    test_lidar_paths = test_w.load_lidar_paths()
    assert len(test_Ks) == len(test_Ts) == len(test_im_paths)
    print(f"Num test images: {test_im_paths}")

    # Get image dimensions.
    # Assume all images have the same dimensions.
    im_h, im_w, _ = ct.io.imread(train_im_paths[0]).shape

    lidar_h, lidar_w, _ = np.load(train_lidar_paths[0]).shape

    # Write train to json.
    json_dict = {
        "w":
            im_w,
        "h":
            im_h,
        "w_lidar":
            lidar_w,
        "h_lidar":
            lidar_h,
        "fl_x":
            float(train_Ks[0][0, 0]),
        "fl_y":
            float(train_Ks[0][1, 1]),
        "cx":
            float(train_Ks[0][0, 2]),
        "cy":
            float(train_Ks[0][1, 2]),
        "aabb_scale":
            2,
        "beam_inclinations":
            train_w.beam_inclinations.tolist(),
        "frames": [{
            "file_path":
                str(train_im_path.relative_to(waymo_parent_dir)),
            "transform_matrix":
                ct.convert.T_to_pose(train_T).tolist(),
            "lidar_file_path":
                str(train_lidar_path.relative_to(waymo_parent_dir)),
            "lidar2world":
                train_lidar2world.tolist(),
        } for (train_im_path, train_T, train_lidar_path,
               train_lidar2world) in zip(train_im_paths, train_Ts,
                                         train_lidar_paths, train_lidar2worlds)]
    }
    json_path = waymo_parent_dir / f'transforms_train.json'
    with open(json_path, "w") as f:
        json.dump(json_dict, f, indent=2)
        print(f"Saved {json_path}.")

    # Write test to json.
    json_dict = {
        "w":
            im_w,
        "h":
            im_h,
        "w_lidar":
            lidar_w,
        "h_lidar":
            lidar_h,
        "fl_x":
            float(test_Ks[0][0, 0]),
        "fl_y":
            float(test_Ks[0][1, 1]),
        "cx":
            float(test_Ks[0][0, 2]),
        "cy":
            float(test_Ks[0][1, 2]),
        "aabb_scale":
            2,
        "beam_inclinations":
            train_w.beam_inclinations.tolist(),
        "frames": [{
            "file_path":
                str(test_im_path.relative_to(waymo_parent_dir)),
            "transform_matrix":
                ct.convert.T_to_pose(test_T).tolist(),
            "lidar_file_path":
                str(test_lidar_path.relative_to(waymo_parent_dir)),
            "lidar2world":
                test_lidar2world.tolist(),
        } for (test_im_path, test_T, test_lidar_path, test_lidar2world) in zip(
            test_im_paths, test_Ts, test_lidar_paths, test_lidar2worlds)]
    }
    json_path = waymo_parent_dir / f'transforms_test.json'
    with open(json_path, "w") as f:
        json.dump(json_dict, f, indent=2)
        print(f"Saved {json_path}.")

    # Copy test to eval (same json)
    shutil.copy(waymo_parent_dir / f'transforms_test.json',
                waymo_parent_dir / f'transforms_val.json')


if __name__ == "__main__":
    main()
