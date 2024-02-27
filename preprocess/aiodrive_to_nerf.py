from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from aiodrive_loader import AIODriveLoader
import camtools as ct
import numpy as np
import json
import shutil


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    aiodrive_parent_dir = data_dir / "aiodrive"
    sequence_id = "64"
    if sequence_id == "64":
        print("Using sqequence 0-63")
        s_frame_id = 0
        e_frame_id = 63  # Inclusive
        val_frame_ids = [13, 26, 39, 52]
    else:
        raise ValueError(f"Invalid sequence id: {sequence_id}")

    frame_ids = list(range(s_frame_id, e_frame_id + 1))
    num_frames = len(frame_ids)

    test_frame_ids = val_frame_ids
    train_frame_ids = [x for x in frame_ids if x not in val_frame_ids]

    # Load aiodrive train.
    aioloader = AIODriveLoader(frame_ids=frame_ids,
                               aiodrive_parent_dir=aiodrive_parent_dir,
                               cameras=["2"])
    Ks, Ts = aioloader.load_cameras()
    im_paths = aioloader.load_image_paths()
    lidar2world = aioloader.load_lidars()
    range_view_paths = aioloader.load_lidar_paths()
    # print(f"Num train images: {train_im_paths}")

    # Get image dimensions.
    # Assume all images have the same dimensions.
    im_h, im_w, _ = ct.io.imread(im_paths[0], alpha_mode="ignore").shape

    lidar_h, lidar_w, _ = np.load(range_view_paths[0]).shape

    # Split by train/test/val.
    all_indices = [i - s_frame_id for i in frame_ids]
    train_indices = [i - s_frame_id for i in train_frame_ids]
    val_indices = [i - s_frame_id for i in val_frame_ids]
    test_indices = [i - s_frame_id for i in test_frame_ids]

    split_to_all_indices = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    for split, indices in split_to_all_indices.items():
        print(f"Split {split} has {len(indices)} frames.")
        im_paths_split = [im_paths[i] for i in indices]
        lidar_paths_split = [range_view_paths[i] for i in indices]
        lidar2world_split = [lidar2world[i] for i in indices]
        Ks_split = [Ks[i] for i in indices]
        Ts_split = [Ts[i] for i in indices]

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
                float(Ks_split[0][0, 0]),
            "fl_y":
                float(Ks_split[0][1, 1]),
            "cx":
                float(Ks_split[0][0, 2]),
            "cy":
                float(Ks_split[0][1, 2]),
            "aabb_scale":
                2,
            "frames": [{
                "file_path":
                    str(path.relative_to(aiodrive_parent_dir)),
                "transform_matrix":
                    ct.convert.T_to_pose(T).tolist(),
                "lidar_file_path":
                    str(lidar_path.relative_to(aiodrive_parent_dir)),
                "lidar2world":
                    lidar2world.tolist(),
            } for (
                path,
                T,
                lidar_path,
                lidar2world,
            ) in zip(
                im_paths_split,
                Ts_split,
                lidar_paths_split,
                lidar2world_split,
            )]
        }
        json_path = aiodrive_parent_dir / f"transforms_{sequence_id}_{split}.json"

        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=2)
            print(f"Saved {json_path}.")


if __name__ == "__main__":
    main()
