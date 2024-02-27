import numpy as np
import argparse

np.set_printoptions(suppress=True)
import os
import json
import tqdm
from lidarnerf.convert import pano_to_lidar
import matplotlib.pyplot as plt


def cal_centerpose_bound_scale(lidar_rangeview_paths,
                               lidar2worlds,
                               intrinsics=None,
                               beam_inclinations=None,
                               bound=1.):
    near = 200
    far = 0
    points_world_list = []
    for i, lidar_rangeview_path in enumerate(lidar_rangeview_paths):
        lidar_rangeview_path = lidar_rangeview_path.replace('train', 'train_offset')
        pano = np.load(lidar_rangeview_path)
        point_cloud = pano_to_lidar(pano=pano[:, :, 2],
                                    lidar_K=intrinsics,
                                    beam_inclinations=beam_inclinations)
        point_cloud = np.concatenate(
            [point_cloud,
             np.ones(point_cloud.shape[0]).reshape(-1, 1)], -1)
        dis = np.sqrt(point_cloud[:, 0]**2 + point_cloud[:, 1]**2 +
                      point_cloud[:, 2]**2)
        near = min(min(dis), near)
        far = max(far, max(dis))
        points_world = (point_cloud @ lidar2worlds[i].T)[:, :3]
        points_world_list.append(points_world)
    print("near, far:", near, far)

    pc_all_w = np.concatenate(points_world_list)[:, :3]


    centerpose = [(np.max(pc_all_w[:, 0]) + np.min(pc_all_w[:, 0])) / 2.,
                  (np.max(pc_all_w[:, 1]) + np.min(pc_all_w[:, 1])) / 2.,
                  (np.max(pc_all_w[:, 2]) + np.min(pc_all_w[:, 2])) / 2.]
    print('centerpose: ', centerpose)
    pc_all_w_centered = pc_all_w - centerpose



    bound_ori = [
        np.max(pc_all_w_centered[:, 0]),
        np.max(pc_all_w_centered[:, 1]),
        np.max(pc_all_w_centered[:, 2])
    ]
    scale = bound / np.max(bound_ori)
    print('scale: ', scale)



def get_path_pose_from_json(root_path, sequence_id=None):
    if sequence_id is not None:
        with open(
                os.path.join(root_path, f'transforms_{sequence_id}_train.json'),
                'r') as f:
            transform = json.load(f)
    else:
        with open(os.path.join(root_path, f'transforms_train.json'), 'r') as f:
            transform = json.load(f)
    frames = transform["frames"]
    if 'beam_inclinations' in transform:
        beam_inclinations = transform['beam_inclinations']
    else:
        beam_inclinations = None
    poses_lidar = []
    paths_lidar = []
    for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
        pose_lidar = np.array(f['lidar2world'], dtype=np.float32)  # [4, 4]
        f_lidar_path = os.path.join(root_path, f['lidar_file_path'])
        # f_lidar_path = f_lidar_path.replace("train", "train_offset")
        poses_lidar.append(pose_lidar)
        paths_lidar.append(f_lidar_path)
    return paths_lidar, poses_lidar, beam_inclinations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="kitti360",
        choices=["kitti360", "nerf_mvl", "waymo", "aiodrive"],
        help="The dataset loader to use.",
    )
    args = parser.parse_args()

    if args.dataset == "kitti360":
        # kitti360
        root_path = 'data_old2/kitti360'
        # root_path = 'data/kitti360'
        sequence_id = 1908
        lidar_rangeview_paths, lidar2worlds, _ = get_path_pose_from_json(
            root_path, sequence_id=sequence_id)
        intrinsics = (2., 26.9)  # fov_up, fov

        cal_centerpose_bound_scale(lidar_rangeview_paths, lidar2worlds,
                                   intrinsics)
    elif args.dataset == "aiodrive":
        # kitti360
        root_path = 'data/aiodrive'
        sequence_id = 64
        lidar_rangeview_paths, lidar2worlds, _ = get_path_pose_from_json(
            root_path, sequence_id=sequence_id)
        intrinsics = (2., 26.9)  # fov_up, fov

        cal_centerpose_bound_scale(lidar_rangeview_paths, lidar2worlds,
                                   intrinsics)

    elif args.dataset == "waymo":
        # waymo
        root_path = 'data/waymo/waymo_seq1137'
        lidar_rangeview_paths, lidar2worlds, beam_inclinations = get_path_pose_from_json(
            root_path)
        cal_centerpose_bound_scale(lidar_rangeview_paths,
                                   lidar2worlds,
                                   beam_inclinations=beam_inclinations)


if __name__ == '__main__':
    main()
