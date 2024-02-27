from pathlib import Path

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Type, List
from numpy.typing import ArrayLike

import camtools as ct
import numpy as np
import open3d as o3d
import pyquaternion
import torch

from rich.console import Console
from typing_extensions import Literal
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt


def points2depthmap(self, points, height, width):
    height, width = height // self.downsample, width // self.downsample
    depth_map = torch.zeros((height, width), dtype=torch.float32)
    coor = torch.round(points[:, :2] / self.downsample)
    depth = points[:, 2]
    kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (coor[:, 1] >= 0) & (
        coor[:, 1] < height) & (depth < self.grid_config['depth'][1]) & (
            depth >= self.grid_config['depth'][0])
    coor, depth = coor[kept1], depth[kept1]
    ranks = coor[:, 0] + coor[:, 1] * width
    sort = (ranks + depth / 100.).argsort()
    coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

    kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
    kept2[1:] = (ranks[1:] != ranks[:-1])
    coor, depth = coor[kept2], depth[kept2]
    coor = coor.to(torch.long)
    depth_map[coor[:, 1], coor[:, 0]] = depth
    return depth_map


def lidar2points2d(points, intrinsics, lidar2cam_RT):
    """
    points.shape, intrinsics.shape, lidar2cam_RT.shape
    (points_nums, 4) (3, 4) (4, 4)
    """
    if points.shape[1] == 3:
        points = np.concatenate(
            [points, np.ones(points.shape[0]).reshape((-1, 1))], axis=1)
    points_2d = points @ lidar2cam_RT.T
    points_2d = points_2d[:, :3] @ intrinsics[:3, :3].T
    return points_2d


def show_pts_2d(pts_2d, img_path, cam_id, sample, img_shape=(376, 1408)):
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    fov_inds = ((pts_2d[:, 0] < img_shape[1]) & (pts_2d[:, 0] >= 0) &
                (pts_2d[:, 1] < img_shape[0]) & (pts_2d[:, 1] >= 0))
    imgfov_pts_2d = pts_2d[fov_inds]

    import cv2
    depth_map = cv2.imread(
        f'/data/aiodrive/AIODrive/depth_2/{sample}.png',
        cv2.IMREAD_COLOR)
    # depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
    depth_map = (256 * 256 * depth_map[:, :, 0] + 256 * depth_map[:, :, 1] +
                 depth_map[:, :, 2]) / (256 * 256 * 256 - 1) * 1000
    # depth_map = (256*256*depth_map[:,:,2] + 256*depth_map[:,:,1] + depth_map[:,:,0]) / (256*256*256-1) * 1000

    err = []
    fake_img = np.zeros(img_shape)
    for x, y, d in imgfov_pts_2d:
        # fake_img[int(y)][int(x)] = d  # can color
        fake_img[int(y)][int(x)] = 1
        # print(d, depth_map[int(y)][int(x)])
        err.append(d - depth_map[int(y)][int(x)])
    print(np.mean(err))
    return np.mean(err)

    # import imageio
    # imageio.imsave("points_2d.png", fake_img)
    # im = Image.open(img_path)
    # fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    # ax.imshow(im)
    # ax.scatter(imgfov_pts_2d[:, 0],
    #            imgfov_pts_2d[:, 1],
    #            c=imgfov_pts_2d[:, 2],
    #            s=0.1,
    #            alpha=0.5)
    # ax.axis('off')
    # plt.savefig(f'points_on_img{cam_id}.png',
    #             bbox_inches='tight',
    #             pad_inches=0,
    #             dpi=500)


@dataclass
class AIODriveLoader:
    frame_ids: List
    aiodrive_parent_dir: Path
    cameras: Tuple[Literal['0', '1', '2', '3', '4', 'ALL'], ...] = ("ALL",)

    def __post_init__(self):
        """
        As a @dataclass, the __init__() function is automatically generated.
        This function is called after __init__().
        """
        self.aiodrive_extract_root = self.aiodrive_parent_dir / 'AIODrive'
        self.lidar_range_image_path = self.aiodrive_parent_dir / 'aiodrive_train'
        camera_list = [
            '0',
            '1',
            '2',
            '3',
            '4',
        ]
        if "ALL" in self.cameras:
            self.cameras = [
                '0',
                '1',
                '2',
                '3',
                '4',
            ]
        self.cameras = [camera_list.index(camera) for camera in self.cameras]

        self.lidar_dir = self.aiodrive_extract_root / f'lidar_velodyne'
        self.calib_path = self.aiodrive_extract_root / 'calib'
        self._parse_dataset()

    def _parse_dataset(self) -> None:
        """
        This function does:
        - Check and load image and lidar paths
        - Load ALL cameras and lidar
        - Split images and cameras, and only keep the selected ones
        """
        # Get samples for scene.
        samples = os.listdir(self.lidar_dir)
        samples = [sample.split('.')[0] for sample in samples]
        samples.sort(key=lambda x: int(x))

        frame_data = []
        # depth_err_list= []
        for sample_id, sample in enumerate(samples):
            pose_file = self.calib_path / f'{sample}.txt'
            with open(pose_file, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                          ]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                          ]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                          ]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                          ]).reshape([3, 4])
            P4 = np.array([float(info) for info in lines[4].split(' ')[1:13]
                          ]).reshape([3, 4])
            camera_intrinsic_list = [P0, P1, P2, P3, P4]

            R0_rect = self.cart_to_homo(
                np.array([float(info) for info in lines[6].split(' ')[1:10]
                         ]).reshape([3, 3]))
            Tr_velo_to_p2 = self.cart_to_homo(
                np.array([float(info) for info in lines[7].split(' ')[1:13]
                         ]).reshape([3, 4]))
            p2_to_world = self.cart_to_homo(
                np.array([float(info) for info in lines[9].split(' ')[1:13]
                         ]).reshape([3, 4]))

            lidar_filename = self.lidar_range_image_path / f'{sample}.npy'
            lidar_pose = p2_to_world @ Tr_velo_to_p2

            selected = sample_id in self.frame_ids

            ori_lidar_filename = self.lidar_dir / f'{sample}.bin'
            lidar_points = np.fromfile(self.lidar_dir / f'{sample}.bin',
                                       dtype=np.float32).reshape(-1, 4)[:, :3]

            for camera in self.cameras:
                image_filename = self.aiodrive_extract_root / f'image_{camera}' / f'{sample}.png'
                P = self.cart_to_homo(camera_intrinsic_list[camera])
                fx = P[0, 0]
                fy = P[1, 1]
                cx = P[0, 2]
                cy = P[1, 2]

                intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

                lidar2img = P @ R0_rect @ Tr_velo_to_p2
                lidar2camera = np.linalg.inv(
                    self.cart_to_homo(intrinsic)) @ lidar2img
                pose = lidar_pose @ np.linalg.inv(
                    lidar2camera)
            
            boxes_filename = self.aiodrive_extract_root / 'box' / f'{sample}.pkl'

            frame_data.append({
                "lidar_filename": lidar_filename,
                "lidar_pose": lidar_pose,
                "selected": selected,
                "image_filename": image_filename,
                "intrinsic": intrinsic,
                "lidar2img": lidar2img,
                "pose": pose,
                "boxes_filename": boxes_filename,
                'ori_lidar_filename': ori_lidar_filename
            })
        # print(depth_err_list, np.mean(depth_err_list))
        # Choose image_filenames and poses based on split, but after auto
        # orient and scaling the poses.
        lidar_paths = [f["lidar_filename"] for f in frame_data if f["selected"]]
        boxes_filename = [
            f["boxes_filename"] for f in frame_data if f["selected"]
        ]
        ori_lidar_filename = [
            f["ori_lidar_filename"] for f in frame_data if f["selected"]
        ]
        im_paths = [f["image_filename"] for f in frame_data if f["selected"]]
        Ks = [f["intrinsic"] for f in frame_data if f["selected"]]
        poses = [f["pose"] for f in frame_data if f["selected"]]
        lidar_poses = [f["lidar_pose"] for f in frame_data if f["selected"]]
        lidar2img = [f["lidar2img"] for f in frame_data if f["selected"]]
        Ts = [
            # ct.convert.pose_to_T(ct.convert.pose_blender_to_pinhole(pose))
            ct.convert.pose_to_T(pose) for pose in poses
        ]

        # Sanity checks.
        for im_path in im_paths:
            if not im_path.is_file():
                raise ValueError(f"Image {im_path} does not exist")

        # Save to class.
        self.im_paths = im_paths
        self.Ks = np.array(Ks).astype(np.float32)
        self.Ts = np.array(Ts).astype(np.float32)
        self.lidar2worlds = np.array(lidar_poses).astype(np.float32)
        self.lidar_paths = lidar_paths
        self.lidar2img = lidar2img
        self.boxes_filename = boxes_filename
        self.ori_lidar_filename = ori_lidar_filename

    def load_image_paths(self) -> Tuple[List[Path], List[Path]]:
        return self.im_paths

    def load_cameras(self) -> Tuple[ArrayLike, ArrayLike]:
        """
        Returns Ks and Ts.
        """
        return self.Ks, self.Ts

    def load_lidar_paths(self) -> Tuple[List[Path]]:
        return self.lidar_paths

    def load_lidars(self) -> Tuple[ArrayLike]:
        """
        Returns Ks and Ts.
        """
        return self.lidar2worlds

    def cart_to_homo(self, mat):
        """Convert transformation matrix in Cartesian coordinates to
        homogeneous format.

        Args:
            mat (np.ndarray): Transformation matrix in Cartesian.
                The input matrix shape is 3x3 or 3x4.

        Returns:
            np.ndarray: Transformation matrix in homogeneous format.
                The matrix shape is 4x4.
        """
        ret = np.eye(4)
        if mat.shape == (3, 3):
            ret[:3, :3] = mat
        elif mat.shape == (3, 4):
            ret[:3, :] = mat
        else:
            raise ValueError(mat.shape)
        return ret


def main():
    sequence_id = "64"
    if sequence_id == "64":
        print("Using sqequence 0-63")
        s_frame_id = 0
        e_frame_id = 63  # Inclusive
        val_frame_ids = [13, 26, 39, 52]
    else:
        raise ValueError(f"Invalid sequence id: {sequence_id}")

    frame_ids = list(range(s_frame_id, e_frame_id + 1))
    aioloader = AIODriveLoader(
        frame_ids=frame_ids,
        aiodrive_parent_dir=Path(
            'data/aiodrive'),
        cameras=["2"])


if __name__ == "__main__":
    main()
