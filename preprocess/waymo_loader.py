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


@dataclass
class WaymoLoader:
    waymo_parent_dir: Path
    version: Literal["v120", "v140"] = "v120"
    cameras: Tuple[Literal[
        'FRONT',
        'FRONT_LEFT',
        'FRONT_RIGHT',
        'SIDE_LEFT',
        'SIDE_RIGHT',
    ], ...] = ("ALL",)
    lidar: Literal["TOP"] = ("TOP")
    train_split_fraction: float = 0.9
    split: Literal["train", "test"] = "train"

    def __post_init__(self):
        """
        As a @dataclass, the __init__() function is automatically generated.
        This function is called after __init__().
        """
        self.waymo_extract_root = self.waymo_parent_dir / 'waymo_extract'
        self.lidar_range_image_path = self.waymo_parent_dir / 'waymo_train'
        camera_list = [
            'FRONT',
            'FRONT_LEFT',
            'FRONT_RIGHT',
            'SIDE_LEFT',
            'SIDE_RIGHT',
        ]
        lidar_list = ['TOP']
        if "ALL" in self.cameras:
            self.cameras = [
                'FRONT',
                'FRONT_LEFT',
                'FRONT_RIGHT',
                'SIDE_LEFT',
                'SIDE_RIGHT',
            ]
        self.cameras = [camera_list.index(camera) for camera in self.cameras]
        self.lidar = [lidar_list.index(lidar) for lidar in self.lidar][0]

        self.lidar_dir = self.waymo_extract_root / f'lidar_{self.lidar}'
        self.calib_path = self.waymo_extract_root / 'calib_0' / '0000000.txt'
        with open(self.calib_path, 'r') as f:
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
        self.camera_intrinsic_list = [P0, P1, P2, P3, P4]
        self.beam_inclinations = np.array(
            [float(info) for info in lines[5].split(' ')[1:]])
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

        # Compute train/eval indices.
        num_samples = len(samples)
        num_train_samples = math.ceil(num_samples * self.train_split_fraction)
        all_ids = np.arange(num_samples)
        train_ids = np.linspace(0,
                                num_samples - 1,
                                num_train_samples,
                                dtype=int)
        eval_ids = np.setdiff1d(all_ids, train_ids)
        if self.split == "train":
            indices = train_ids
        elif self.split in ["val", "test"]:
            indices = eval_ids
        else:
            raise ValueError(f"Unknown dataparser split {self.split}")
        print(f"Train ids   : {train_ids}")
        print(f"Eval ids    : {eval_ids}")
        print(f"Selected ids: {indices}")
        selected_indices = set(indices.tolist())

        frame_data = []
        for sample_id, sample in enumerate(samples):
            # lidar_filename = self.lidar_dir / f'{sample}.bin'
            lidar_filename = self.lidar_range_image_path / f'{sample}.npy'
            lidar_pose_file = self.waymo_extract_root / f'lidar2world_{self.lidar}' / f'{sample}.txt'
            lidar_pose = np.loadtxt(lidar_pose_file)

            selected = sample_id in selected_indices

            for camera in self.cameras:
                image_filename = self.waymo_extract_root / f'image_{camera}' / f'{sample}.jpg'
                pose_file = self.waymo_extract_root / f'cam2world_{camera}' / f'{sample}.txt'
                pose = np.loadtxt(pose_file)
                intrinsic = self.camera_intrinsic_list[camera]

            frame_data.append({
                "lidar_filename": lidar_filename,
                "lidar_pose": lidar_pose,
                "selected": selected,
                "image_filename": image_filename,
                "intrinsic": intrinsic,
                "pose": pose,
            })

        # Choose image_filenames and poses based on split, but after auto
        # orient and scaling the poses.
        lidar_paths = [f["lidar_filename"] for f in frame_data if f["selected"]]
        im_paths = [f["image_filename"] for f in frame_data if f["selected"]]
        Ks = [f["intrinsic"] for f in frame_data if f["selected"]]
        poses = [f["pose"] for f in frame_data if f["selected"]]
        lidar_poses = [f["lidar_pose"] for f in frame_data if f["selected"]]
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


def main():
    pass


if __name__ == "__main__":
    main()
