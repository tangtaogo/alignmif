import json
import os

import cv2
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass, field

from lidarnerf.dataset.base_dataset import get_lidar_rays, BaseDataset, get_rays, show_pts_2d, project_points, lidar2points2d, show_pts_2d, get_lidar_depth_image
from lidarnerf.convert import pano_to_lidar
from PIL import Image
import matplotlib.pyplot as plt


@dataclass
class WaymoDataset(BaseDataset):

    device: str = "cpu"
    split: str = "train"  # train, val, test
    root_path: str = "data/waymo"
    sequence_id: str = "scene-0001"  # not support now
    preload: bool = True  # preload data into GPU
    scale: float = 1  # camera radius scale to make sure camera are inside the bounding box.
    offset: list = field(default_factory=list)  # offset
    # bound = opt.bound  # bounding box half length, also used as the radius to random sample poses.
    fp16: bool = True  # if preload, load into fp16.
    patch_size: int = 1  # size of the image to extract from the scene.
    patch_size_lidar: int = 1  # size of the image to extract from the Lidar.
    enable_lidar: bool = True
    enable_rgb: bool = False
    color_space: str = 'srgb'
    num_rays: int = 4096
    num_rays_lidar: int = 4096

    def __post_init__(self):
        # print(f"Using scene: {self.sequence_id}")

        self.training = self.split in ['train', 'all', 'trainval']
        self.num_rays = self.num_rays if self.training else -1
        self.num_rays_lidar = self.num_rays_lidar if self.training else -1
        # load nerf-compatible format data.
        # with open(
        #         os.path.join(
        #             self.root_path,
        #             f'transforms_{self.sequence_id}_{self.split}.json'),
        #         'r') as f:
        with open(os.path.join(self.root_path, f'transforms_{self.split}.json'),
                  'r') as f:
            transform = json.load(f)

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h'])
            self.W = int(transform['w'])
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None

        if 'h_lidar' in transform and 'w_lidar' in transform:
            self.H_lidar = int(transform['h_lidar'])
            self.W_lidar = int(transform['w_lidar'])

        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...

        self.poses = []
        self.images = []
        self.poses_lidar = []
        self.images_lidar = []
        self.image_paths = []
        self.image_depths = []
        for f in tqdm.tqdm(frames, desc=f'Loading {self.split} data'):
            f_path = os.path.join(self.root_path, f['file_path'])
            self.image_paths.append(f_path)
            pose = np.array(f['transform_matrix'], dtype=np.float32)  # [4, 4]
            image = cv2.imread(f_path,
                               cv2.IMREAD_UNCHANGED)  # [H, W, 3] o [H, W, 4]
            # add support for the alpha channel as a mask.
            if image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            if image.shape[0] != self.H or image.shape[1] != self.W:
                image = cv2.resize(image, (self.W, self.H),
                                   interpolation=cv2.INTER_AREA)
            image = image.astype(np.float32) / 255  # [H, W, 3/4]
            self.poses.append(pose)
            self.images.append(image)

            pose_lidar = np.array(f['lidar2world'], dtype=np.float32)  # [4, 4]
            f_lidar_path = os.path.join(self.root_path, f['lidar_file_path'])
            # channel1 None, channel2 intensity , channel3 depth
            pc = np.load(f_lidar_path)
            ray_drop = np.where(pc.reshape(-1, 3)[:, 2] == 0.0, 0.0,
                                1.0).reshape(self.H_lidar, self.W_lidar, 1)
            image_lidar = np.concatenate(
                [
                    ray_drop,
                    np.clip(pc[:, :, 1, None], 0, 1),
                    pc[:, :, 2, None] * self.scale
                ],
                axis=-1,
            )
            self.poses_lidar.append(pose_lidar)
            self.images_lidar.append(image_lidar)
            if not self.training:
                fl_x = (transform['fl_x']
                        if 'fl_x' in transform else transform['fl_y'])
                fl_y = (transform['fl_y']
                        if 'fl_y' in transform else transform['fl_x'])
                cx = (transform['cx']) if 'cx' in transform else (self.W / 2)
                cy = (transform['cy']) if 'cy' in transform else (self.H / 2)
                #  self.intrinsics = np.array([fl_x, fl_y, cx, cy])
                intrinsic = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]])
                from lidarnerf.convert import pano_to_lidar
                pc = pano_to_lidar(
                    pc[:, :, 2],
                    beam_inclinations=transform['beam_inclinations'])
                pts_2d = lidar2points2d(pc, intrinsic,
                                        np.linalg.inv(pose) @ pose_lidar)
                image_depth = get_lidar_depth_image(pts_2d,
                                                    img_shape=(self.H, self.W))
                self.image_depths.append(image_depth)

        self.poses = np.stack(self.poses, axis=0)
        self.poses[:, :3,
                   -1] = (self.poses[:, :3, -1] - self.offset) * self.scale
        self.poses = torch.from_numpy(self.poses)  # [N, 4, 4]

        # self.poses = torch.from_numpy(np.stack(self.poses, axis=0))  # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images,
                                                    axis=0))  # [N, H, W, C]

        self.poses_lidar = np.stack(self.poses_lidar, axis=0)
        self.poses_lidar[:, :3, -1] = (self.poses_lidar[:, :3, -1] -
                                       self.offset) * self.scale
        self.poses_lidar = torch.from_numpy(self.poses_lidar)  # [N, 4, 4]

        if self.images_lidar is not None:
            self.images_lidar = torch.from_numpy(
                np.stack(self.images_lidar, axis=0)).float()  # [N, H, W, C]

        if len(self.image_depths) > 0:
            self.image_depths = torch.from_numpy(
                np.stack(self.image_depths, axis=0)).float()  # [N, H, W, C]
        else:
            self.image_depths = None

        # calculate mean radius of all camera poses
        # self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)

            self.poses_lidar = self.poses_lidar.to(self.device)
            if self.images_lidar is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16:
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images_lidar = self.images_lidar.to(dtype).to(self.device)

        # load intrinsics
        fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y'])
        fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x'])
        cx = (transform['cx']) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy']) if 'cy' in transform else (self.H / 2)
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

        self.intrinsics_lidar = None
        self.beam_inclinations = transform['beam_inclinations']

    def collate(self, index):

        B = len(index)  # a list of length 1

        results = {}

        if self.enable_rgb:
            poses = self.poses[index].to(self.device)  # [B, 4, 4]
            rays = get_rays(poses, self.intrinsics, self.H, self.W,
                            self.num_rays, self.patch_size)
            results.update({
                'H': self.H,
                'W': self.W,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d']
            })

        if self.enable_lidar:
            poses_lidar = self.poses_lidar[index].to(self.device)  # [B, 4, 4]
            rays_lidar = get_lidar_rays(
                poses=poses_lidar,
                H=self.H_lidar,
                W=self.W_lidar,
                beam_inclinations=self.beam_inclinations,
                N=self.num_rays_lidar,
                patch_size=self.patch_size_lidar,
            )

            results.update({
                'H_lidar': self.H_lidar,
                'W_lidar': self.W_lidar,
                'rays_o_lidar': rays_lidar['rays_o'],
                'rays_d_lidar': rays_lidar['rays_d']
            })

        if self.images is not None and self.enable_rgb:
            images = self.images[index].to(self.device)  # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1,
                                      torch.stack(C * [rays['inds']],
                                                  -1))  # [B, N, 3/4]
            elif self.image_depths is not None:
                image_depths = self.image_depths[index].to(self.device)
                results['image_depths'] = image_depths
            results['images'] = images

        if self.images_lidar is not None and self.enable_lidar:
            images_lidar = self.images_lidar[index].to(
                self.device)  # [B, H, W, 3/4]
            if self.training:
                C = images_lidar.shape[-1]
                images_lidar = torch.gather(images_lidar.view(B, -1, C), 1,
                                            torch.stack(
                                                C * [rays_lidar['inds']],
                                                -1))  # [B, N, 3/4]
            results['images_lidar'] = images_lidar

        return results

    def dataloader(self):
        size = len(self.poses_lidar)
        loader = DataLoader(list(range(size)),
                            batch_size=1,
                            collate_fn=self.collate,
                            shuffle=self.training,
                            num_workers=0)
        loader._data = self
        loader.has_gt = self.images_lidar is not None
        return loader

    def __len__(self):
        """
        Returns # of frames in this dataset.
        """
        num_frames = len(self.poses_lidar)
        return num_frames
