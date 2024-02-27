import json
import os

import cv2
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from dataclasses import dataclass, field

from lidarnerf.dataset.base_dataset import get_lidar_rays_with_mask, BaseDataset, get_rays_with_mask


@dataclass
class AIODriveDataset(BaseDataset):

    device: str = "cpu"
    split: str = "train"  # train, val, test
    root_path: str = "data/aiodrive"
    sequence_id: str = "64"
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
        if self.sequence_id == "64":
            print("Using sqequence 0-63")
        else:
            raise ValueError(f"Invalid sequence id: {sequence_id}")

        self.training = self.split in ['train', 'all', 'trainval']
        self.num_rays = self.num_rays if self.training else -1
        self.num_rays_lidar = self.num_rays_lidar if self.training else -1
        # load nerf-compatible format data.
        with open(
                os.path.join(
                    self.root_path,
                    f'transforms_{self.sequence_id}_{self.split}.json'),
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
        self.image_depths = []
        self.masks = []
        self.poses_lidar = []
        self.images_lidar = []
        self.masks_lidar = []
        for f in tqdm.tqdm(frames, desc=f'Loading {self.split} data'):
            f_path = os.path.join(self.root_path, f['file_path'])
            mask_path = str(f_path).replace("image", "mask")
            image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            threshold = 128
            _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
            mask_array = mask / 255
            self.masks.append(mask_array)

            depth_path = str(f_path).replace("image", "depth")
            depth_map = cv2.imread(depth_path, cv2.IMREAD_COLOR)
            depth_map = (256 * 256 * depth_map[:, :, 0] +
                         256 * depth_map[:, :, 1] +
                         depth_map[:, :, 2]) / (256 * 256 * 256 - 1) * 1000
            self.image_depths.append(depth_map * self.scale)

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
            f_lidar_path_ori = os.path.join(self.root_path,
                                            f['lidar_file_path'])
            f_lidar_path = str(f_lidar_path_ori).replace(
                "aiodrive_train", "aiodrive_train2")
            # channel1 None, channel2 intensity , channel3 depth
            pc = np.load(f_lidar_path)
            pc_ori = np.load(f_lidar_path_ori)
            ray_drop = np.where(pc.reshape(-1, 3)[:, 2] == 0.0, 0.0,
                                1.0).reshape(self.H_lidar, self.W_lidar, 1)
            lidar_mask = np.where(
                pc_ori.reshape(-1, 3)[:, 2] == pc.reshape(-1, 3)[:, 2], 1.0,
                0.0).reshape(self.H_lidar, self.W_lidar, 1)
            # cv2.imwrite(os.path.join('mask_lidar.png'),
            #             (lidar_mask * 255).astype(np.uint8))
            self.masks_lidar.append(lidar_mask)
            image_lidar = np.concatenate(
                [ray_drop, pc[:, :, 1, None], pc[:, :, 2, None] * self.scale],
                axis=-1,
            )
            self.poses_lidar.append(pose_lidar)
            self.images_lidar.append(image_lidar)

        self.poses = np.stack(self.poses, axis=0)
        self.poses[:, :3,
                   -1] = (self.poses[:, :3, -1] - self.offset) * self.scale
        self.poses = torch.from_numpy(self.poses)  # [N, 4, 4]

        # self.poses = torch.from_numpy(np.stack(self.poses, axis=0))  # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images,
                                                    axis=0))  # [N, H, W, C]

        if self.masks is not None:
            self.masks = torch.from_numpy(np.stack(
                self.masks, axis=0)).float()  # [N, H, W, C]

        if self.masks_lidar is not None:
            self.masks_lidar = torch.from_numpy(
                np.stack(self.masks_lidar, axis=0)).float()  # [N, H, W, C]

        if self.image_depths is not None:
            self.image_depths = torch.from_numpy(
                np.stack(self.image_depths, axis=0)).float()  # [N, H, W, C]

        self.poses_lidar = np.stack(self.poses_lidar, axis=0)
        self.poses_lidar[:, :3, -1] = (self.poses_lidar[:, :3, -1] -
                                       self.offset) * self.scale
        self.poses_lidar = torch.from_numpy(self.poses_lidar)  # [N, 4, 4]

        if self.images_lidar is not None:
            self.images_lidar = torch.from_numpy(
                np.stack(self.images_lidar, axis=0)).float()  # [N, H, W, C]

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

            if self.masks_lidar is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16:
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.masks_lidar = self.masks_lidar.to(dtype).to(self.device)

            if self.masks is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16:
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.masks = self.masks.to(dtype).to(self.device)

        self.intrinsics_lidar = (2., 26.9)  # fov_up, fov

        # load intrinsics
        fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y'])
        fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x'])
        cx = (transform['cx']) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy']) if 'cy' in transform else (self.H / 2)
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])

    def collate(self, index):

        B = len(index)  # a list of length 1

        results = {}

        if self.enable_rgb:
            poses = self.poses[index].to(self.device)  # [B, 4, 4]
            masks = self.masks[index].to(self.device)  # [B, 4, 4]
            rays = get_rays_with_mask(poses, self.intrinsics, self.H, self.W,
                                      masks, self.num_rays, self.patch_size)
            results.update({
                'H': self.H,
                'W': self.W,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                'pose':
                    self.poses[index]  # for normal
            })

        if self.enable_lidar:
            poses_lidar = self.poses_lidar[index].to(self.device)  # [B, 4, 4]
            masks_lidar = self.masks_lidar[index].to(self.device)  # [B, 4, 4]
            # masks_lidar = None
            rays_lidar = get_lidar_rays_with_mask(
                poses_lidar,
                self.intrinsics_lidar,
                self.H_lidar,
                self.W_lidar,
                masks_lidar,
                self.num_rays_lidar,
                self.patch_size_lidar,
            )

            results.update({
                'H_lidar': self.H_lidar,
                'W_lidar': self.W_lidar,
                'rays_o_lidar': rays_lidar['rays_o'],
                'rays_d_lidar': rays_lidar['rays_d'],
                'pose_lidar':
                    self.poses_lidar[index]  # for normal
            })

        vis_pose = False
        if vis_pose:
            import camtools as ct
            color_rays_o = rays['rays_o'].cpu().numpy().reshape(
                (-1, 3))  # green
            color_rays_d = rays['rays_d'].cpu().numpy().reshape((-1, 3))  # red
            lidar_rays_o = rays_lidar['rays_o'].cpu().numpy().reshape(
                (-1, 3))  # green
            lidar_rays_d = rays_lidar['rays_d'].cpu().numpy().reshape(
                (-1, 3))  # red

            distance = 0.01
            color_rays_s = color_rays_o
            color_rays_e = color_rays_o + color_rays_d * distance
            lidar_rays_s = lidar_rays_o
            lidar_rays_e = lidar_rays_o + lidar_rays_d * distance

            points = np.concatenate(
                [color_rays_s, color_rays_e, lidar_rays_s, lidar_rays_e],
                axis=0,
            )
            colors = np.concatenate(
                [
                    np.ones_like(color_rays_s) * np.array([0, 1, 0]),  # green
                    np.ones_like(color_rays_e) * np.array([1, 0, 0]),  # red
                    np.ones_like(lidar_rays_s) * np.array([0, 1, 0]),  # green
                    np.ones_like(lidar_rays_e) * np.array([1, 0, 0]),  # red
                ],
                axis=0,
            )

            fx, fy, cx, cy = self.intrinsics
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ])
            Ts = [
                ct.convert.pose_to_T(pose) for pose in self.poses.cpu().numpy()
            ]
            Ts_lidar = [
                ct.convert.pose_to_T(pose)
                for pose in self.poses_lidar.cpu().numpy()
            ]
            Ks = [K] * len(Ts)

            np.savez(
                "aiodrive_data_packed.npz",
                points=points,
                colors=colors,
                Ks=Ks,
                Ts=Ts,
                Ts_lidar=Ts_lidar,
            )

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points)
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # camera_frames = ct.camera.create_camera_ray_frames(
            #     Ks,
            #     Ts,
            #     size=0.005,
            #     center_line=False,
            # )
            # lidar_camera_frames = ct.camera.create_camera_ray_frames(
            #     Ks,
            #     Ts_lidar,
            #     size=0.005,
            #     center_line=False,
            # )
            # o3d.visualization.draw_geometries([
            #     camera_frames,
            #     lidar_camera_frames,
            #     pcd,
            # ])
            exit(0)

        if self.images is not None and self.enable_rgb:
            images = self.images[index].to(self.device)  # [B, H, W, 3/4]
            masks = self.masks[index].to(self.device)  # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1,
                                      torch.stack(C * [rays['inds']],
                                                  -1))  # [B, N, 3/4]
                C = 1
                # image_depths = torch.gather(image_depths.view(B, -1, C), 1,
                #                             torch.stack(C * [rays['inds']],
                #                                         -1))  # [B, N, 3/4]
            else:
                image_depths = self.image_depths[index].to(self.device)
                results['image_depths'] = image_depths
            results['images'] = images
            results['masks'] = masks

        if self.images_lidar is not None and self.enable_lidar:
            images_lidar = self.images_lidar[index].to(
                self.device)  # [B, H, W, 3/4]
            masks_lidar = self.masks_lidar[index].to(
                self.device)  # [B, H, W, 3/4]
            if self.training:
                C = images_lidar.shape[-1]
                images_lidar = torch.gather(images_lidar.view(B, -1, C), 1,
                                            torch.stack(
                                                C * [rays_lidar['inds']],
                                                -1))  # [B, N, 3/4]
            results['images_lidar'] = images_lidar
            results['masks_lidar'] = masks_lidar

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
