import numpy as np
import torch
import trimesh
from packaging import version as pver
from dataclasses import dataclass
from PIL import Image
import matplotlib.pyplot as plt


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


def show_pts_2d(pts_2d, img_path, img_shape=(376, 1408)):
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    fov_inds = ((pts_2d[:, 0] < img_shape[1]) & (pts_2d[:, 0] >= 0) &
                (pts_2d[:, 1] < img_shape[0]) & (pts_2d[:, 1] >= 0))
    imgfov_pts_2d = pts_2d[fov_inds]

    from PIL import Image
    import matplotlib.pyplot as plt
    im = Image.open(img_path)
    fig, ax = plt.subplots(1, 1)
    ax.imshow(im)
    ax.scatter(
        imgfov_pts_2d[:, 0],
        imgfov_pts_2d[:, 1],
        c=imgfov_pts_2d[:, 2],
        marker='.',
        s=0.005,
        # alpha=0.8
    )
    ax.axis('off')
    plt.savefig(f'vis/points_on_img.png',
                bbox_inches='tight',
                pad_inches=0,
                dpi=2000)


def get_lidar_depth_image(pts_2d, img_shape=(376, 1408)):
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    fov_inds = ((pts_2d[:, 0] < img_shape[1]) & (pts_2d[:, 0] >= 0) &
                (pts_2d[:, 1] < img_shape[0]) & (pts_2d[:, 1] >= 0))
    imgfov_pts_2d = pts_2d[fov_inds]
    fake_img = np.zeros(img_shape)
    for x, y, d in imgfov_pts_2d:
        if fake_img[int(y)][int(x)] == 0 or fake_img[int(y)][int(x)] > d:
            fake_img[int(y)][int(x)] = d
    return fake_img


def project_points(points_3d, intrinsics, lidar2world, camera2world,
                   image_width, image_height):
    # Transform lidar points to world coordinates
    if points_3d.shape[1] == 3:
        points_3d = np.concatenate(
            [points_3d, np.ones(points_3d.shape[0]).reshape((-1, 1))], axis=1)
    points_3d_camera = points_3d @ (np.linalg.inv(camera2world) @ lidar2world).T

    # Project camera coordinates to image coordinates
    fx = intrinsics[0]
    fy = intrinsics[1]
    cx = intrinsics[2]
    cy = intrinsics[3]
    points_2d = np.zeros((points_3d_camera.shape[0], 2))
    points_3d_camera[:, 2] = np.clip(points_3d_camera[:, 2],
                                     a_min=1e-5,
                                     a_max=99999)
    points_2d[:, 0] = fx * points_3d_camera[:, 0] / points_3d_camera[:, 2] + cx
    points_2d[:, 1] = fy * points_3d_camera[:, 1] / points_3d_camera[:, 2] + cy

    # Find points that lie within the image boundaries
    valid_indices = np.where((points_2d[:, 0] >= 0) &
                             (points_2d[:, 0] < image_width) &
                             (points_2d[:, 1] >= 0) &
                             (points_2d[:, 1] < image_height))[0]

    return points_2d, valid_indices


def plot_pts_2d(pts_2d, img_path, cam_id, img_shape=(376, 1408)):
    # pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
    # pts_2d[:, 0] /= pts_2d[:, 2]
    # pts_2d[:, 1] /= pts_2d[:, 2]
    fov_inds = ((pts_2d[:, 0] < img_shape[1]) & (pts_2d[:, 0] >= 0) &
                (pts_2d[:, 1] < img_shape[0]) & (pts_2d[:, 1] >= 0))
    imgfov_pts_2d = pts_2d[fov_inds]
    fake_img = np.zeros(img_shape)
    for x, y in imgfov_pts_2d:
        # fake_img[int(y)][int(x)] = d  # can color
        fake_img[int(y)][int(x)] = 1
    # imageio.imsave("points_2d.png", fake_img)
    im = Image.open(img_path)
    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    ax.imshow(im)
    ax.scatter(
        imgfov_pts_2d[:, 0],
        imgfov_pts_2d[:, 1],
        # c=imgfov_pts_2d[:, 2],
        s=0.1,
        alpha=0.5)
    ax.axis('off')
    plt.savefig(f'points_on_img{cam_id}.png',
                bbox_inches='tight',
                pad_inches=0,
                dpi=500)


def custom_meshgrid(*args):
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


@torch.cuda.amp.autocast(enabled=False)
def get_rays_with_mask(poses, intrinsics, H, W, mask=None, N=-1, patch_size=1):
    """ get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    """

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device),
                           torch.linspace(0, H - 1, H, device=device))  # float
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    results = {}
    if N > 0:
        N = min(N, H * W)

        if patch_size > 1:
            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner
            # pixels... but I don't have other ideas.
            num_patch = N // (patch_size**2)
            inds_x = torch.randint(0,
                                   H - patch_size,
                                   size=[num_patch],
                                   device=device)
            inds_y = torch.randint(0,
                                   W - patch_size,
                                   size=[num_patch],
                                   device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device),
                                     torch.arange(patch_size, device=device))
            offsets = torch.stack(
                [pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  # [np, p^2, 2]
            inds = inds.view(-1, 2)  # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten

            inds = inds.expand([B, N])
        elif mask is not None:
            indices = torch.nonzero(mask.view(mask.size(0), -1), as_tuple=False)
            inds = indices[torch.randperm(indices.size(0))[:N]]
            inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten
            assert B == 1
            inds = inds.expand([B, N])
        else:
            inds = torch.randint(0, H * W, size=[N],
                                 device=device)  # may duplicate
            inds = inds.expand([B, N])

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


@torch.cuda.amp.autocast(enabled=False)
def get_lidar_rays_with_mask(poses,
                             intrinsics=None,
                             H=0,
                             W=0,
                             mask=None,
                             N=-1,
                             patch_size=1,
                             beam_inclinations=None):
    """
    Get lidar rays.

    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [2]
        H, W, N: int
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    """
    device = poses.device
    B = poses.shape[0]

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device),
                           torch.linspace(0, H - 1, H, device=device))  # float
    # i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    # j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    i = i.t().reshape([1, H * W]).expand([B, H * W])
    j = j.t().reshape([1, H * W]).expand([B, H * W])
    results = {}
    if N > 0:
        N = min(N, H * W)

        if isinstance(patch_size, int):
            patch_size_x, patch_size_y = patch_size, patch_size
        elif len(patch_size) == 1:
            patch_size_x, patch_size_y = patch_size[0], patch_size[0]
        else:
            patch_size_x, patch_size_y = patch_size

        if patch_size_x > 0:
            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner
            # pixels... but I don't have other ideas.
            num_patch = N // (patch_size_x * patch_size_y)
            inds_x = torch.randint(0,
                                   H - patch_size_x,
                                   size=[num_patch],
                                   device=device)
            inds_y = torch.randint(0,
                                   W - patch_size_y,
                                   size=[num_patch],
                                   device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size_x, device=device),
                                     torch.arange(patch_size_y, device=device))
            offsets = torch.stack(
                [pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  # [np, p^2, 2]
            inds = inds.view(-1, 2)  # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten

            inds = inds.expand([B, N])
        elif mask is not None:
            indices = torch.nonzero(mask.view(mask.size(0), -1), as_tuple=False)
            inds = indices[torch.randperm(indices.size(0))[:N]]
            inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten
            assert B == 1
            inds = inds.expand([B, N])

        else:
            inds = torch.randint(0, H * W, size=[N],
                                 device=device)  # may duplicate
            inds = inds.expand([B, N])

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])
        results['inds'] = inds

    beta = -(i - W / 2) / W * 2 * np.pi
    if beam_inclinations is not None:
        alpha = torch.FloatTensor(beam_inclinations[::-1]).to(device)
        alpha = alpha.reshape([1, H, 1]).expand([B, H, W]).reshape([B, H * W])
        alpha = torch.gather(alpha, -1, inds)
    else:
        fov_up, fov = intrinsics
        alpha = (fov_up - j / H * fov) / 180 * np.pi

    directions = torch.stack([
        torch.cos(alpha) * torch.cos(beta),
        torch.cos(alpha) * torch.sin(beta),
        torch.sin(alpha)
    ], -1)
    # directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)
    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


@torch.cuda.amp.autocast(enabled=False)
def get_lidar_rays(poses,
                   intrinsics=None,
                   H=0,
                   W=0,
                   N=-1,
                   patch_size=1,
                   beam_inclinations=None):
    """
    Get lidar rays.

    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [2]
        H, W, N: int
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    """
    device = poses.device
    B = poses.shape[0]

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device),
                           torch.linspace(0, H - 1, H, device=device))  # float
    # i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    # j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    i = i.t().reshape([1, H * W]).expand([B, H * W])
    j = j.t().reshape([1, H * W]).expand([B, H * W])
    results = {}
    if N > 0:
        N = min(N, H * W)

        if isinstance(patch_size, int):
            patch_size_x, patch_size_y = patch_size, patch_size
        elif len(patch_size) == 1:
            patch_size_x, patch_size_y = patch_size[0], patch_size[0]
        else:
            patch_size_x, patch_size_y = patch_size

        if patch_size_x > 0:
            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner
            # pixels... but I don't have other ideas.
            num_patch = N // (patch_size_x * patch_size_y)
            inds_x = torch.randint(0,
                                   H - patch_size_x,
                                   size=[num_patch],
                                   device=device)
            inds_y = torch.randint(0,
                                   W - patch_size_y,
                                   size=[num_patch],
                                   device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size_x, device=device),
                                     torch.arange(patch_size_y, device=device))
            offsets = torch.stack(
                [pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  # [np, p^2, 2]
            inds = inds.view(-1, 2)  # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten

            inds = inds.expand([B, N])

        else:
            inds = torch.randint(0, H * W, size=[N],
                                 device=device)  # may duplicate
            inds = inds.expand([B, N])

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])
        results['inds'] = inds

    beta = -(i - W / 2) / W * 2 * np.pi
    if beam_inclinations is not None:
        alpha = torch.FloatTensor(beam_inclinations[::-1]).to(device)
        alpha = alpha.reshape([1, H, 1]).expand([B, H, W]).reshape([B, H * W])
        alpha = torch.gather(alpha, -1, inds)
    else:
        fov_up, fov = intrinsics
        alpha = (fov_up - j / H * fov) / 180 * np.pi

    directions = torch.stack([
        torch.cos(alpha) * torch.cos(beta),
        torch.cos(alpha) * torch.sin(beta),
        torch.sin(alpha)
    ], -1)
    # directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)
    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, patch_size=1):
    """ get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    """

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W - 1, W, device=device),
                           torch.linspace(0, H - 1, H, device=device))  # float
    i = i.t().reshape([1, H * W]).expand([B, H * W]) + 0.5
    j = j.t().reshape([1, H * W]).expand([B, H * W]) + 0.5

    results = {}
    if N > 0:
        N = min(N, H * W)

        if patch_size > 1:
            # random sample left-top cores.
            # NOTE: this impl will lead to less sampling on the image corner
            # pixels... but I don't have other ideas.
            num_patch = N // (patch_size**2)
            inds_x = torch.randint(0,
                                   H - patch_size,
                                   size=[num_patch],
                                   device=device)
            inds_y = torch.randint(0,
                                   W - patch_size,
                                   size=[num_patch],
                                   device=device)
            inds = torch.stack([inds_x, inds_y], dim=-1)  # [np, 2]

            # create meshgrid for each patch
            pi, pj = custom_meshgrid(torch.arange(patch_size, device=device),
                                     torch.arange(patch_size, device=device))
            offsets = torch.stack(
                [pi.reshape(-1), pj.reshape(-1)], dim=-1)  # [p^2, 2]

            inds = inds.unsqueeze(1) + offsets.unsqueeze(0)  # [np, p^2, 2]
            inds = inds.view(-1, 2)  # [N, 2]
            inds = inds[:, 0] * W + inds[:, 1]  # [N], flatten

            inds = inds.expand([B, N])

        else:
            inds = torch.randint(0, H * W, size=[N],
                                 device=device)  # may duplicate
            inds = inds.expand([B, N])

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H * W, device=device).expand([B, H * W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2)  # (B, N, 3)

    rays_o = poses[..., :3, 3]  # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ],
                        dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c],
                         [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


@dataclass
class BaseDataset:

    pass
