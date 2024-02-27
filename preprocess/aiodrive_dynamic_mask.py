import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Type, List
import cv2
import numpy as np
from tqdm import tqdm
from aiodrive_loader import AIODriveLoader
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import math
from generate_train_rangeview import LiDAR_2_Pano

exp_bbox = 0.1


def filter_bbox_dataset(pc, OBB_local):
    bbox_mask = np.isnan(pc[:, 0])
    z_min, z_max = min(OBB_local[:, 2]), max(OBB_local[:, 2])
    for i, (c1, c2) in enumerate(
            zip(pc[:, 2] <= z_max + exp_bbox, pc[:, 2] >= z_min - exp_bbox)):
        bbox_mask[i] = (c1 and c2)
    OBB_local = sorted(OBB_local, key=lambda p: p[2])
    OBB_2D = np.array(OBB_local)[:4, :2]
    mask = filter_poly(pc, OBB_2D, bbox_mask)
    pc = pc[mask]
    return pc


def filter_poly(pcs, OBB_2D, bbox_mask):
    OBB_2D = sort_quadrilateral(OBB_2D)
    mask = []
    for i, pc in enumerate(pcs):
        mask.append(not (bbox_mask[i] and is_in_poly(pc[0], pc[1], OBB_2D)))
    return mask


def expand_2d_box(box, width_increase, height_increase):
    center_x = (box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4
    center_y = (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4

    dx1 = box[1][0] - box[0][0]
    dy1 = box[1][1] - box[0][1]
    width = math.sqrt(dx1**2 + dy1**2)
    angle = math.atan2(dy1, dx1)

    new_width = width + width_increase
    new_height = width * (height_increase / width_increase) + height_increase
    new_angle = angle

    dx2 = new_width / 2 * math.cos(new_angle)
    dy2 = new_width / 2 * math.sin(new_angle)
    dx3 = new_height / 2 * math.sin(new_angle)
    dy3 = new_height / 2 * math.cos(new_angle)

    new_box = [
        [center_x - dx2 - dx3, center_y - dy2 + dy3],  
        [center_x + dx2 - dx3, center_y + dy2 + dy3], 
        [center_x + dx2 + dx3, center_y + dy2 - dy3],  
        [center_x - dx2 + dx3, center_y - dy2 - dy3] 
    ]

    return new_box


def sort_quadrilateral(points):
    points = points.tolist()
    top_left = min(points, key=lambda p: p[0] + p[1])

    bottom_right = max(points, key=lambda p: p[0] + p[1])
    points.remove(top_left)
    points.remove(bottom_right)
    bottom_left, top_right = points
    if bottom_left[1] > top_right[1]:
        bottom_left, top_right = top_right, bottom_left

    if np.sqrt((top_left[0] - bottom_right[0])**2 +
               (top_left[1] - bottom_right[1])**2) < 1.5:
        return expand_2d_box([top_left, top_right, bottom_right, bottom_left],
                             0.5, 0.5)

    return [top_left, top_right, bottom_right, bottom_left]


def is_in_poly(px, py, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and
                                       y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return is_in


def lidar2points2d(points, lidar2img, img_shape=(720, 1920)):
    if points.shape[1] == 3:
        points = np.concatenate(
            [points, np.ones(points.shape[0]).reshape((-1, 1))], axis=1)
    pts_2d = points @ lidar2img.T
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    pts_2d = pts_2d[:, :2]

    pts_2d = np.array(pts_2d)

    fov_inds = ((pts_2d[:, 0] < img_shape[1]) | (pts_2d[:, 0] >= 0) |
                (pts_2d[:, 1] < img_shape[0]) | (pts_2d[:, 1] >= 0))
    pts_2d = pts_2d[fov_inds]

    return pts_2d


@dataclass
class ProcessAIODriveMasks:
    """Use cuboid detections to render masks for dynamic objects."""
    aiodrive_parent_dir: Path
    """Path to NuScenes dataset."""
    output_dir: Path

    def main(self) -> None:
        """Generate NuScenes dynamic object masks."""
        cameras = ["2"]
        sequence_id = "64"
        if sequence_id == "64":
            print("Using sqequence 0-63")
            s_frame_id = 0
            e_frame_id = 63  # Inclusive
            val_frame_ids = [13, 26, 39, 52]
        else:
            raise ValueError(f"Invalid sequence id: {sequence_id}")

        frame_ids = list(range(s_frame_id, e_frame_id + 1))
        aioloader = AIODriveLoader(frame_ids=frame_ids,
                                   aiodrive_parent_dir=self.aiodrive_parent_dir,
                                   cameras=cameras)

        lidar2imgs = aioloader.lidar2img
        boxes_filename = aioloader.boxes_filename
        im_paths = aioloader.load_image_paths()
        ori_lidar_filename = aioloader.ori_lidar_filename

        for frame_id in tqdm(frame_ids):
            pkl_path = boxes_filename[frame_id]
            lidar2img = lidar2imgs[frame_id]
            label_path = str(im_paths[frame_id]).replace('image',
                                                         'label').replace(
                                                             'png', 'txt')
            with open(label_path, 'r') as lf:
                lines = lf.readlines()

            with open(pkl_path, 'rb') as f:
                box = pickle.load(f, encoding='iso-8859-1')['lidar']

                # pc = np.fromfile(ori_lidar_filename[frame_id],
                #                  dtype=np.float32).reshape(-1, 4)
                # for b_id, corners_3d in box.items():
                #     pc = filter_bbox_dataset(pc, np.array(corners_3d))

                # H = 66
                # W = 1030
                # intrinsics = (2., 26.9)  # fov_up, fov
                # pc = pc.reshape((-1, 4))
                # pano = LiDAR_2_Pano(pc, H, W, intrinsics)
                # frame_name = str(ori_lidar_filename[frame_id]).split('/')[-1]
                # suffix = frame_name.split('.')[-1]
                # frame_name = frame_name.replace(suffix, 'npy')
                # out_dir = self.aiodrive_parent_dir / "aiodrive_train2"
                # np.save(out_dir / frame_name, pano)

                for camera in cameras:
                    mask = np.ones((720, 1920), dtype=np.uint8)
                    for b_id, corners_3d in box.items():
                        # for camera
                        if lines[b_id].split(" ")[2] == '1.00':
                            continue
                        # if float(lines[b_id].split(" ")[2]) > 0:
                        #     continue
                        # project box to image plane and rasterize each face
                        corners = lidar2points2d(corners_3d, lidar2img)
                        # show_pts_2d(corners, im_paths[frame_id], b_id)
                        corners = np.transpose(corners, axes=(1, 0))
                        corners = np.round(corners).astype(int).T
                        cv2.fillPoly(mask, [corners[[0, 1, 2, 3]]], 0)  # front
                        cv2.fillPoly(mask, [corners[[4, 5, 6, 7]]], 0)  # back
                        cv2.fillPoly(mask, [corners[[0, 1, 5, 4]]], 0)  # top
                        cv2.fillPoly(mask, [corners[[2, 3, 7, 6]]], 0)  # bottom
                        cv2.fillPoly(mask, [corners[[0, 3, 7, 4]]], 0)  # left
                        cv2.fillPoly(mask, [corners[[1, 2, 6, 5]]], 0)  # right
                    maskname = os.path.split(im_paths[frame_id])[1]
                    cv2.imwrite(
                        str(self.output_dir / f'mask_{camera}_t1' / maskname),
                        mask * 255)
                    print(maskname)
                    image = cv2.imread(str(im_paths[frame_id]))
                    image = image * mask[..., None]
                    cv2.imwrite(
                        str(self.output_dir / f'image_mask_{camera}' /
                            maskname), image)


def show_pts_2d(pts_2d, img_path, cam_id, img_shape=(720, 1920)):
    fov_inds = ((pts_2d[:, 0] < img_shape[1]) & (pts_2d[:, 0] >= 0) &
                (pts_2d[:, 1] < img_shape[0]) & (pts_2d[:, 1] >= 0))
    imgfov_pts_2d = pts_2d[fov_inds]

    im = Image.open(img_path)
    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    ax.imshow(im)

    ax.scatter(
        imgfov_pts_2d[:, 0],
        imgfov_pts_2d[:, 1],
        # c=imgfov_pts_2d[:, 2],
        s=5,
        # alpha=0.5
    )
    ax.axis('off')
    plt.savefig(f'temp_vis/points_on_img{cam_id}.png',
                bbox_inches='tight',
                pad_inches=0,
                dpi=500)
    plt.close()


def main():
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    aiodrive_parent_dir = data_dir / "aiodrive"
    output_dir = aiodrive_parent_dir / "AIODrive"

    ProcessAIODriveMask = ProcessAIODriveMasks(
        aiodrive_parent_dir=aiodrive_parent_dir, output_dir=output_dir)
    ProcessAIODriveMask.main()


if __name__ == "__main__":
    main()