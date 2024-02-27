"""Adapted from Waymo to KITTI converter of mmdetection3d and waymo-open-dataset
"""

try:
    from waymo_open_dataset import dataset_pb2
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-1-0==1.2.0" '
        'to install the official devkit first.')

import os
from glob import glob
from os.path import exists, join

import mmengine
import numpy as np
import tensorflow as tf
from waymo_open_dataset.utils import range_image_utils, transform_utils
from waymo_open_dataset.utils.frame_utils import \
    parse_range_image_and_camera_projection


class WaymoExtractor(object):
    """Waymo extractor.

    This class serves as the converter to change the waymo raw data to KITTI
    format.

    Args:
        load_dir (str): Directory to load waymo raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (int, optional): Number of workers for the parallel process.
            Defaults to 64.
        test_mode (bool, optional): Whether in the test_mode.
            Defaults to False.
        save_cam_sync_labels (bool, optional): Whether to save cam sync labels.
            Defaults to True.
    """

    def __init__(self, load_dir, save_dir, prefix, workers=64):
        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.selected_waymo_locations = None
        self.filter_no_label_zone_points = True

        # keep the order defined by the official protocol
        self.cam_list = [
            '_FRONT',
            '_FRONT_LEFT',
            '_FRONT_RIGHT',
            '_SIDE_LEFT',
            '_SIDE_RIGHT',
        ]
        self.lidar_list = ['TOP', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR']

        self.load_dir = load_dir
        self.save_dir = save_dir
        self.prefix = prefix
        self.workers = int(workers)

        self.tfrecord_pathnames = sorted(glob(join(self.load_dir,
                                                   '*.tfrecord')))

        self.image_save_dir = f'{self.save_dir}/image_'
        self.cam2world_save_dir = f'{self.save_dir}/cam2world_'
        self.point_cloud_save_dir = f'{self.save_dir}/lidar_'
        self.lidar2world_save_dir = f'{self.save_dir}/lidar2world_'
        self.calib_save_dir = f'{self.save_dir}/calib_'

        self.create_folder()

    def save_calib(self, frame, file_idx, frame_idx):
        """Parse and save the calibration data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        camera_calibs = []
        calib_context = ''

        for camera in frame.context.camera_calibrations:
            # intrinsic parameters
            camera_calib = np.zeros((3, 4))
            camera_calib[0, 0] = camera.intrinsic[0]
            camera_calib[1, 1] = camera.intrinsic[1]
            camera_calib[0, 2] = camera.intrinsic[2]
            camera_calib[1, 2] = camera.intrinsic[3]
            camera_calib[2, 2] = 1
            camera_calib = list(camera_calib.reshape(12))
            camera_calib = [f'{i:e}' for i in camera_calib]
            camera_calibs.append(camera_calib)

        calibrations = sorted(frame.context.laser_calibrations,
                              key=lambda c: c.name)
        for c in calibrations:
            if c.name != dataset_pb2.LaserName.TOP:
                continue
            beam_inclinations = tf.constant(c.beam_inclinations)
            # beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            beam_inclinations = list(beam_inclinations.numpy())
            beam_inclinations = [f'{i:e}' for i in beam_inclinations]

        for i in range(5):
            calib_context += 'P' + str(i) + ': ' + \
                ' '.join(camera_calibs[i]) + '\n'

        calib_context += 'TOP' + ': ' + ' '.join(beam_inclinations) + '\n'

        with open(
                f'{self.calib_save_dir}0/{self.prefix}' +
                f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt',
                'w+') as fp_calib:
            fp_calib.write(calib_context)
            fp_calib.close()

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        mmengine.track_parallel_progress(self.convert_one, range(len(self)),
                                         self.workers)
        print('\nFinished ...')

    def convert_one(self, file_idx=0, pathname=None):
        """Convert action for single file.

        Args:
            file_idx (int): Index of the file to be converted.
        """
        if pathname is None:
            pathname = self.tfrecord_pathnames[file_idx]
        dataset = tf.data.TFRecordDataset(pathname, compression_type='')

        for frame_idx, data in enumerate(dataset):

            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            if (self.selected_waymo_locations is not None and
                    frame.context.stats.location
                    not in self.selected_waymo_locations):
                continue
            if frame_idx == 0:
                self.save_calib(frame, file_idx, frame_idx)
            self.save_image(frame, file_idx, frame_idx)
            self.save_cam2world(frame, file_idx, frame_idx)
            self.save_lidar_and_lidar2world(frame, file_idx, frame_idx)

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)

    def save_image(self, frame, file_idx, frame_idx):
        """Parse and save the images in jpg format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        for img in frame.images:
            img_path = f'{self.image_save_dir}{str(img.name - 1)}/' + \
                f'{self.prefix}{str(file_idx).zfill(3)}' + \
                f'{str(frame_idx).zfill(3)}.jpg'
            with open(img_path, 'wb') as fp:
                fp.write(img.image)

    def save_cam2world(self, frame, file_idx, frame_idx):
        """Parse and save the calibration data.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        T_front_cam_to_ref = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0],
                                       [1.0, 0.0, 0.0]])

        # pose: vehicle2world
        pose = np.array(frame.pose.transform).reshape(4, 4)
        for camera in frame.context.camera_calibrations:
            # extrinsic parameters
            cam2vehicle = np.array(camera.extrinsic.transform).reshape(4, 4)
            cam2world = pose @ cam2vehicle @ np.linalg.inv(
                self.cart_to_homo(T_front_cam_to_ref))

            cam2world_path = f'{self.cam2world_save_dir}{str(camera.name - 1)}/' + \
                f'{self.prefix}{str(file_idx).zfill(3)}' + \
                f'{str(frame_idx).zfill(3)}.txt'
            np.savetxt(cam2world_path, cam2world)

    def save_lidar_and_lidar2world(self, frame, file_idx, frame_idx):
        """Parse and save the lidar data in psd format.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        range_images, camera_projections, range_image_top_pose = \
            parse_range_image_and_camera_projection(frame)

        # First return
        self.convert_range_image_to_point_cloud(frame,
                                                file_idx,
                                                frame_idx,
                                                range_images,
                                                camera_projections,
                                                range_image_top_pose,
                                                ri_index=0)

    def create_folder(self):
        """Create folder for data preprocessing."""
        dir_list1 = [self.image_save_dir, self.cam2world_save_dir]
        dir_list2 = [
            self.point_cloud_save_dir, self.lidar2world_save_dir,
            self.calib_save_dir
        ]
        for d in dir_list1:
            for i in range(5):
                mmengine.mkdir_or_exist(f'{d}{str(i)}')
        for d in dir_list2:
            for i in range(1):
                mmengine.mkdir_or_exist(f'{d}{str(i)}')

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

    def convert_range_image_to_point_cloud(self,
                                           frame,
                                           file_idx,
                                           frame_idx,
                                           range_images,
                                           camera_projections,
                                           range_image_top_pose,
                                           ri_index=0):
        """Convert range images to point cloud.

        Args:
            frame (:obj:`Frame`): Open dataset frame.
            range_images (dict): Mapping from laser_name to list of two
                range images corresponding with two returns.
            camera_projections (dict): Mapping from laser_name to list of two
                camera projections corresponding with two returns.
            range_image_top_pose (:obj:`Transform`): Range image pixel pose for
                top lidar.
            ri_index (int, optional): 0 for the first return,
                1 for the second return. Default: 0.

        Returns:
            tuple[list[np.ndarray]]: (List of points with shape [N, 3],
                camera projections of points with shape [N, 6], intensity
                with shape [N, 1], elongation with shape [N, 1], points'
                position in the depth map (element offset if points come from
                the main lidar otherwise -1) with shape[N, 1]). All the
                lists have the length of lidar numbers (5).
        """
        calibrations = sorted(frame.context.laser_calibrations,
                              key=lambda c: c.name)

        frame_pose = tf.convert_to_tensor(
            value=np.reshape(np.array(frame.pose.transform), [4, 4]))
        # [H, W, 6]
        range_image_top_pose_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image_top_pose.data),
            range_image_top_pose.shape.dims)
        # [H, W, 3, 3]
        range_image_top_pose_tensor_rotation = \
            transform_utils.get_rotation_matrix(
                range_image_top_pose_tensor[..., 0],
                range_image_top_pose_tensor[..., 1],
                range_image_top_pose_tensor[..., 2])
        range_image_top_pose_tensor_translation = \
            range_image_top_pose_tensor[..., 3:]
        range_image_top_pose_tensor = transform_utils.get_transform(
            range_image_top_pose_tensor_rotation,
            range_image_top_pose_tensor_translation)
        for c in calibrations:
            if c.name != dataset_pb2.LaserName.TOP:
                continue
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            lidar2world = frame_pose @ extrinsic
            lidar2world_path = f'{self.lidar2world_save_dir}{str(c.name - 1)}/' + \
                f'{self.prefix}{str(file_idx).zfill(3)}' + \
                f'{str(frame_idx).zfill(3)}.txt'
            np.savetxt(lidar2world_path, lidar2world)

            range_image = range_images[c.name][ri_index]
            if len(c.beam_inclinations) == 0:
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant(
                        [c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(value=range_image.data),
                range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0

            if self.filter_no_label_zone_points:
                nlz_mask = range_image_tensor[..., 3] != 1.0  # 1.0: in NLZ
                range_image_mask = range_image_mask & nlz_mask

            range_image_cartesian = \
                range_image_utils.extract_point_cloud_from_range_image(
                    tf.expand_dims(range_image_tensor[..., 0], axis=0),
                    tf.expand_dims(extrinsic, axis=0),
                    tf.expand_dims(tf.convert_to_tensor(
                        value=beam_inclinations), axis=0),
                    pixel_pose=pixel_pose_local,
                    frame_pose=frame_pose_local)

            mask_index = tf.where(range_image_mask)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)

            # in vehicle
            points_vehicle_frame = tf.gather_nd(range_image_cartesian,
                                                mask_index)

            vehicle_to_laser = tf.linalg.inv(
                tf.cast(extrinsic, dtype=tf.float32))
            points_tensor = tf.einsum(
                'ij,kj->ik', points_vehicle_frame,
                vehicle_to_laser[0:3, 0:3]) + vehicle_to_laser[0:3, 3]

            intensity_tensor = tf.gather_nd(range_image_tensor[..., 1],
                                            mask_index)

            elongation_tensor = tf.gather_nd(range_image_tensor[..., 2],
                                             mask_index)

            save_oir_points = True
            if save_oir_points:
                if c.name == 1:
                    mask_index = (
                        ri_index * range_image_mask.shape[0] + mask_index[:, 0]
                    ) * range_image_mask.shape[1] + mask_index[:, 1]
                    mask_index = mask_index.numpy().astype(
                        elongation_tensor.numpy().dtype)
                else:
                    mask_index = np.full_like(elongation_tensor.numpy(), -1)

                point_cloud = np.column_stack(
                    (points_tensor.numpy(), intensity_tensor.numpy(),
                     elongation_tensor.numpy(), mask_index))

                pc_path = f'{self.point_cloud_save_dir}{str(c.name - 1)}/{self.prefix}' + \
                    f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.bin'
                point_cloud.astype(np.float32).tofile(pc_path)


def main():
    load_dir = '/waymo_v120/tfrecord_training'
    save_dir = '/waymo_seq1137/waymo_extract'
    pathname = 'segment-11379226583756500423_6230_810_6250_810_with_camera_labels.tfrecord'
    waymoextractor = WaymoExtractor(load_dir=load_dir,
                                    save_dir=save_dir,
                                    prefix=0)
    waymoextractor.convert_one(pathname=os.path.join(load_dir, pathname))


if __name__ == "__main__":
    main()