import cv2
import numpy as np
from modules.draw import Plotter3d, draw_poses
from modules.inference_engine_pytorch import InferenceEnginePyTorch
from modules.parse_poses import parse_poses


def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d


class PoseService:
    base_height = 256
    fx = -1
    delay = 1
    esc_code = 27
    p_code = 112
    space_code = 32
    mean_time = 0
    stride = 8
    net = InferenceEnginePyTorch('./model.pth', 'GPU')
    R = np.array([
        [
            0.1656794936,
            0.0336560618,
            -0.9856051821
        ],
        [
            -0.09224101321,
            0.9955650135,
            0.01849052095
        ],
        [
            0.9818563545,
            0.08784972047,
            0.1680491765
        ]
    ], dtype=np.float32)
    T = np.array([[17.76193366], [126.741365], [286.3860507]], dtype=np.float32)

    def __init__(self, image):
        self.image = image
        pass

    def get_pose(self):
        if self.image is None:
            return []

        input_scale = PoseService.base_height / self.image.shape[0]
        scaled_img = cv2.resize(self.image, dsize=None, fx=input_scale, fy=input_scale)
        scaled_img = scaled_img[:,
                     0:scaled_img.shape[1] - (
                             scaled_img.shape[1] % PoseService.stride)]  # better to pad, but cut out for demo

        PoseService.fx = np.float32(0.8 * self.image.shape[1])

        inference_result = PoseService.net.infer(scaled_img)
        poses_3d, poses_2d = parse_poses(inference_result, input_scale, PoseService.stride, PoseService.fx, False)

        if len(poses_3d):
            poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]

        # draw_poses(self.image, poses_2d)

        # cv2.imwrite('2d.png', self.image)

        if poses_3d.size == 0 or poses_2d.size == 0:
            return {"pose2D": [], "pose3D": []}

        return {"pose2D": poses_2d[0].flatten().tolist(), "pose3D": poses_3d[0].flatten().tolist()}
