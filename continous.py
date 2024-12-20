import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple
from framestate import FrameState

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class ContinuousVO:
    def __init__(self, K: np.ndarray):
        """
        Initializes the Continuous Visual Odometry pipeline.
        
        Args:
            K (np.ndarray): Intrinsic camera matrix.
        """
        self.K = K
        #self.point_tracker = cv2.TrackerKLT_create()

    def process_frame(self, img_current: np.ndarray, img_prev: np.ndarray, state_prev: FrameState) -> Tuple[FrameState, np.ndarray]:
        """
        Process the current frame, update the frame state, and estimate the current pose.

        Args:
            img_current (np.ndarray): Current frame (grayscale image).
            img_prev (np.ndarray): Previous frame (grayscale image).
            state_prev (FrameState): Previous frame state.

        Returns:
            Tuple[FrameState, np.ndarray]: Updated frame state and the current pose as a 4x4 transformation matrix.
        """
        
        show_plots = False
        
        tracked_points, status, _ = self.track_keypoints(state_prev.landmarks_image, img_prev, img_current)
        
        valid_idx = np.where(status == 1)[0]
        Pi = tracked_points[:, valid_idx]
        Xi = state_prev.landmarks_world[:, valid_idx]
        

        if show_plots:
            # Optional: Visualize tracked points
            img_old = img_prev.copy()
            Pi_prev = state_prev.landmarks_image[:, valid_idx]
            for pt in range(Pi_prev.shape[1]):
                x, y = Pi_prev[0,pt], Pi_prev[1,pt]
                cv2.circle(img_old, (int(x), int(y)), 3, (255, 255, 125), -1)

            img_tracked = img_current.copy()
            for pt in range(Pi_prev.shape[1]):
                x, y = Pi[0,pt], Pi[1,pt]
                cv2.circle(img_tracked, (int(x), int(y)), 3, (255, 255, 125), -1)

            cv2.imshow("Initial Points", img_old)
            cv2.imshow("Tracked Points", img_tracked)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        #pose, inlier_mask = self.ransacLocalization(Pi, Xi.T)
        pose, inlier_mask = self.estimate_pose(Pi, Xi.T)

        pose = pose[:3,:].reshape((1,-1))

        inliers_idx = np.where(inlier_mask == 1)[0]
        Pi = Pi[:, inliers_idx]
        Xi = Xi[:, inliers_idx]

        # Triangulate new landmarks
        state_prev = self.triangulate_new_landmarks(state_prev, img_current, pose)

        new_state = FrameState(
            landmarks_image=Pi,
            landmarks_world=Xi,
            cand_landmarks_image_current=state_prev.cand_landmarks_image_current,
            cand_landmarks_image_first=state_prev.cand_landmarks_image_first,
            cand_landmarks_transform=state_prev.cand_landmarks_transform
        )

        return new_state, pose

    def track_keypoints(self, points: np.ndarray, img_prev: np.ndarray, img_current: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tracks keypoints from the previous frame to the current frame using KLT.

        Args:
            points (np.ndarray): Keypoints in the previous frame (shape 2xK).
            img_prev (np.ndarray): Previous frame (grayscale image).
            img_current (np.ndarray): Current frame (grayscale image).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                Tracked keypoints (2xK), status array, and error array.
        """
        points = points.T.astype(np.float32)
        lk_params = dict(winSize=(11, 11), maxLevel=0, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        points_tracked, status, error = cv2.calcOpticalFlowPyrLK(img_prev, img_current, points, None, **lk_params)
        return points_tracked.T, status.ravel(), error.ravel()

    def estimate_pose(self, points2D: np.ndarray, points3D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimates the camera pose using P3P and RANSAC.

        Args:
            points2D (np.ndarray): 2D keypoints in the current frame (shape 2xK).
            points3D (np.ndarray): Corresponding 3D landmarks (shape 3xK).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                Pose as a 4x4 transformation matrix and inlier mask.
        """
        #points2D = np.flip(points2D, axis=0)
        points2D = points2D.T
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points3D, points2D, self.K, None
        )

        if rvec is None or tvec is None or inliers is None:
            raise ValueError("Pose estimation failed.")

        R_mat = cv2.Rodrigues(rvec)[0]
        pose = np.eye(4)
        pose[:3, :3] = R_mat.T
        pose[:3, 3] = ((-R_mat)@tvec).flatten()

        inlier_mask = np.zeros(points2D.shape[0], dtype=np.uint8)
        inlier_mask[inliers.flatten()] = 1

        return pose, inlier_mask

    def compute_baseline_angle(self, first_keypoint, current_keypoint, first_pose, current_pose):
        """
        Compute the baseline angle between two keypoints and their associated poses.
            Args:
                first_keypoint: Keypoint in the first frame.
                current_keypoint: Keypoint in the current frame.
                first_pose: Pose of the camera at the first frame.
                current_pose: Pose of the camera at the current frame.
        """
        first_bearing = np.linalg.inv(self.K) @ np.append(first_keypoint, 1)
        current_bearing = np.linalg.inv(self.K) @ np.append(current_keypoint, 1)

        first_bearing_world = first_pose[:3, :3] @ first_bearing
        current_bearing_world = current_pose[:3, :3] @ current_bearing
        cosine_angle = np.dot(first_bearing_world, current_bearing_world) / (
            np.linalg.norm(first_bearing_world) * np.linalg.norm(current_bearing_world)
        )
        return np.arccos(np.clip(cosine_angle, -1, 1))

    # def triangulate(self, first_pose, current_pose, first_keypoint, current_keypoint):
    #     """
    #     Triangulate a 3D point from two keypoints and their associated poses.
        
    #         Args:
    #             first_pose: Pose of the camera at the first frame.
    #             current_pose: Pose of the camera at the current frame.
    #             first_keypoint: Keypoint in the first frame.
    #             current_keypoint: Keypoint in the current frame.
    #     """
    #     proj_matrix1 = self.K @ first_pose[:3, :]
    #     proj_matrix2 = self.K @ current_pose[:3, :]
    #     points_4d = cv2.triangulatePoints(proj_matrix1, proj_matrix2, first_keypoint, current_keypoint)

    #     return (points_4d[:3] / points_4d[3]).reshape(3, 1)

    # def triangulate_new_landmarks(self, state_prev: FrameState, img_current: np.ndarray, pose: np.ndarray) -> FrameState:
    #     """
    #     Triangulates new landmarks using candidate keypoints.

    #     Args:
    #         state_prev (FrameState): Previous frame state.
    #         img_current (np.ndarray): Current frame (grayscale image).
    #         pose (np.ndarray): Current camera pose (4x4 transformation matrix).

    #     Returns:
    #         FrameState: Updated frame state with new landmarks.
    #     """
    #     new_candidates = []
    #     for i in range(state_prev.cand_landmarks_image_current.shape[1]):
    #         c_first = state_prev.cand_landmarks_image_first[:, i]
    #         c_current = state_prev.cand_landmarks_image_current[:, i]

    #         T_first = state_prev.cand_landmarks_transform[:, i].reshape(4, 4)
    #         T_current = pose

    #         angle = self.compute_baseline_angle(c_first, c_current, T_first, T_current)

    #         if angle > np.radians(1):
    #             landmark_3D = self.triangulate(T_first, T_current, c_first, c_current)
    #             state_prev.landmarks_world = np.hstack((state_prev.landmarks_world, landmark_3D))
    #             state_prev.landmarks_image = np.hstack((state_prev.landmarks_image, c_current.reshape(2, 1)))

    #             new_candidates.append(i)

    #     state_prev.cand_landmarks_image_current = np.delete(state_prev.cand_landmarks_image_current, new_candidates, axis=1)
    #     state_prev.cand_landmarks_image_first = np.delete(state_prev.cand_landmarks_image_first, new_candidates, axis=1)
    #     state_prev.cand_landmarks_transform = np.delete(state_prev.cand_landmarks_transform, new_candidates, axis=1)

    #     return state_prev
