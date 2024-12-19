import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple
from framestate import FrameState

from utils import Constants, get_k_params_imgs
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class ContinuousVO:
    def __init__(self, K: np.ndarray, datachoice: str):
        """
        Initializes the Continuous Visual Odometry pipeline.
        
        Args:
            K (np.ndarray): Intrinsic camera matrix.
        """
        self.K = K
        _, self.params, _ = get_k_params_imgs(datachoice)

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

        Ci, Fi , Ti = self.handle_candidates(img_current, img_prev, state_prev, Pi)

        new_state = FrameState(
            landmarks_image=Pi,
            landmarks_world=Xi,
            cand_landmarks_image_current=Ci,
            cand_landmarks_image_first=Fi,
            cand_landmarks_transform=Ti
        )

        return new_state, pose  

    def track_keypoints(self, prev_keypoints: np.ndarray, img_prev: np.ndarray, img_current: np.ndarray):
        """
        Track keypoints between two frames using Lucas-Kanade optical flow.

        Args:
            prev_keypoints (np.ndarray): Previous keypoints to track (2xN array of [x, y] coordinates).
            img_prev (np.ndarray): Previous frame (grayscale image).
            img_current (np.ndarray): Current frame (grayscale image).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Tracked points in the current frame (2xN array).
                - Status array indicating tracking success (1 if successful, 0 otherwise).
                - Error array for each tracked point.
        """
        if prev_keypoints.size == 0:
            # No keypoints to track
            return np.empty((2, 0)), np.empty((0,), dtype=np.uint8), np.empty((0,), dtype=np.float32)

        # Convert keypoints to the correct shape for OpenCV (Nx1x2)
        prev_keypoints_cv = prev_keypoints.T.reshape(-1, 1, 2).astype(np.float32)

        lk_params = dict(winSize=(11, 11), maxLevel=0, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        tracked_points, status, error = cv2.calcOpticalFlowPyrLK(img_prev, img_current, prev_keypoints_cv, None, **lk_params)

        # Reshape tracked points back to 2xN
        tracked_points = tracked_points.reshape(-1, 2).T
        status = status.flatten()
        error = error.flatten()

        return tracked_points, status, error


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
        pose[:3, 3] = ((-R_mat.T)@tvec).flatten()

        inlier_mask = np.zeros(points2D.shape[0], dtype=np.uint8)
        inlier_mask[inliers.flatten()] = 1

        return pose, inlier_mask
    

    def handle_candidates(self, img_current, img_prev, state_prev, Pi):

        Ci = state_prev.cand_landmarks_image_current
        Fi = state_prev.cand_landmarks_image_first
        Ti = state_prev.cand_landmarks_transform

        # Detect new features in the current frame
        features_current = cv2.goodFeaturesToTrack(img_current, maxCorners=1000, qualityLevel=0.01, minDistance=10)
        features_current = np.float32(features_current).reshape(-1, 2) 

        # Track features from the previous frame
        tracked_features, status, _ = self.track_keypoints(Ci, img_prev, img_current)
        valid_idx = np.where(status == 1)[0]
        tracked_features = tracked_features[:, valid_idx]

        # Update Ci, Fi, Ti with tracked features
        Ci = tracked_features
        Fi = Fi[:, valid_idx]
        Ti = Ti[:, valid_idx]

        # Combine existing points (tracked_features and Pi)
        existing_features = np.vstack([tracked_features.T, Pi.T])

        # Compute pairwise distances and filter out duplicates
        distance_matrix = cdist(features_current, existing_features)
        min_distances = np.min(distance_matrix, axis=1) 
        new_feature_mask = min_distances > Constants.THRESHOLD_NEW_KEYPOINTS

        # Filter out new features
        new_features = features_current[new_feature_mask].T  

        # Add new features to Ci, Fi, Ti
        if new_features.size > 0:  
            Ci = np.hstack([Ci, new_features])  
            Fi = np.hstack([Fi, new_features])  
            Ti = np.hstack([Ti, np.ones((Ti.shape[0], new_features.shape[1]))]) 

        return Ci, Fi, Ti