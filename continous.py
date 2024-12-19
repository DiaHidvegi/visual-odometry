import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Tuple
from framestate import FrameState
from scipy.spatial.distance import cdist

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
        new_feature_mask = min_distances > 10  

        # Filter out new features
        new_features = features_current[new_feature_mask].T  

        # Add new features to Ci, Fi, Ti
        if new_features.size > 0:  
            Ci = np.hstack([Ci, new_features])  
            Fi = np.hstack([Fi, new_features])  
            Ti = np.hstack([Ti, np.ones((Ti.shape[0], new_features.shape[1]))]) 

        return Ci, Fi, Ti

    def ransacLocalization(self, matched_query_keypoints, corresponding_landmarks):
        """
        Perform RANSAC-based localization using P3P to estimate pose.

        Args:
            matched_query_keypoints (np.ndarray): 2xN array of 2D keypoints in the image.
            corresponding_landmarks (np.ndarray): Nx3 array of 3D landmarks.
            K (np.ndarray): 3x3 camera intrinsic matrix.

        Returns:
            R_C_W (np.ndarray): 3x3 rotation matrix.
            t_C_W (np.ndarray): 3x1 translation vector.
            best_inlier_mask (np.ndarray): Boolean array of inliers.
        """
    
        num_iterations = 1000
        pixel_tolerance = 10
        k = 3  # Number of points required for P3P

        # Initialize variables
        best_inlier_mask = np.zeros(matched_query_keypoints.shape[1], dtype=bool)
        max_num_inliers = 0

        # Flip keypoints to (u, v) format
        #matched_query_keypoints = np.flip(matched_query_keypoints, axis=0)

        for _ in range(num_iterations):
            # Randomly sample k points
            indices = np.random.permutation(corresponding_landmarks.shape[0])[:k]
            landmark_sample = corresponding_landmarks[indices, :]
            keypoint_sample = matched_query_keypoints[:, indices]

            # Solve P3P
            success, rotation_vectors, translation_vectors = cv2.solveP3P(
                landmark_sample, keypoint_sample.T, self.K, None, flags=cv2.SOLVEPNP_P3P
            )

            if not success:
                continue

            # Evaluate all P3P solutions
            for rvec, tvec in zip(rotation_vectors, translation_vectors):
                R_C_W_guess = cv2.Rodrigues(rvec)[0]
                t_C_W_guess = tvec

                # Project landmarks into the image
                C_landmarks = (
                    np.matmul(R_C_W_guess, corresponding_landmarks.T).T + t_C_W_guess.T
                )
                #projected_points = self.projectPoints(C_landmarks)
                #projected_points, _ = cv2.projectPoints(C_landmarks, np.zeros(3), np.zeros(3), self.K, None)
                projected_points, _ = cv2.projectPoints(corresponding_landmarks.T, rvec, tvec, self.K, None)
                projected_points = projected_points.squeeze()

                # Calculate reprojection errors
                difference = matched_query_keypoints.T - projected_points
                errors = np.linalg.norm(difference, axis=1)
                is_inlier = errors < pixel_tolerance

                # Update the best solution if this one is better
                num_inliers = np.sum(is_inlier)
                if num_inliers > max_num_inliers:
                    max_num_inliers = num_inliers
                    best_inlier_mask = is_inlier
                    best_R_C_W = R_C_W_guess
                    best_t_C_W = t_C_W_guess

        pose = np.eye(4)
        pose[:3, :3] = best_R_C_W.T
        pose[:3, 3] = ((-best_R_C_W.T)@best_t_C_W).flatten()

        return pose, best_inlier_mask
    
    def projectPoints(self, points_3d):
        """
        Projects 3d points to the image plane (3xN), given the camera matrix (3x3) and
        distortion coefficients (4x1).
        """
        # get image coordinates
        projected_points = np.matmul(self.K, points_3d[:, :, None]).squeeze(-1)
        projected_points /= projected_points[:, 2, None]

        return projected_points[:,:2]