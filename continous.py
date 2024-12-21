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

        pose, inlier_mask = self.estimate_pose(Pi, Xi)

        inliers_idx = np.where(inlier_mask == 1)[0]
        Pi = Pi[:, inliers_idx]
        Xi = Xi[:, inliers_idx]

        Pi, Xi, Ci, Fi, Ti = self.handle_candidates(img_current, img_prev, state_prev, Pi, Xi, pose)

        #print(Ci.shape)

        pose = pose[:3,:].reshape((1,-1))

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
        points2D = points2D.T
        points3D = points3D.T
        
        # Initial pose estimation with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points3D, points2D, self.K, None
        )

        if rvec is None or tvec is None or inliers is None:
            raise ValueError("Pose estimation failed.")

        # Refine pose using only inliers
        inlier_mask = np.zeros(points2D.shape[0], dtype=np.uint8)
        inlier_mask[inliers.flatten()] = 1
        
        # Nonlinear refinement using Levenberg-Marquardt
        if inliers.size > 0:
            points3D_refined = points3D[inliers.flatten()]
            points2D_refined = points2D[inliers.flatten()]
            success, rvec_refined, tvec_refined = cv2.solvePnP(
                points3D_refined, 
                points2D_refined, 
                self.K, 
                None,
                rvec,  # Initial rotation guess
                tvec,  # Initial translation guess
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                rvec = rvec_refined
                tvec = tvec_refined

        R_mat = cv2.Rodrigues(rvec)[0]
        pose = np.eye(4)
        pose[:3, :3] = R_mat.T
        pose[:3, 3] = ((-R_mat.T)@tvec).flatten()

        return pose, inlier_mask

    def compute_baseline_angle(self, point3D, t_cur, t_idx):
        baseline = np.linalg.norm(t_cur - t_idx)
        v1 = np.linalg.norm(point3D - t_idx)
        v2 = np.linalg.norm(point3D - t_cur)

        alpha = np.arccos((v1**2 + v2**2 - baseline**2)/(2*v1*v2)) * 180/np.pi
        return alpha

    def handle_candidates(self, img_current: np.ndarray, img_prev: np.ndarray, state_prev: FrameState, Pi: np.ndarray, Xi: np.ndarray, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes feature candidates between frames, updates candidate landmarks, and triangulates new 3D points.

        Args:
        - img_current (np.ndarray): Current image frame (grayscale).
        - img_prev (np.ndarray): Previous image frame (grayscale).
        - state_prev (FrameState): State object containing previous landmarks and candidates.
            - state_prev.cand_landmarks_image_current (np.ndarray): Current candidate landmarks in image coordinates.
            - state_prev.cand_landmarks_image_first (np.ndarray): First observed candidate landmarks in image coordinates.
            - state_prev.cand_landmarks_transform (np.ndarray): Transformations associated with candidate landmarks.
        - Pi (np.ndarray): Existing tracked landmarks in current frame in image coordinates (shape: 2 x N).
        - Xi (np.ndarray): Existing tracked landmarks in current frame in 3D world coordinates (shape: 3 x N).
        - pose (np.ndarray): Current camera pose as a 3x4 transformation matrix.

        Returns:
        - Tuple [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - Pi: Updated tracked landmarks in image coordinates for current frame.
            - Xi: Updated tracked landmarks in 3D world coordinates for current frame.
            - Ci: Remaining candidate landmarks in image coordinates.
            - Fi: Corresponding first observed candidate landmarks in image coordinates.
            - Ti: Transformations (TCW not TWC atm.) associated with candidate landmarks from first appearance.
        """

        Ci = state_prev.cand_landmarks_image_current
        Fi = state_prev.cand_landmarks_image_first
        Ti = state_prev.cand_landmarks_transform

        # Detect new features in the current frame
        # TODO: Use self.params instead of hardcoding the parameters
        features_current = cv2.goodFeaturesToTrack(img_current, maxCorners=1000, qualityLevel=0.01, minDistance=10)
        features_current = np.float32(features_current).reshape(-1, 2) 

        # Track features from the previous frame
        tracked_features, status, _ = self.track_keypoints(Ci, img_prev, img_current)
        valid_idx = np.where(status == 1)[0]

        # Update Ci, Fi, Ti with tracked features
        Ci = tracked_features[:, valid_idx]
        Fi = Fi[:, valid_idx]
        Ti = Ti[:, valid_idx]

        # Filter points that didn't move more pixels than a certain threshold (e.g. close to epipole)
        distances = np.sqrt(np.sum((Ci - Fi) ** 2, axis=0))
        good_candidates = (distances > Constants.THRESHOLD_PIXEL_DIST_CANDIDATES_MIN) & (distances < Constants.THRESHOLD_PIXEL_DIST_CANDIDATES_MAX)

        Ci = Ci[:, good_candidates]
        Fi = Fi[:, good_candidates]
        Ti = Ti[:, good_candidates]

        # Combine existing points (tracked_features and Pi)
        existing_features = np.vstack([Ci.T, Pi.T])

        # Compute pairwise distances and filter out duplicates
        distance_matrix = cdist(features_current, existing_features)
        min_distances = np.min(distance_matrix, axis=1) 
        new_feature_mask = min_distances > Constants.THRESHOLD_NEW_KEYPOINTS

        # Filter out new features
        new_features = features_current[new_feature_mask].T

        #Verify landmarks, candidates and new_candidates in a plot
        show_plots = False

        if show_plots:
            img_now = img_current.copy()
            # Draw Ci (white)
            for pt in range(Ci.shape[1]):
                x, y = Ci[0, pt], Ci[1, pt]
                cv2.circle(img_now, (int(x), int(y)), 3, 255, -1)  # White

            # Draw Pi (grey)
            for pt in range(Pi.shape[1]):
                x, y = Pi[0, pt], Pi[1, pt]
                cv2.circle(img_now, (int(x), int(y)), 3, 128, -1)  # Grey (128)

            # Draw new_features (black)
            for pt in range(new_features.shape[1]):
                x, y = new_features[0, pt], new_features[1, pt]
                cv2.circle(img_now, (int(x), int(y)), 3, 0, -1)  # Black

            # Display the image with the updated features
            cv2.imshow("Features in Greyscale (Ci=White, Pi=Grey, New=Black)", img_now)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        t_cur = pose[:3,3]

        distance_matrix = np.sqrt(np.sum((Ci - Fi) ** 2, axis=0))

        new_landmarks = distance_matrix > Constants.THRESHOLD_PIXEL_DIST_TRIANGULATION
        
        if new_landmarks.sum() > 0:
            points3D = np.empty((3, 0))
            for idx in range(new_landmarks.shape[0]):
                if not new_landmarks[idx]:
                    continue
                # Triangulate new points
                # Create projection matrices for the first and second views
                Ti_idx_formatted = np.reshape(Ti[:, idx], (3, 4))
                proj1 = self.K @ self.invert_transformation(Ti_idx_formatted)
                proj2 = self.K @ self.invert_transformation(pose[:3,:])

                point4D_hom = cv2.triangulatePoints(proj1, proj2, Fi[:, idx], Ci[:, idx])
                point3D = point4D_hom[:3] / point4D_hom[3]

                t_idx = Ti_idx_formatted[:3,3]

                # Check for negative depth
                if point3D[2] < -t_idx[2]:
                    new_landmarks[idx] = False
                    continue

                # Calculate the baseline angle
                alpha = self.compute_baseline_angle(point3D, t_cur, t_idx)

                if alpha > Constants.THRESHOLD_CANDIDATES_ALPHA:
                    points3D = np.hstack([points3D, point3D])
                else:
                    new_landmarks[idx] = False


            Pi = np.hstack([Pi, Ci[:, new_landmarks]])
            Xi = np.hstack([Xi, points3D])
            no_new_landmark = np.where(new_landmarks == 0)[0]
            Ci = Ci[:, no_new_landmark]
            Fi = Fi[:, no_new_landmark]
            Ti = Ti[:, no_new_landmark]

        # Add new features to Ci, Fi, Ti
        if new_features.size > 0:  
            pose = pose[:3,:].reshape((-1,1))
            tiled_pose = np.tile(pose, (1, new_features.shape[1]))
            Ci = np.hstack([Ci, new_features])  
            Fi = np.hstack([Fi, new_features])  
            Ti = np.hstack([Ti, tiled_pose])
        

        return Pi, Xi, Ci, Fi, Ti
    
    def invert_transformation(self, T_3x4: np.ndarray) -> np.ndarray:
        """
        Inverts a 3x4 transformation matrix.

        Args:
        - T_3x4 (np.ndarray): Input 3x4 transformation matrix.

        Returns:
        - np.ndarray: Inverted 3x4 transformation matrix.
        """
        # Convert to 4x4 matrix
        T_4x4 = np.eye(4)
        T_4x4[:3, :4] = T_3x4

        # Invert the 4x4 matrix
        R = T_4x4[:3, :3]
        t = T_4x4[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t

        T_4x4_inv = np.eye(4)
        T_4x4_inv[:3, :3] = R_inv
        T_4x4_inv[:3, 3] = t_inv

        # Return the inverted 3x4 matrix
        return T_4x4_inv[:3, :4]
