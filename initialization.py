import random
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set all seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

cv2.setRNGSeed(SEED)


class Initialization:
    def __init__(self, dataset, show_plots=False):
        # extract parameters
        K, params, imgs = self.get_k_params_imgs(dataset)

        # Load the images
        self.imgs = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in imgs]

        # store camera intrinsics
        self.K = K

        # compute and store point correspondences
        self.points3D, self.points2D = self.get_features_and_k(
            self.imgs, K, params)

        # plot if desired
        if show_plots:
            self.make_plots()

    def get_initial_landmarks(self):
        return self.points3D, self.points2D

    def get_features_and_k(self, imgs, K, params):
        # Step 1: Detect keypoints in the first image using Shi-Tomasi corner detector
        corners = cv2.goodFeaturesToTrack(
            imgs[0], maxCorners=params["maxCorners"], qualityLevel=params["qualityLevel"], minDistance=params["minDistance"]
        ).astype(np.float32)
        good_old = np.float32(corners).reshape(-1, 1, 2)
        good_new = good_old

        # Step 2: Track keypoints, remove far-away features + outliers, estimate essential matrix
        for i in range(len(imgs)-1):
            # Step 2.1: Track keypoints from first to last image using optical flow
            # Note: maxLevel=0 reduces quality but makes sure that no pyramids are used (not implemented in exercise)
            lk_params = dict(winSize=params["winSize"], maxLevel=0, criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            tracked_points, status, _ = cv2.calcOpticalFlowPyrLK(
                imgs[i], imgs[i+1], good_new, None, **lk_params)

            # Keep only good points
            good_old = good_old[status.flatten() == 1]
            good_new = tracked_points[status.flatten() == 1]

            # Step 2.2: filter points that didn't move more pixels than a certain threshold (e.g. close to epipole)
            # distances = np.linalg.norm(good_new - good_old, axis=2)
            distances = np.sqrt(
                np.sum((good_new - good_old)**2, axis=2)).flatten().astype(np.float32)
            drop = np.where(distances < params["dist_threshold_move"])
            good_old = np.delete(good_old, drop, axis=0)
            good_new = np.delete(good_new, drop, axis=0)

            # Step 2.3: Estimate the Essential Matrix using RANSAC
            E, mask = cv2.findEssentialMat(good_new, good_old, cameraMatrix=K, method=cv2.RANSAC,
                                           prob=params["RANSAC_prob"], threshold=params["RANSAC_threshold"])
            good_old = good_old[mask.flatten() == 1]
            good_new = good_new[mask.flatten() == 1]

        # Recover pose from Essential Matrix
        _, R, t, _ = cv2.recoverPose(E, good_old, good_new, K)
        print("projection matrix from last position to first position")
        print(np.hstack((R, t)))

        # Step 3: Triangulate 3D points
        # Create projection matrices for the first and second views
        # Projection matrix for the first view
        proj1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        proj2 = K @ np.hstack((R, t))  # Projection matrix for the second view

        # Convert points to homogeneous format for triangulation
        points4D_hom = cv2.triangulatePoints(proj1, proj2, good_old, good_new)
        # Convert from homogeneous to 3D
        points3D = points4D_hom[:3] / points4D_hom[3]

        # Step 4: Use reprojection error to remove bad points
        # Back to homogeneous coordinates
        points3D_hom = np.vstack((points3D, np.ones(points3D.shape[1])))
        reprojected1 = proj1 @ points3D_hom
        reprojected2 = proj2 @ points3D_hom

        # Normalize reprojected points
        reprojected1 /= reprojected1[2]
        reprojected2 /= reprojected2[2]

        # Compute reprojection error
        error1 = np.linalg.norm(good_old.squeeze(
            axis=1).T - reprojected1[:2], axis=0)
        error2 = np.linalg.norm(good_new.squeeze(
            axis=1).T - reprojected2[:2], axis=0)

        # Set threshold to flag outliers
        threshold = params["repro_threshold"]  # in pixels
        outliers = np.where((error1 > threshold) | (
            error2 > threshold) | (points3D[2] < 0))

        points3D = np.delete(points3D, outliers, axis=1)
        good_old = np.delete(good_old, outliers, axis=0)
        good_new = np.delete(good_new, outliers, axis=0)

        # store data for plotting
        self._plt_good_old = good_old
        self._plt_good_new = good_new
        self._plt_points3D = points3D

        print(f"Number of points {points3D.shape[1]}")

        return points3D.reshape(3, -1), good_old.squeeze(axis=1).T

    def make_plots(self):
        # Visualize tracked points
        img_old = self.imgs[0].copy()
        for pt in self._plt_good_old:
            x, y = pt.ravel()
            cv2.circle(img_old, (int(x), int(y)), 3, (255, 255, 125), -1)

        img_tracked = self.imgs[-1].copy()
        for pt_old, pt_new in zip(self._plt_good_old, self._plt_good_new):
            x_old, y_old = pt_old.ravel()
            x_new, y_new = pt_new.ravel()

            # Draw the current point in img_tracked
            cv2.circle(img_tracked, (int(x_new), int(y_new)),
                       3, (255, 255, 125), -1)

            # Draw a line from the old position to the new position
            cv2.line(img_tracked, (int(x_old), int(y_old)),
                     (int(x_new), int(y_new)), (0, 255, 0), 1)

        cv2.imshow("Initial Points", img_old)
        cv2.imshow("Tracked Points", img_tracked)

        # Create a 3D scatter plot for the triangulated points
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)

        # Scatter the 3D points
        ax.scatter(self._plt_points3D[0], self._plt_points3D[2],
                   c='blue', marker='o', alpha=0.7, label='Landmarks')

        # Set labels and title
        ax.set_title("3D Landmarks (Camera Coordinate System)")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.legend()

        # Display the plot
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_landmarks(self, filename):
        pts = np.hstack((self.points3D.T, self.points2D.T))
        # 5 columns in file: X,Y,Z,u,w
        np.savetxt(filename, pts, fmt="%.6f", delimiter=",")

    def get_k_params_imgs(self, dataset):
        assert dataset in [
            "kitti", "parking", "malaga"], f'dataset should be in "kitti","parking","malaga", got: {dataset}'
        K = {
            "kitti": np.array([[707.09,  0, 601.88], [0, 707.09, 183.11], [0,  0,  1]], dtype=np.float32),
            "parking": np.array([[331.37,  0, 320], [0, 369.568, 240], [0,  0,  1]], dtype=np.float32),
            "malaga": np.array([[621.184287,  0, 404.00760], [0, 621.18428, 309.05989], [0,  0,  1]], dtype=np.float32)
        }

        malaga_base_path = "data/malaga/malaga-urban-dataset-extract-07_rectified_800x600_Images/"
        imgs = {
            "kitti": [f'data/kitti/05/image_0/{str(i).zfill(6)}.png' for i in range(3)],
            "parking": [f'data/parking/images/img_{str(i).zfill(5)}.png' for i in range(7)],
            "malaga": sorted([malaga_base_path + file for file in os.listdir(malaga_base_path) if "left" in file])[:5]
        }

        params = {
            "kitti":   {"maxCorners": 1000, "qualityLevel": np.float32(0.01), "minDistance": np.float32(10), "dist_threshold_move": np.float32(2), "winSize": (11, 11), "RANSAC_prob": np.float32(0.999), "RANSAC_threshold": np.float32(0.5), "repro_threshold": np.float32(3.0)},
            "parking": {"maxCorners": 1000, "qualityLevel": np.float32(0.01), "minDistance": np.float32(10), "dist_threshold_move": np.float32(0), "winSize": (11, 11), "RANSAC_prob": np.float32(0.999), "RANSAC_threshold": np.float32(0.5), "repro_threshold": np.float32(1.0)},
            "malaga":  {"maxCorners": 400, "qualityLevel": np.float32(0.01), "minDistance": np.float32(10), "dist_threshold_move": np.float32(0), "winSize": (41, 41), "RANSAC_prob": np.float32(0.999), "RANSAC_threshold": np.float32(1.5), "repro_threshold": np.float32(3.0)}
            # "malaga":  {"maxCorners": 400, "qualityLevel":0.01, "minDistance":10, "dist_threshold_move":0, "winSize":(41, 41), "RANSAC_prob":0.999, "RANSAC_threshold":1.5, "repro_threshold":5.0} # on 5 frames
            # "malaga":  {"maxCorners": 400, "qualityLevel":0.01, "minDistance":10, "dist_threshold_move":0, "winSize":(41, 41), "RANSAC_prob":0.999, "RANSAC_threshold":1.5, "repro_threshold":3.0} # on 5 frames
        }
        return K[dataset], params[dataset], imgs[dataset]


if __name__ == "__main__":
    data_choice = "malaga"
    initialization = Initialization(data_choice, True)
    points3D, points2D = initialization.get_initial_landmarks()
    # initialization.save_landmarks("landmarks/malaga.txt")
    print(f"Got points with sizes {points3D.shape} and {points2D.shape}")
