import cv2
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
                        
class Initialization:
    def __init__(self, img1, img2, K, show_plots=False):
        # Load the images
        self.img1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
        self.img2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

        # store camera intrinsics
        self.K = K

        # compute and store point correspondences
        self.points3D, self.points2D = self.get_features_and_k(self.img1, self.img2, K, show_plots)

    def get_features_and_k(self, img1, img2, K, show_plots):
        # Step 1: Detect keypoints in the first image using Shi-Tomasi corner detector
        corners = cv2.goodFeaturesToTrack(img1, maxCorners=200, qualityLevel=0.01, minDistance=10)

        # Step 2: Track keypoints in the last image using optical flow
        # Note: maxLevel=0 reduces quality but makes sure that no pyramids are used (not implemented in exercise)
        lk_params = dict(winSize=(11, 11), maxLevel=0, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        corners = np.float32(corners).reshape(-1, 1, 2)
        tracked_points, status, _ = cv2.calcOpticalFlowPyrLK(img1, img2, corners, None, **lk_params)

        # Keep only good points
        good_old = corners[status.flatten() == 1]
        good_new = tracked_points[status.flatten() == 1]

        # Step 3: Estimate the Essential Matrix using RANSAC
        E, mask = cv2.findEssentialMat(good_new, good_old, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        good_old = good_old[mask.flatten() == 1]
        good_new = good_new[mask.flatten() == 1]

        # Recover pose from Essential Matrix
        _, R, t, _ = cv2.recoverPose(E, good_new, good_old)

        # Step 4: Triangulate 3D points
        # Create projection matrices for the first and second views
        proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Projection matrix for the first view
        proj2 = np.hstack((R, t))  # Projection matrix for the second view

        # Convert points to homogeneous format for triangulation
        points4D_hom = cv2.triangulatePoints(proj1, proj2, good_old.T, good_new.T)
        points3D = points4D_hom[:3] / points4D_hom[3]  # Convert from homogeneous to 3D

        if show_plots:
            # Optional: Visualize tracked points
            img_old = img1.copy()
            for pt in good_old:
                x, y = pt.ravel()
                cv2.circle(img_old, (int(x), int(y)), 3, (255, 255, 125), -1)

            img_tracked = img2.copy()
            for pt in good_new:
                x, y = pt.ravel()
                cv2.circle(img_tracked, (int(x), int(y)), 3, (255, 255, 125), -1)

            cv2.imshow("Initial Points", img_old)
            cv2.imshow("Tracked Points", img_tracked)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return points3D.reshape(3, -1), good_old.reshape(2, -1)
    
if __name__ == "__main__":
    K = np.array([[707.09,  0, 601.88],
                  [ 0, 707.09, 183.11],
                  [ 0,  0,  1]])
    initialization = Initialization(
        'data/kitti/05/image_0/000000.png',
        'data/kitti/05/image_0/000002.png', K, True)
    
    print(f"Number of points {len(initialization.points2D)}")