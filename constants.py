import numpy as np
import os
import random
import cv2
# Set all seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
cv2.setRNGSeed(SEED)


class Constants:
    THRESHOLD_NEW_KEYPOINTS = np.int64(15)
    THRESHOLD_PIXEL_DIST_TRIANGULATION = np.int64(15)
    THRESHOLD_PIXEL_DIST_CANDIDATES_MIN = np.int64(2)
    THRESHOLD_PIXEL_DIST_CANDIDATES_MAX = np.int64(100)
    THRESHOLD_CANDIDATES_ALPHA = np.int64(1)


def get_k_params_imgs(dataset):
    assert dataset in ["kitti", "parking",
                       "malaga"], f'dataset should be in "kitti","parking","malaga", got: {dataset}'
    K = {
        "kitti": np.array([[707.09,  0, 601.88], [0, 707.09, 183.11], [0,  0,  1]], dtype=np.float32),
        "parking": np.array([[331.37,  0, 320], [0, 369.568, 240], [0,  0,  1]], dtype=np.float32),
        "malaga": np.array([[621.184287,  0, 404.00760], [0, 621.18428, 309.05989], [0,  0,  1]], dtype=np.float32)
    }

    malaga_base_path = "data/malaga/malaga-urban-dataset-extract-07_rectified_800x600_Images/"
    imgs = {
        "kitti": [f'data/kitti/05/image_0/{str(i).zfill(6)}.png' for i in range(3)],
        "parking": [f'data/parking/images/img_{str(i).zfill(5)}.png' for i in range(7)],
        "malaga": [malaga_base_path + file for file in os.listdir(malaga_base_path) if "left" in file][:2]
    }

    params = {
        "kitti": {
            "maxCorners": np.int64(2000),
            "qualityLevel": np.float32(0.01),
            "minDistance": np.float32(10),
            "winSize": (np.int64(21), np.int64(21)),
            "iterative_params": {
                "turning": {
                    "confidence": np.float32(0.99),
                    "reprojection_error": np.float32(2.5)
                },
                "straight": {
                    "confidence": np.float32(0.99),
                    "reprojection_error": np.float32(1.15)
                }
            }
        },
        "parking": {
            "maxCorners": np.int64(1000),
            "qualityLevel": np.float32(0.01),
            "minDistance": np.float32(10),
            "dist_threshold_move": np.int64(0),
            "winSize": (np.int64(11), np.int64(11)),
            "RANSAC_threshold": np.float32(0.5),
            "repro_threshold": np.float32(1.0),
            "iterative_params": {
                "turning": {
                    "confidence": np.float32(0.99),
                    "reprojection_error": np.float32(2.5)
                },
                "straight": {
                    "confidence": np.float32(0.99),
                    "reprojection_error": np.float32(1.15)
                }
            }
        },
        "malaga": {
            "maxCorners": np.int64(1000),
            "qualityLevel": np.float32(0.01),
            "minDistance": np.float32(10),
            "dist_threshold_move": np.int64(5),
            "winSize": (np.int64(21), np.int64(21)),
            "RANSAC_prob": np.float32(0.999),
            "RANSAC_threshold": np.float32(0.5),
            "repro_threshold": np.float32(5.0),
            "iterative_params": {
                "turning": {
                    "confidence": np.float32(0.99),
                    "reprojection_error": np.float32(3.0)
                },
                "straight": {
                    "confidence": np.float32(0.99),
                    "reprojection_error": np.float32(1.7)
                }
            }
        }
    }
    return K[dataset], params[dataset], imgs[dataset]
