import numpy as np
import os


class Constants:
    THRESHOLD_NEW_KEYPOINTS = np.float64(15)
    THRESHOLD_PIXEL_DIST_TRIANGULATION = np.float64(15)
    THRESHOLD_PIXEL_DIST_CANDIDATES_MIN = np.float64(2)
    THRESHOLD_PIXEL_DIST_CANDIDATES_MAX = np.float64(100)
    THRESHOLD_CANDIDATES_ALPHA = np.float64(1)


def get_k_params_imgs(dataset):
    assert dataset in ["kitti", "parking",
                       "malaga"], f'dataset should be in "kitti","parking","malaga", got: {dataset}'
    K = {
        "kitti": np.array([[707.09,  0, 601.88], [0, 707.09, 183.11], [0,  0,  1]], dtype=np.float64),
        "parking": np.array([[331.37,  0, 320], [0, 369.568, 240], [0,  0,  1]], dtype=np.float64),
        "malaga": np.array([[621.184287,  0, 404.00760], [0, 621.18428, 309.05989], [0,  0,  1]], dtype=np.float64)
    }

    malaga_base_path = "data/malaga/malaga-urban-dataset-extract-07_rectified_800x600_Images/"
    imgs = {
        "kitti": [f'data/kitti/05/image_0/{str(i).zfill(6)}.png' for i in range(3)],
        "parking": [f'data/parking/images/img_{str(i).zfill(5)}.png' for i in range(7)],
        "malaga": [malaga_base_path + file for file in os.listdir(malaga_base_path) if "left" in file][:2]
    }

    params = {
        "kitti": {
            "maxCorners": 2000,
            "qualityLevel": 0.01,
            "minDistance": 10,
            "winSize": (21, 21),
            "iterative_params": {
                "turning": {
                    "confidence": 0.99,
                    "reprojection_error": 2.5
                },
                "straight": {
                    "confidence": 0.99,
                    "reprojection_error": 1.15
                }
            }
        },
        "parking": {
            "maxCorners": 1000,
            "qualityLevel": 0.01,
            "minDistance": 10,
            "dist_threshold_move": 0,
            "winSize": (11, 11),
            "RANSAC_threshold": 0.5,
            "repro_threshold": 1.0,
            "iterative_params": {
                "turning": {
                    "confidence": 0.99,
                    "reprojection_error": 2.5
                },
                "straight": {
                    "confidence": 0.99,
                    "reprojection_error": 1.15
                }
            }
        },
        "malaga": {
            "maxCorners": 1000,
            "qualityLevel": 0.01,
            "minDistance": 10,
            "dist_threshold_move": 5,
            "winSize": (31, 31),
            "RANSAC_prob": 0.999,
            "RANSAC_threshold": 0.5,
            "repro_threshold": 5.0,
            "iterative_params": {
                "turning": {
                    "confidence": 0.99,
                    "reprojection_error": 3.0
                },
                "straight": {
                    "confidence": 0.99,
                    "reprojection_error": 1.7
                }
            }
        }
    }
    return K[dataset], params[dataset], imgs[dataset]
