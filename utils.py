import numpy as np

class Constants:
    THRESHOLD_NEW_KEYPOINTS = 5 # given in pixels 
    THRESHOLD_PIXEL_DIST_TRIANGULATION = 10 # given in pixels
    THRESHOLD_PIXEL_DIST_CANDIDATES_MIN = 2 # given in pixels
    THRESHOLD_PIXEL_DIST_CANDIDATES_MAX = 100 # given in pixels
    THRESHOLD_CANDIDATES_ALPHA = 1

def get_k_params_imgs(dataset):
    assert dataset in ["kitti","parking","malaga"], f'dataset should be in "kitti","parking","malaga", got: {dataset}'
    K = {
        "kitti": np.array([[707.09,  0, 601.88], [ 0, 707.09, 183.11], [ 0,  0,  1]]),
        "parking": np.array([[331.37,  0, 320], [ 0, 369.568, 240], [ 0,  0,  1]]),
        "malaga": np.array([[621.184287,  0, 404.00760], [ 0, 621.18428, 309.05989], [ 0,  0,  1]])
    }

    malaga_base_bath = "data/malaga/malaga-urban-dataset-extract-07_rectified_800x600_Images/"
    imgs = {
        "kitti": [f'data/kitti/05/image_0/{str(i).zfill(6)}.png' for i in range(3)],
        "parking": [f'data/parking/images/img_{str(i).zfill(5)}.png' for i in range(7)],
        #"malaga": [malaga_base_bath + file for file in os.listdir(malaga_base_bath) if "left" in file][:2]
    }
    
    params = {
        "kitti":   {"maxCorners":1000, "qualityLevel":0.01, "minDistance":10, "dist_threshold_move":5, "winSize":(11, 11), "RANSAC_prob":0.999, "RANSAC_threshold":0.5, "repro_threshold":3.0},
        "parking": {"maxCorners":1000, "qualityLevel":0.01, "minDistance":10, "dist_threshold_move":0, "winSize":(11, 11), "RANSAC_prob":0.999, "RANSAC_threshold":0.5, "repro_threshold":1.0},
        "malaga":  {"maxCorners":1000, "qualityLevel":0.01, "minDistance":10, "dist_threshold_move":5, "winSize":(31, 31), "RANSAC_prob":0.999, "RANSAC_threshold":0.5, "repro_threshold":5.0}
    }
    return K[dataset], params[dataset], imgs[dataset]