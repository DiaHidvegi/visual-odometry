import numpy as np
import cv2

from initialization import Initialization
from visualizer import Visualizer
from framestate import FrameState

def main():
    # for KITTI
    img1 = 'data/kitti/05/image_0/000000.png'
    img2 = 'data/kitti/05/image_0/000002.png'
    K = np.array([[707.09,  0, 601.88],
                  [ 0, 707.09, 183.11],
                  [ 0,  0,  1]])
    
    init = Initialization(img1, img2, K, show_plots=False)
    state0 = FrameState(
        landmarks_image=init.points2D, 
        landmarks_world=init.points3D, 
        cand_landmarks_image_current=np.empty((2, 0)),
        cand_landmarks_image_first=np.empty((2, 0)),
        cand_landmarks_transform=np.empty((16, 0))
        )
    
    print(state0)

    # next steps: run continuous module

if __name__ == "__main__":
    main()
