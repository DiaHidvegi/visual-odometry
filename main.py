import numpy as np
import cv2

from initialization import Initialization
from visualizer import Visualizer
from framestate import FrameState
from continous import ContinuousVO

from time import sleep

def main():
    data_choice = "kitti"
    initialization = Initialization(data_choice, False)
    points3D, points2D = initialization.get_initial_landmarks()

    state0 = FrameState(
        landmarks_image=points2D, 
        landmarks_world=points3D, 
        cand_landmarks_image_current=np.empty((2, 0)),
        cand_landmarks_image_first=np.empty((2, 0)),
        cand_landmarks_transform=np.empty((12, 0))
        )

    visualizer = Visualizer()
    vo = ContinuousVO(K=initialization.K, datachoice=data_choice)

    frame_state = state0

    for i in range(1, 1000):
        try:
            # image_path = f"data/parking/images/img_{str(i).zfill(5)}.png"
            #kitti
            image_path_prev = f"data/{data_choice}/05/image_0/{str(i-1).zfill(6)}.png"
            image_path_current = f"data/{data_choice}/05/image_0/{str(i).zfill(6)}.png"
            #parking
            # image_path_prev = f"data/{data_choice}/images/img_{str(i-1).zfill(5)}.png"
            # image_path_current = f"data/{data_choice}/images/img_{str(i).zfill(5)}.png"

            img_prev = cv2.imread(image_path_prev, cv2.IMREAD_GRAYSCALE)
            img_current = cv2.imread(image_path_current, cv2.IMREAD_GRAYSCALE)

            frame_state, pose = vo.process_frame(img_current, img_prev, frame_state)
            
            visualizer.update(frame_state, pose, img_current)

            sleep(0.05)

        except Exception as e:
            print(f"---- Error at frame {i}: {e} ----")
            break
    
    visualizer.close()

if __name__ == "__main__":
    main()
