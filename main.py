import numpy as np
import cv2
import os
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

from initialization import Initialization
from visualizer import Visualizer
from framestate import FrameState
from continuous import ContinuousVO

from time import sleep


@dataclass
class Config:
    dataset: str
    max_frames: int = 1000
    visualization_delay: float = 0.05


class ImageLoader:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.base_path = Path("data") / dataset

    def get_image_paths(self, frame_idx: int) -> Tuple[str, str]:
        if self.dataset == "malaga":
            malaga_base_path = self.base_path / \
                "malaga-urban-dataset-extract-07_rectified_800x600_Images"
            malaga_paths = sorted(
                [str(p) for p in malaga_base_path.glob("*left*")]
            )
            return malaga_paths[frame_idx-1], malaga_paths[frame_idx]

        elif self.dataset == "kitti":
            return (
                str(self.base_path / "05/image_0" /
                    f"{(frame_idx-1):06d}.png"),
                str(self.base_path / "05/image_0" / f"{frame_idx:06d}.png")
            )

        elif self.dataset == "parking":
            return (
                str(self.base_path / "images" /
                    f"img_{(frame_idx-1):05d}.png"),
                str(self.base_path / "images" / f"img_{frame_idx:05d}.png")
            )

        raise ValueError(f"Unknown dataset: {self.dataset}")

    def load_images(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        prev_path, current_path = self.get_image_paths(frame_idx)
        img_prev = cv2.imread(prev_path, cv2.IMREAD_GRAYSCALE)
        img_current = cv2.imread(current_path, cv2.IMREAD_GRAYSCALE)

        if img_prev is None or img_current is None:
            raise FileNotFoundError(
                f"Could not load images for frame {frame_idx}")

        return img_prev, img_current


def initialize_state(initialization: Initialization) -> FrameState:
    points3D, points2D = initialization.get_initial_landmarks()
    return FrameState(
        landmarks_image=points2D,
        landmarks_world=points3D,
        cand_landmarks_image_current=np.empty((2, 0)),
        cand_landmarks_image_first=np.empty((2, 0)),
        cand_landmarks_transform=np.empty((12, 0))
    )


def main() -> None:
    # Configuration
    config = Config(dataset="malaga")  # kitti, parking, malaga

    # Initialize components
    initialization = Initialization(config.dataset, False)
    visualizer = Visualizer()
    vo = ContinuousVO(K=initialization.K, datachoice=config.dataset)
    image_loader = ImageLoader(config.dataset)

    # Initialize state
    frame_state = initialize_state(initialization)

    try:
        for frame_idx in range(1, config.max_frames):
            try:
                # Load and process images
                img_prev, img_current = image_loader.load_images(frame_idx)
                frame_state, pose = vo.process_frame(
                    img_current, img_prev, frame_state)

                # Visualize results
                visualizer.update(frame_state, pose, img_current)
                sleep(config.visualization_delay)

            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                raise

    finally:
        visualizer.close()


if __name__ == "__main__":
    main()
