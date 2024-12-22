import numpy as np
from dataclasses import dataclass


@dataclass
class FrameState:
    landmarks_image: np.ndarray  # expected shape (2 x K)
    landmarks_world: np.ndarray  # expected shape (3 x K)
    cand_landmarks_image_current: np.ndarray  # expected shape (2 x M)
    cand_landmarks_image_first: np.ndarray   # expected shape (2 x M)
    # expected shape (12 x M) or (16 x M)
    cand_landmarks_transform: np.ndarray

    def __post_init__(self):
        # run data validations
        self._validate_landmarks()
        self._validate_candidate_landmarks()

    def _validate_landmarks(self):
        # Validate that 'landmarks_image' is 2 x K
        if self.landmarks_image.ndim != 2 or self.landmarks_image.shape[0] != 2:
            raise ValueError(
                f"'landmarks_image' must have dimensions 2 x K. Got shape {self.landmarks_image.shape}.")

        # Validate that 'landmarks_world' is 3 x K
        if self.landmarks_world.ndim != 2 or self.landmarks_world.shape[0] != 3:
            raise ValueError(
                f"'landmarks_world' must have dimensions 3 x K. Got shape {self.landmarks_world.shape}.")

        # Validate that 'landmarks_image' and 'landmarks_world' have same amount of points
        if self.landmarks_image.shape[1] != self.landmarks_world.shape[1]:
            raise ValueError(
                f"'landmarks_image' and 'landmarks_world' must have shapes (2 x K) and (3 x K). Got {self.landmarks_image.shape} and {self.landmarks_world.shape}.")

    def _validate_candidate_landmarks(self):
        # Validate that 'cand_landmarks_image_current' is 2 x M
        if self.cand_landmarks_image_current.ndim != 2 or self.cand_landmarks_image_current.shape[0] != 2:
            raise ValueError(
                f"'cand_landmarks_image_current' must have dimensions 2 x M. Got shape {self.cand_landmarks_image_current.shape}.")

        # Validate that 'cand_landmarks_image_first' is 2 x M
        if self.cand_landmarks_image_first.ndim != 2 or self.cand_landmarks_image_first.shape[0] != 2:
            raise ValueError(
                f"'cand_landmarks_image_first' must have dimensions 2 x M. Got shape {self.cand_landmarks_image_first.shape}.")

        # Validate that 'cand_landmarks_transform' has valid dimensions
        if self.cand_landmarks_transform.ndim != 2 or self.cand_landmarks_transform.shape[0] not in {12, 16}:
            raise ValueError(
                f"'cand_landmarks_transform' must have dimensions (12 x M) or (16 x M). Got shape {self.cand_landmarks_transform.shape}.")

        # Validate that 'cand_landmarks_image_current', 'cand_landmarks_image_first', and 'cand_landmarks_transform' refer to the same M
        num_cand_landmarks_current = self.cand_landmarks_image_current.shape[1]
        num_cand_landmarks_first = self.cand_landmarks_image_first.shape[1]
        num_cand_landmarks_transform = self.cand_landmarks_transform.shape[1]

        if num_cand_landmarks_current != num_cand_landmarks_first or \
           num_cand_landmarks_current != num_cand_landmarks_transform:
            raise ValueError(
                f"'cand_landmarks_image_current', 'cand_landmarks_image_first', and 'cand_landmarks_transform' must all refer to the same number of landmarks (M). Got shapes {self.cand_landmarks_image_current.shape}, {self.cand_landmarks_image_first.shape}, and {self.cand_landmarks_transform.shape}.")

    # provide nicer string-representation
    def __str__(self):
        return (
            f"FrameState with {self.landmarks_image.shape[1]} landmarks and {self.cand_landmarks_image_current.shape[1]} candidate landmarks:\n"
            f"  landmarks_image: shape={self.landmarks_image.shape},\n"
            f"  landmarks_world: shape={self.landmarks_world.shape},\n"
            f"  cand_landmarks_image_current: shape={self.cand_landmarks_image_current.shape},\n"
            f"  cand_landmarks_image_first: shape={self.cand_landmarks_image_first.shape},\n"
            f"  cand_landmarks_transform: shape={self.cand_landmarks_transform.shape}"
        )


if __name__ == "__main__":
    K = 5  # Number of main landmarks
    M = 4  # Number of candidate landmarks

    # Create valid data
    landmarks_image = np.random.rand(2, K)  # Shape (2, K)
    landmarks_world = np.random.rand(3, K)  # Shape (3, K)

    cand_landmarks_image_current = np.random.rand(2, M)  # Shape (2, M)
    cand_landmarks_image_first = np.random.rand(2, M)    # Shape (2, M)
    cand_landmarks_transform = np.random.rand(16, M)     # Shape (12, M)

    # Create the object
    frame_state = FrameState(
        landmarks_image=landmarks_image,
        landmarks_world=landmarks_world,
        cand_landmarks_image_current=cand_landmarks_image_current,
        cand_landmarks_image_first=cand_landmarks_image_first,
        cand_landmarks_transform=cand_landmarks_transform
    )

    print(frame_state)


def generate_frame_state():
    # generate a framestate with random values
    K = 5  # Number of main landmarks
    M = 4  # Number of candidate landmarks

    # Create valid data
    landmarks_image = np.random.rand(2, K)  # Shape (2, K)
    landmarks_world = np.random.rand(3, K)  # Shape (3, K)

    cand_landmarks_image_current = np.random.rand(2, M)  # Shape (2, M)
    cand_landmarks_image_first = np.random.rand(2, M)    # Shape (2, M)
    cand_landmarks_transform = np.random.rand(16, M)     # Shape (12, M)

    # Create the object
    frame_state = FrameState(
        landmarks_image=landmarks_image,
        landmarks_world=landmarks_world,
        cand_landmarks_image_current=cand_landmarks_image_current,
        cand_landmarks_image_first=cand_landmarks_image_first,
        cand_landmarks_transform=cand_landmarks_transform
    )
    return frame_state
