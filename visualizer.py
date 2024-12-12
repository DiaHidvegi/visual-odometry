import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
from framestate import FrameState, generate_frame_state

class Visualizer:
    def __init__(self):
        # create a custom layout using gridspec
        self.fig = plt.figure(figsize=(10, 6))
        self.gs = gridspec.GridSpec(2, 4, figure=self.fig)
        
        # initialize subplots
        self.image_ax = self.fig.add_subplot(self.gs[0, 0:2])
        self.keypoints_ax = self.fig.add_subplot(self.gs[1, 0])
        self.global_trajectory_ax = self.fig.add_subplot(self.gs[1, 1])
        self.local_trajectory_ax = self.fig.add_subplot(self.gs[0:, 2:])
        
        # initialize data storage
        self.landmarks_data = []
        self.candidates_data = []
        self.poses = np.zeros((0, 12))

        # initialize interactive mode
        plt.tight_layout()
        plt.ion()
        plt.show()

    def update(self, frame_state: FrameState, pose: np.ndarray, image):
        """
        Update the visualization with the new data.
        :param frame_state: FrameState object
        :param pose: N x 12 array representing the camera pose.
        :param image: image to display.
        """
        assert pose.shape == (1, 12), 'Wrong shape of pose, expected (1, 12)'

        # add data to data structures
        self.landmarks_data.append(frame_state.landmarks_image.shape[1])
        self.candidates_data.append(frame_state.cand_landmarks_image_current.shape[1])
        self.poses = np.vstack((self.poses, pose))

        # find which frames to display
        HISTORY_LENGTH = 20
        frame_start = max(len(self.landmarks_data) - HISTORY_LENGTH, 0)
        frame_end = len(self.landmarks_data) + 1

        # Update local trajectory
        if not hasattr(self, 'local_trajectory_line'):
            self.local_trajectory_line, = self.local_trajectory_ax.plot([], [], 'b-', label="Trajectory")
            self.local_points_scatter = self.local_trajectory_ax.scatter([], [], color='green', label="Landmarks")
            self.local_trajectory_ax.set_title("Local Trajectory")
        self.local_trajectory_line.set_data(self.poses[frame_start:frame_end, 3], self.poses[frame_start:frame_end, 11])
        self.local_points_scatter.set_offsets(frame_state.landmarks_world[:, [0, 2]])
        self.local_trajectory_ax.relim()
        self.local_trajectory_ax.autoscale_view()
        self.local_trajectory_ax.set_aspect('equal', adjustable='datalim')

        # Update global trajectory
        if not hasattr(self, 'global_trajectory_line'):
            self.global_trajectory_line, = self.global_trajectory_ax.plot([], [], 'b-', label="Trajectory")
            self.global_trajectory_ax.set_title("Global Trajectory")
        self.global_trajectory_line.set_data(self.poses[:, 3], self.poses[:, 11])
        self.global_trajectory_ax.relim()
        self.global_trajectory_ax.autoscale_view()
        self.global_trajectory_ax.set_aspect('equal', adjustable='datalim')

        # Update keypoints plots
        if not hasattr(self, 'landmarks_plot'):
            self.landmarks_plot, = self.keypoints_ax.plot([], [], '-', label="Landmarks", color='green')
            self.candidates_plot, = self.keypoints_ax.plot([], [], '-', label="Candidates", color='red')
            self.keypoints_ax.legend()
        self.landmarks_plot.set_data(range(frame_start + 1, frame_end), self.landmarks_data[frame_start:frame_end])
        self.candidates_plot.set_data(range(frame_start + 1, frame_end), self.candidates_data[frame_start:frame_end])
        self.keypoints_ax.relim()
        self.keypoints_ax.autoscale_view()
        self.keypoints_ax.set_ylim(bottom=0, top=self.keypoints_ax.get_ylim()[1])

        # Update the image display
        if not hasattr(self, 'image_display'):
            self.image_display = self.image_ax.imshow(image, animated=False)
            self.image_ax.set_title("Current Frame")
            self.points_plot_lm, = self.image_ax.plot([], [], 'go', markersize=10)
            self.points_plot_cnd, = self.image_ax.plot([], [], 'rx', markersize=10)
        else:
            self.image_display.set_data(image)
            self.points_plot_lm.set_data(frame_state.landmarks_image[:,0], frame_state.landmarks_image[:,1])
            #self.points_plot_cnd.set_data(frame_state.cand_landmarks_image_current[:,0], frame_state.cand_landmarks_image_current[:,1])
        self.image_ax.axis('off')

        # Show updated plots
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self):
        """Close the visualization."""
        plt.close(self.fig)

# Example Usage
if __name__ == "__main__":    
    visualizer = Visualizer()

    for i in range(100):
        frame_state = generate_frame_state()
        #image_path = f"data/parking/images/img_{str(i).zfill(5)}.png"
        image_path = f"data/kitti/05/image_0/{str(i).zfill(6)}.png"
        image = cv2.imread(image_path)
        pose = np.random.rand(1, 12)
        
        visualizer.update(frame_state, pose, image)
    
    visualizer.close()
