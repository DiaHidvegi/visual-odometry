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
        
        # Set fixed positions for legends
        self.image_ax.legend(['Landmarks', 'Candidates'], 
                           loc='upper right', 
                           bbox_to_anchor=(0.98, 0.98))
        
        self.local_trajectory_ax.legend(['Trajectory', 'Landmarks'],
                                      loc='upper right',
                                      bbox_to_anchor=(0.98, 0.98))

        # initialize data storage
        self.landmarks_data = []
        self.candidates_data = []
        self.poses = np.zeros((1, 12))

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
            # Add legend with fixed position
            self.local_trajectory_ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right')
        self.local_trajectory_line.set_data(self.poses[frame_start:frame_end, 3], self.poses[frame_start:frame_end, 11])
        
        # Update scatter points data
        self.local_points_scatter.set_offsets(frame_state.landmarks_world[[0, 2], :].T)  # Transpose for scatter

        x_min, x_max, y_min, y_max = self.determine_lims(frame_state, frame_start, frame_end)

        # Update axis limits manually  (overwritten by set_aspect but adding all 3 seems best)
        self.local_trajectory_ax.set_xlim(x_min, x_max)
        self.local_trajectory_ax.set_ylim(y_min, y_max)

        # enforce equal aspect ratio (overwrites xlim and ylim but adding all 3 seems best)
        self.local_trajectory_ax.set_aspect('equal', adjustable='datalim')

        # Update global trajectory
        if not hasattr(self, 'global_trajectory_line'):
            self.global_trajectory_line, = self.global_trajectory_ax.plot([], [], 'b-', label="Trajectory")
            self.global_trajectory_ax.set_title("Global Trajectory")
            self.global_trajectory_ax.grid(True)
        self.global_trajectory_line.set_data(self.poses[:, 3], self.poses[:, 11])
        
        # Always show the complete trajectory with padding
        x_data = self.poses[:, 3]
        y_data = self.poses[:, 11]
        x_range = max(x_data.max() - x_data.min(), 1e-6)  # Avoid division by zero
        y_range = max(y_data.max() - y_data.min(), 1e-6)  # Avoid division by zero
        
        # Calculate center of trajectory
        x_center = (x_data.max() + x_data.min()) / 2
        y_center = (y_data.max() + y_data.min()) / 2
        
        # Set the limits to the larger of the two ranges to maintain aspect ratio
        max_range = max(x_range, y_range) * 1.1  # 10% padding
        self.global_trajectory_ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
        self.global_trajectory_ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
        self.global_trajectory_ax.set_aspect('equal')

        # Update keypoints plots with improved visibility
        if not hasattr(self, 'landmarks_plot'):
            self.landmarks_plot, = self.keypoints_ax.plot([], [], '-', label="Landmarks", color='green')
            self.candidates_plot, = self.keypoints_ax.plot([], [], '-', label="Candidates", color='red')
            self.keypoints_ax.legend()
            self.keypoints_ax.grid(True)
            self.keypoints_ax.set_xlabel('Frame')
            self.keypoints_ax.set_ylabel('Count')
        
        self.landmarks_plot.set_data(range(frame_start + 1, frame_end), self.landmarks_data[frame_start:frame_end])
        self.candidates_plot.set_data(range(frame_start + 1, frame_end), self.candidates_data[frame_start:frame_end])
        
        # Improve keypoints axis visibility
        all_counts = self.landmarks_data[frame_start:frame_end] + self.candidates_data[frame_start:frame_end]
        if len(all_counts) > 0:
            max_count = max(max(all_counts), 1)  # Ensure we don't get a zero max
            self.keypoints_ax.set_ylim(0, max_count * 1.1)  # Add 10% padding
        self.keypoints_ax.set_xlim(frame_start, frame_end)

        # Update the image display
        if not hasattr(self, 'image_display'):
            self.image_display = self.image_ax.imshow(image, cmap='gray', animated=False)
            self.image_ax.set_title("Current Frame")
            self.points_plot_lm, = self.image_ax.plot([], [], 'go', markersize=2, label="Landmarks")
            self.points_plot_cnd, = self.image_ax.plot([], [], 'rx', markersize=2, label="Candidates")
            # Add legend with fixed position
            self.image_ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right')
        else:
            self.image_display.set_data(image)
            self.points_plot_lm.set_data(frame_state.landmarks_image[0,:], frame_state.landmarks_image[1,:])
            self.points_plot_cnd.set_data(frame_state.cand_landmarks_image_current[0,:], frame_state.cand_landmarks_image_current[1,:])
        self.image_ax.axis('off')

        # Show updated plots
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def determine_lims(self, frame_state, frame_start, frame_end):
        # Combine data to ensure axis limits include all points
        x_data = np.concatenate([self.poses[frame_start:frame_end, 3], frame_state.landmarks_world[0, :]])
        y_data = np.concatenate([self.poses[frame_start:frame_end, 11], frame_state.landmarks_world[2, :]])

        # Determine the initial axis limits
        x_min, x_max = x_data.min(), x_data.max()
        y_min, y_max = y_data.min(), y_data.max()

        # Compute the spans of each axis
        x_span = (x_max - x_min) * 1.1
        y_span = (y_max - y_min) * 1.1

        # Determine the larger span and adjust limits symmetrically
        if x_span > y_span:
            y_center = (y_min + y_max) / 2
            y_min = y_center - x_span / 2
            y_max = y_center + x_span / 2
        else:
            x_center = (x_min + x_max) / 2
            x_min = x_center - y_span / 2
            x_max = x_center + y_span / 2

        return x_min, x_max, y_min, y_max

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
