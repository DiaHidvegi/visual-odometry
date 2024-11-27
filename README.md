# visual-odometry
Visual Odometry Pipeline for VAMR 

## Setup /data subdirectory
Add the data in an additional ./data subdirectory (you might need to rename the first level after downloading the data)

```plaintext
data/
├── kitti/
│   ├── 05/
│   │   ├── calib.txt
│   │   ├── image_0/
│   │   │   └── ...
│   │   ├── image_1/
│   │   │   └── ...
│   │   └── times.txt
│   ├── poses/
│   │   └── ...
│
├── malaga/
│   ├── Images/
│   │   └── ...
│   ├── malaga-urban-dataset-extract-07_rectified_1024x768_Images/
│   │   └── ...
│   ├── malaga-urban-dataset-extract-07_rectified_800x600_Images/
│   │   └── ...
│   ├── README_extracts.txt
│   └── ...
│
├── parking/
│   ├── images/
│   │   └── ...
│   ├── K.txt
│   └── poses.txt
```
