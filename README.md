# Visual Odometry Pipeline

A monocular visual odometry pipeline implementation that works with the KITTI, Parking, and Malaga datasets.

## Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/DiaHidvegi/visual-odometry.git
cd visual-odometry
```

2. Create and activate the conda environment from the provided YAML file:
```bash
conda env create -f vamr_proj.yaml
conda activate vamr_proj
```

## Dataset Setup

Create a `data` directory in the root of the project and organize the datasets as follows:

```plaintext
data/
├── kitti/
│   ├── 05/
│   │   ├── calib.txt
│   │   ├── image_0/
│   │   │   └── 000000.png, 000001.png, ...
│   │   ├── image_1/
│   │   │   └── ...
│   │   └── times.txt
│   └── poses/
│       └── ...
│
├── malaga/
│   ├── malaga-urban-dataset-extract-07_rectified_800x600_Images/
│   │   └── left_000000000000.jpg, left_000000000001.jpg, ...
│   └── ...
│
└── parking/
    ├── images/
    │   └── img_00000.png, img_00001.png, ...
    ├── K.txt
    └── poses.txt
```

### Dataset Downloads

1. **KITTI Dataset**: 
   - Download from [here](https://rpg.ifi.uzh.ch/docs/teaching/2024/kitti05.zip)
   - Extract to `data/kitti/`

2. **Malaga Dataset**:
   - Download from [here](https://rpg.ifi.uzh.ch/docs/teaching/2024/malaga-urban-dataset-extract-07.zip)
   - After unzipping, rename the top folder from "malaga-urban-dataset-extract-07" to "malaga"
   - Extract to `data/malaga/`

3. **Parking Dataset**:
   - Download from [here](https://rpg.ifi.uzh.ch/docs/teaching/2024/parking.zip)
   - Extract to `data/parking/`

## Running the Pipeline

1. Select which dataset to run by modifying the `config` variable in `main.py`:

```python
# Configuration
config = Config(
    dataset="kitti",  # Options: "kitti", "parking", "malaga"
    max_frames=1000,  # Maximum number of frames to process
    visualization_delay=0.1  # Delay between frames (seconds)
)
```

2. Run the pipeline:
```bash
python main.py
```

## Visualization

The pipeline provides real-time visualization with four panels:
- Current frame with candidate and tracked features
- Number of candidate and tracked landmarks over time
- Global trajectory
- Local trajectory with visible landmarks

## Troubleshooting

If you encounter any issues:
1. Verify that the dataset paths match the expected structure
2. Check that the conda environment was created and activated correctly
3. Ensure the selected dataset exists in the `data` directory
4. Verify that the images are in the correct format (PNG for KITTI/Parking, JPG for Malaga)

## Authors

- Nicolas Schuler [(NicSchuler)](https://github.com/NicSchuler)
- Jakob Schlieter [(jschli)](https://github.com/jschli)
- Max Stralz [(maxstralz)](https://github.com/maxstralz)
- Diana Hidvegi [(Dia Hidvegi)](https://github.com/DiaHidvegi)