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

Decide which dataset you want the pipeline to run on and execute the following command in a terminal where the ```vamr_proj``` conda environment is active:

```bash
python main.py --dataset DATASET
```

Therein, ```DATASET``` can take one of three values, depending which dataset you want to run:
* ```parking```: To run the Parking dataset.
* ```kitti```: To run the KITTI dataset.
* ```malaga```: To run the Malaga dataset.

If no value is set for the ```--dataset``` argument, ```parking``` is chosen as default. Example command to run the pipeline on the KITTI dataset:
```bash
python main.py --dataset kitti
```

## Visualization

The pipeline provides real-time visualization with four panels:
- Current frame with candidate and tracked landmarks.
- Number of candidate and tracked landmarks over time.
- Global trajectory.
- Local trajectory with visible landmarks.

## Screencasts
The pipeline has been tested on the previously mentioned three datasets Parking, KITTI and Malaga. The links to the screencasts (YouTube links) as well as the links to download the datasets are provided in the following table:

| **Dataset** | **Link to Screencast**                           | **Dataset Download Link**                                                                                       |
|-------------|--------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| Parking     | [https://youtu.be/21qXO7S11wA](https://youtu.be/21qXO7S11wA) | [Download](https://rpg.ifi.uzh.ch/docs/teaching/2024/parking.zip)       |
| KITTI       | [https://youtu.be/bQ9UpPFiB0k](https://youtu.be/bQ9UpPFiB0k) | [Download](https://rpg.ifi.uzh.ch/docs/teaching/2024/kitti05.zip)         |
| Malaga      | [https://youtu.be/dmmEDB3kfuk](https://youtu.be/dmmEDB3kfuk) | [Download](https://rpg.ifi.uzh.ch/docs/teaching/2024/malaga-urban-dataset-extract-07.zip) |

The screencasts are recorded on a PC with the following hardware specifications:
|                         | MacBook Pro 13-inch, 2020 |
|:------------------------|:--------------------------|
| **CPU**                | Apple M1                 |
| **RAM**                | 16 GB               |
| **OS**                 | macOS Sonoma 14.6.1      |
| **# threads**          | 1                        |


## Troubleshooting

If you encounter any issues:
1. Verify that the dataset paths match the expected structure.
2. Check that the conda environment was created and activated properly.
3. Ensure the selected dataset exists in the `data` directory.
4. Verify that the images are in the correct format (PNG for KITTI/Parking, JPG for Malaga).

## Authors

- Nicolas Schuler [(NicSchuler)](https://github.com/NicSchuler)
- Jakob Schlieter [(jschli)](https://github.com/jschli)
- Max Stralz [(maxstralz)](https://github.com/maxstralz)
- Diana Hidvegi [(Dia Hidvegi)](https://github.com/DiaHidvegi)