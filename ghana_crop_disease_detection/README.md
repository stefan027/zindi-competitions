# Ghana Crop Disease Detection Challenge

## Solution overview
The solutions consists of the following three elements: 
 1. Training and inference
 2. Explainability module
 3. Deployment module

Please refer to the relevant sections below for more details on each element.

## Installation
The required Python packages can be installed using the steps below. Alternatively, the depenencies can be install directly in the Jupyter notebooks.

>Make sure you are in the base folder when installing `MMDetection` (i.e., the same folder where `requirements.txt`, `Train.csv`, `Test.csv`, the Jupyter notebooks, and the images are; see `Setting up the folders` below for more details.)

1. Install required Python packages:
```bash
pip install -q -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```

2. Install OpenMIM, MMEngine and MMCV:
```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0rc4, <2.2.0"
```

3. Install MMDetection:
```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection && pip install -e .
```

## Training and inference

### Setting up the folders
Before running the notebooks, make sure that all three notebooks are in the same folder as the `Train.csv`, `Test.csv`, and `SampleSubmission.csv` files. The path to the images must be specified in the notebooks. 

    Base directory
    - Train.csv
    - Test.csv
    - SampleSubmission.csv
    - requirements.txt
    - 01_ghana-crop-dza-dino.ipynb
    - 02_object_detection_shap.ipynb
    - 03_convert_to_onnx.ipynb

### Training and inference
To run training and inference end-to-end, simply run the `01_ghana-crop-dza-dino.ipynb` from top-to-bottom. Make sure to set the correct paths for `IMAGE_DIR` and `DATA_DIR` at the top of the notebook.

>The file submission file is called `submission_dino_convnext_t.csv`.

## Hardware specifications
The model was trained on a single Kaggle T4 GPU. The total time for training and inference is approximately 7 to 7.5 hours. The full training log from Kaggle is provided which shows that the submission run completed in 25,386 seconds.

## Explainability module
A notebook to calculate [Shapley values](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html) is provided. Please see `02_object_detection_shap.ipynb`

## Deployment module
To convert the  model to [ONNX](https://onnx.ai/) for deployment, please see `03_convert_to_onnx.ipynb`.
