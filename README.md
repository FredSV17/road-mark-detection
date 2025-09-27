## Road Mark Detection

This project focuses on detecting road markings using computer vision and deep learning techniques. The goal is to analyze the dataset, identify biases, and later build models that can robustly detect and classify different types of road marks.

At the current stage, the work has primarily focused on data analysis and visualization. This step helps uncover dataset characteristics that may affect model performance.

Dataset from Kaggle: https://www.kaggle.com/datasets/pkdarabi/road-mark-detection
## How to Run the Code

### Clone the repository

    git clone https://github.com/FredSV17/road-mark-detection

### Set up the environment

It is recommended to use a virtual environment (e.g., venv or conda).

    python -m venv venv
    source venv/bin/activate   # On Linux/Mac
    venv\Scripts\activate      # On Windows

### Install dependencies

    pip install -r requirements.txt

### Run Data Analysis

The main script is inside the data_analysis/ folder.

    python -m data_analysis.analysis --path <base_dataset_path> --dataset <dataset_type> --save_bboxes --verbose

Available Arguments:

| Argument | Description |
| :-- | :-- |
| --path | Base dataset path containing all datasets (e.g., ./data).|
| --dataset | Dataset type. Must match the dataset folder/file name. If analyzing multiple datasets, separate them with commas (e.g., train,valid).|
| --save_bboxes | Flag to save images with bounding box visualizations.|
| --verbose | Flag to enable verbose output for detailed logs.|

Example:

    python -m data_analysis.analysis --path ./data --dataset train,val --save_bboxes --verbose
## Data Analysis Insights
**1. Spatial Bias**

- Bounding box heatmaps show that most objects are concentrated in the lower part of the image (expected due to the camera perspective).

- However, this introduces spatial bias into the model, since it may overfit to certain positions.

**2. Class Imbalance**

- The dataset has significant class imbalance.

- Most bounding boxes belong to Class 2, while Classes 10 and 11 have almost no representation.

- This imbalance can severely impact model generalization and needs to be addressed (e.g., via re-weighting, data augmentation, or sampling strategies).

**3. Objects per Image Distribution**

- The number of objects per image is also unevenly distributed.

- Many images contain only a few bounding boxes, while a small portion has up to 25 objects.

- This imbalance can lead to challenges during training, as the model might underperform on crowded scenes.

**4. Bounding box Area Distribution**

- The bounding box area distribution is skewed: most bounding boxes are small.

- This increases the difficulty of detection, since small objects are harder for models (specially for single stage detectors) to localize accurately.

- A skewed distribution may bias the model toward learning certain object scales while underperforming on rare, larger objects.

### Next Steps

- Implement preprocessing pipelines to mitigate dataset imbalances.

- Begin model development and training for road mark detection.

- Explore strategies like focal loss, class weighting, and augmentation to improve minority class representation.