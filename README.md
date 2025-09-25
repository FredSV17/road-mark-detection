## Road Mark Detection

This project focuses on detecting road markings using computer vision and deep learning techniques. The goal is to analyze the dataset, identify biases, and later build models that can robustly detect and classify different types of road marks.

At the current stage, the work has primarily focused on data analysis and visualization. This step helps uncover dataset characteristics that may affect model performance.

### Data Analysis Insights
1. Spatial Bias

Bounding box heatmaps show that most objects are concentrated in the lower part of the image (expected due to the camera perspective).

However, this introduces spatial bias into the model, since it may overfit to certain positions.

2. Class Imbalance

The dataset has significant class imbalance.

Most bounding boxes belong to Class 2, while Classes 10 and 11 have almost no representation.

This imbalance can severely impact model generalization and needs to be addressed (e.g., via re-weighting, data augmentation, or sampling strategies).

3. Objects per Image Distribution

The number of objects per image is also unevenly distributed.

Many images contain only a few bounding boxes, while a small portion has up to 25 objects.

This imbalance can lead to challenges during training, as the model might underperform on crowded scenes.

### Next Steps

- Implement preprocessing pipelines to mitigate dataset imbalances.

- Begin model development and training for road mark detection.

- Explore strategies like focal loss, class weighting, and augmentation to improve minority class representation.