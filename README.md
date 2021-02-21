## Optimized indicators for ultra-fast semantic segmentation and celery counting
This is the official repository to the paper **"Optimized indicators for ultra-fast semantic segmentation and celery counting"** by Paweł Majewski and Jacek Reiner.

![alt text](https://github.com/PabloMaj/Optimized-indicators-for-ultra-fast-semantic-segmentation-and-celery-counting/Diff_indicators_comparision.png)

## Data

Datasets are available under the link https://drive.google.com/drive/folders/13ZJbOUWGAEkRtd9LJnYEJrmyercfWyOq?usp=sharing.

Put data folder in the same location as *.py files

## Source Code

1) *add_color_spaces_testing.py* - Training and testing models with additional channels from HSV and Lab color spaces
2) *counting_plants.py* - Training and testing models for plants counting
3) *create_chart_f1_score_vs_no_samples.py* - Creating charts f1-score vs number of labeled samples
4) *f1_score_vs_no_samples.py* - Training and testing models for segmentation with a different number of labeled samples.
5) *fraction_indicator_PSO_optimatization.py* - PSO implementation for fraction indicators optimatization
6) *opt_parameters.py* - Optimized parameters for chosen models
7) *segmentation_plants.py* - Training and testing models for plants segmentation
8) *universality.py* - Checking universality of chosen models
9) *utilities.py* - Universal functions used in implementation
10) *visualization_segmentation.py* - Visualization of segmentation for chosen models
