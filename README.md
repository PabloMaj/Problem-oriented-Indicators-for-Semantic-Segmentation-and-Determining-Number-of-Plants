## Problem-oriented Indicators for Semantic Segmentation and Determining Number of Plants
This is the official repository to the paper **"Problem-oriented Indicators for Semantic Segmentation and Determining Number of Plants"** by Paweł Majewski and Jacek Reiner.

![](Diff_indicators_comparision.png)

## Data

Datasets are available under the link https://drive.google.com/drive/folders/13ZJbOUWGAEkRtd9LJnYEJrmyercfWyOq?usp=sharing.

Put data folder in the same location as *.py files

## Source Code

1) *add_color_spaces_testing.py* - Training and testing models with additional channels from HSV and Lab color spaces
2) *counting_plants.py* - Training and testing models for plants counting
3) *create_chart_f1_score_vs_no_samples.py* - Creating charts f1-score vs number of labeled samples
4) *error_VARI_bar_plot.py* - Creating bar chart to show estimation errors of mean VARI for plants 
5) *f1_score_vs_no_samples.py* - Training and testing models for segmentation with a different number of labeled samples.
6) *fraction_indicator_PSO_optimatization.py* - PSO implementation for fraction indicators optimatization
7) *mean_VARI_segmentation.py* - Calculating estimation errors of mean VARI for plants
8) *opt_parameters.py* - Optimized parameters for chosen models
9) *ROC_segmentation.py* - Creating ROC-curves as threshold-indepentednt analysis for segmentation
10) *segmentation_plants.py* - Training and testing models for plants segmentation
11) *universality.py* - Checking universality of chosen models
12) *utilities.py* - Universal functions used in implementation
13) *visualization_segmentation.py* - Visualization of segmentation for chosen models



