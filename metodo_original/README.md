### Introduction
This project aims to offer easy access to Hypergraph-based Spiking Neural P Systems for Predicting the Overall Survival Time of Glioblastoma Patients. 
It can be trained to predict the overall survival (OS) time of glioblastoma (GBM) patients if corresponding ground truth labels are provided for training.

### Requirements
V100 16G; cuda10.2; torch 1.6; openslide; PCV

### Dataset
GBM cohorts from The Cancer Genome Atlas (TCGA-GBM) **[1]**

### Pre-Processing
The whole slide histopathological images (WSIs) should be in the **.svs** format.
During preprocessing, all WSIs are cut into patches by openslide. 
Then the patches of the background area are discarded by the pixel value. All patches should be in the **.png** format.
```bash
python preprocessing_slide.py
```

### Clustering 
Grayscale and resize the patches of all patients.
Then k-means clustering is performed on all the patches' phenotypes. 
Finally, patches in each cluster are restored to the original images.
```bash
python grayscale_processing.py
python picture_resize.py
python kmeans_clustering.py
python returned.py
```

### Selection of Survival-Related Clusters
Patches in each cluster train a ResNet50 model respectively.
Then the survival-related clusters are selected according to the OS classification results.
```bash
python select_clusters.py
```

###Final OS Time Prediction
Finally, all survival-related clusters are used for OS time prediction.

**train**
```bash
python OS_time_prediction_train.py
```
**test**
```bash
python OS_time_prediction_test.py
```

[1] Weinstein, J. N., Collisson, E. A., Mills, G. B., Shaw, K. R., Ozenberger, B. A., Ellrott, K., Shmulevich, I., Sander, C., and Stuart, J. M. (2013). The cancer genome atlas pan-cancer analysis project. Nature genetics, 45(10):1113–1120.
