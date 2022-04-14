### Project Goal
#### Find the body type that is most similar to you without any index(like obesity indexs)

#### Datasets
3D scan data (total 1524 people)

There are 67 variables obtained through 3D data (height, Area and volume of each body part,..)

#### Process
1) normalization variables
2) Remove unnecessary variables and reduce dimensionality through PCA
3) Clustering with KMeans algorithm → 4 clusters are derived
4) Calculate the similarity and derive the most similar body type with Euclidean!
