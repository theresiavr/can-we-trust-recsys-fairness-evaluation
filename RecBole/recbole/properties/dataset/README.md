The `num_clusters` parameter refers to the model setting of NCL.

They are set to a value approximately $\min(numUser, numItem)$//39. 

Why 39? To avoid having too few points in a cluster (see [link](https://github.com/facebookresearch/faiss/blob/833d417db1b6b6fd4b19e092f735373f07eab33f/Clustering.cpp#L40))
