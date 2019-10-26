## A lightweight face detector with integrating the convolutional neural network into the image pyramid ##
A Tensorflow Implementation.

|     | WIDERFACE easy | WIDERFACE medium | WIDERFACE hard | FDDB  |
|-----|----------------|------------------|----------------|-------|
| mAP | 0.871          | 0.873            | 0.780          | 0.979 |

### Description
Model is trained on WIDERFACE train set and can be evaluate on WIDERFACE validation, test set and FDDB.

The link of WIDERFACE is [http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/].

The link of FDDB is [http://vis-www.cs.umass.edu/fddb/].

### The code is tested on
* python 3.6
* tensorflow 1.12

### Prepare data
1. Download WIDER face dataset.
1. Add the root directory of this repo to PYTHONPATH.
1. ```cd datasets```
1. Depending on your dataset path, change the path in datasets/generate_cropped_tfrecords.py
1. ```python datasets/generate_cropped_tfrecords.py```

### Train 
The code support Multi-GPU training. To enable it, you are supposed to change the ```os.environ['CUDA_VISIBLE_DEVICES']``` on ```run/train.py```

To start training, run ```python run/train.py```

### Eval
```python run/eval_wider.py```

```python run/eval_fddb.py```


Training code will release soon.
