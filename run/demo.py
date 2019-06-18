import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import os, re, time
from PIL import Image, ImageDraw

import tensorflow as tf
import numpy as np

img_dir='imgs/'
out_dir='outputs/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

from nets.cascadeOut import CascadeOut
net = CascadeOut()

model_dir = 'checkpoints/tinyNet_s8'
params = {'thres': 0.4,
          'init_scale': 1.,
          'max_output_size': 800,
          'iou_thres': 0.2}

def get_imgs():
    fnames = []
    for file in os.listdir(img_dir):
        if file.endswith('.jpg') or file.endswith('.png'):
            fnames.append(os.path.join(img_dir, file))
    return fnames

def get_dataset():
    fnames = get_imgs()
    dataset = tf.data.Dataset.from_tensor_slices((fnames,))
    def fn(fname):
        raw_img = tf.read_file(fname)
        image = tf.image.decode_jpeg(raw_img, channels=3)
        image = tf.cast(image, tf.float32)
        return {'fname': fname, 'image': image}
    dataset = dataset.map(fn, num_parallel_calls=8)
    dataset.prefetch(None)
    return dataset

def save_result(fnames, result):
    i=0
    for fname in fnames:
        i += 1
        out_fname = os.path.split(fname)[-1]
        out_fname = os.path.join(out_dir, out_fname)
        res = next(result)
        bboxes, scores = res['bboxes'], res['scores']
        im = Image.open(fname)
        imDraw = ImageDraw.Draw(im)
        for _i in range(len(scores)):
            s, b =scores[_i], bboxes[_i]
            b=[int(np.round(x)) for x in b]
            imDraw.rectangle([b[1], b[0], b[3], b[2]], outline='yellow', width=1)
        print('output: {}\t{}'.format(out_fname, len(bboxes)))
        im.save(out_fname)

def detect():
    runConfig = tf.estimator.RunConfig(model_dir=model_dir)
    classifier = tf.estimator.Estimator(model_fn=net.model_fn,
                                        params=params,
                                        config=runConfig)
    results = classifier.predict(get_dataset)
    fnames = get_imgs()
    save_result(fnames, results)

if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    detect()