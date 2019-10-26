import tensorflow as tf
import os, json
import numpy as np
from PIL import Image
import utils as ut
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from models.detector import Detector

cfg_path = 'base.json'
cfg = json.load(open(cfg_path))

img_path = 'imgs/speed_test.png'
ckpt_dir = 'checkpoints/lightweight'
anchor_scales = [16, 32]
anchor_strides = [8, 8]

repeat_times = 1010
output_image = False

eval_params = {'match_thres': 0.8,
               'init_scale': 1.,
               'pyramid_scale': 0.5,
               'max_output_size': 400,
               'enable_flip': False}


def get_dataset():
    image = Image.open(img_path)
    image = np.array(image).astype(np.float32)
    dataset = tf.data.Dataset.from_tensors(image)
    dataset = dataset.repeat(repeat_times)
    dataset = dataset.batch(10, drop_remainder=True)
    dataset.prefetch(None)
    return dataset

def save_result(result):
    i=0
    tic = None
    for res in result:
        if i % 100 == 0:
            print(i)
        i += 1
        if i == 1:
            print('tic')
            tic = time.time()
        boxes, scores = res['bboxes'], res['scores']
        #if output_image:
        #    out_fname = os.path.join('output', 'speed_test.jpg')
        #    boxes = boxes[0]
        #    scores = scores[0]
        #    im = Image.open(img_path)
        #    im = ut.np_ops.draw_boxes(im, boxes, width=2)
        #    print('output: {}\t{}'.format(out_fname, len(boxes)))
        #    im.save(out_fname)
    print('toc')
    toc = time.time()
    print('total_time: {}'.format(toc-tic))

def detect():
    detector = Detector(anchor_scales=anchor_scales,
                        anchor_strides=anchor_strides,
                        eval_params=eval_params)
    classifier = tf.estimator.Estimator(model_fn=detector.model_fn,
                                        model_dir=ckpt_dir,)
    results = classifier.predict(get_dataset, yield_single_examples=False)
    save_result(results)

if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    detect()
