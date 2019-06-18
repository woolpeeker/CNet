import tensorflow as tf
import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import tracemalloc

from models.detector import Detector, RefineDetector
from datasets.widerface import WiderEval, save_wider_result

wider_type = 'wider_val'
cfg_path = 'base.json'
cfg = json.load(open(cfg_path))


ckpt_dir = 'checkpoints/fused16_32'
anchor_scales = [16, 32]
anchor_strides = [8, 8]

eval_params = {'match_thres': 0.01,
               'init_scale': 1.,
               'pyramid_scale': 0.5,
               'max_output_size': 400}

imgs_dir = cfg[wider_type]['image_dir']
anno_mat = cfg[wider_type]['mat_path']

def eval_on_wider():
    output_dir = os.path.join(ckpt_dir, '%s_result' % wider_type)
    os.makedirs(output_dir, exist_ok=True)
    tf.logging.info('output_dir: %s'%output_dir)

    dataset = WiderEval(imgs_dir=imgs_dir, anno_mat=anno_mat)

    detector = Detector(anchor_scales=anchor_scales,
                        anchor_strides=anchor_strides,
                        eval_params=eval_params)

    classifier = tf.estimator.Estimator(model_fn=detector.model_fn,
                                        model_dir=ckpt_dir)

    results = classifier.predict(dataset.get_dataset, yield_single_examples=False)
    save_wider_result(output_dir, dataset.imgs, results)

if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    ta = time.time()

    tracemalloc.start()

    eval_on_wider()
    snapshot = tracemalloc.take_snapshot()
    tb = time.time()
    print('total_time: {}'.format(tb-ta))
