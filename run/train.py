
import os, json, random

import tensorflow as tf
from tensorflow.contrib.distribute import MirroredStrategy

from datasets.widerface import WiderTrainTfrecord
from models.detector import Detector
import utils as ut

tfrecord_path = 'datasets/wider_train_cropped320_scaleTo16.32_repeat100.tfrecord'
cfg_path = 'base.json'
ckpt_dir = 'checkpoints/fused16_32'
anchor_scales = [16, 32]
anchor_strides = [8, 8]

def train():
    cfg = json.load(open(cfg_path))
    enable_cuda = cfg['train']['enable_cuda']
    strategy = None
    if enable_cuda:
        gpus = cfg['train']['GPUS']
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in gpus])
        if len(gpus) > 1:
            strategy = MirroredStrategy(num_gpus=len(gpus))
        else:
            strategy = None
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""

    os.makedirs(ckpt_dir, exist_ok=True)
    tf.logging.info('checkpoint path: %s' % ckpt_dir)

    runConfig = tf.estimator.RunConfig(model_dir=ckpt_dir,
                                       train_distribute=strategy,
                                       save_checkpoints_steps=cfg['train']['save_checkpoints_steps'],
                                       save_summary_steps=cfg['train']['save_summary_steps'])
    train_params = {
        'train_image_shape': cfg['train']['image_shape'],
        'cls_match_thres': 0.35,
        'cls_unmatch_thres': 0.3,
        'reg_match_thres': 0.3
    }
    detector = Detector(anchor_scales=anchor_scales,
                        anchor_strides=anchor_strides,
                        train_params=train_params)
    estimator = tf.estimator.Estimator(model_fn=detector.model_fn,
                                       config=runConfig)

    #=================dataset==================================
    def transform_fn(sample):
        image = sample['image']
        s = train_params['train_image_shape']
        image.set_shape([s, s, 3])
        boxes = tf.reshape(sample['boxes'], [-1, 4])
        labels = sample['labels']
        anchor_obj = detector.create_anchor_fn()
        cls_t, cls_w, reg_t, reg_w = anchor_obj.encode(boxes)
        sample =  {'image':image, 'boxes':boxes, 'labels':labels,
                   'cls_t': cls_t, 'cls_w': cls_w,
                   'reg_t': reg_t, 'reg_w': reg_w}
        sample = ut.tf_ops.convert_sample(sample)
        return sample

    dataset = WiderTrainTfrecord(tfrecord_path=tfrecord_path)
    estimator.train(lambda: dataset.get_dataset(transform_fn=transform_fn,
                                                batch_size=cfg['train']['batch_size'],
                                                n_thread=cfg['train']['data_loader_threads'],
                                                prefetch_num=2000,
                                                shuffle_num=2000),
                    max_steps=cfg['train']['max_steps'])

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    train()
