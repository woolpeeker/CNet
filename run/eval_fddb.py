import tensorflow as tf
import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import time

from models.detector import Detector, RefineDetector

fddb_imgs_file = '/home/luojiapeng/datasets/fddb/imgList.txt'
fddb_base_dir = '/home/luojiapeng/datasets/fddb'

cfg_path = 'base.json'
cfg = json.load(open(cfg_path))


ckpt_dir = 'checkpoints/fused16_32'
anchor_scales = [16]
anchor_strides = [8]

eval_params = {'match_thres': 0.01,
               'init_scale': 1.,
               'pyramid_scale': 0.5,
               'max_output_size': 400}

def get_fddb_imgs():
    anno_file = open(fddb_imgs_file, 'r')
    fnames = []
    for line in anno_file.readlines():
        line = line.rstrip()
        if line:
            fnames.append(line)
    return fnames

def get_fddb_dataset():
    fnames = get_fddb_imgs()
    fnames = [os.path.join(fddb_base_dir, x)+'.jpg' for x in fnames]
    dataset = tf.data.Dataset.from_tensor_slices((fnames,))
    def fn(fname):
        raw_img = tf.read_file(fname)
        image = tf.image.decode_jpeg(raw_img, channels=3)
        image = tf.cast(image, tf.float32)
        return image
    dataset = dataset.map(fn, num_parallel_calls=8)
    dataset = dataset.batch(1, drop_remainder=True)
    dataset.prefetch(None)
    return dataset

def save_fddb_result(outFile, fnames, result):
    with open(outFile,'w') as fid:
        i=0
        for fname in fnames:
            i += 1
            res = next(result)
            bboxes, scores = res['bboxes'], res['scores']
            bboxes, scores = bboxes[0], scores[0]
            assert len(bboxes) == len(scores)

            fid.write(fname+'\n')
            if bboxes is None:
                fid.write(str(1) + '\n')
                fid.write('%d %d %d %d %.8f\n' % (0, 0, 0, 0, 0))
                continue
            else:
                fid.write(str(len(bboxes)) + '\n')
                for _i in range(len(scores)):
                    s, b =scores[_i], bboxes[_i]
                    b=[int(np.round(x)) for x in b]
                    fid.write('%d %d %d %d %.8f\n' % (b[1], b[0], b[3] - b[1], b[2] - b[0], s))
            if i % 100 == 0 and i:
                print(i)
        fid.close()

def eval_on_fddb():
    outFile = os.path.join(ckpt_dir, 'detectBboxes.txt')
    detector = Detector(anchor_scales=anchor_scales,
                        anchor_strides=anchor_strides,
                        eval_params=eval_params)

    classifier = tf.estimator.Estimator(model_fn=detector.model_fn,
                                        model_dir=ckpt_dir)

    results = classifier.predict(get_fddb_dataset, yield_single_examples=False)
    fnames = get_fddb_imgs()
    save_fddb_result(outFile, fnames, results)

if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    ta = time.time()
    eval_on_fddb()
    tb = time.time()
    print('total_time: {}'.format(tb-ta))