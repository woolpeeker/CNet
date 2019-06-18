import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

import os, re, time

import tensorflow as tf
import tf_extend as tfe
import numpy as np

fddb_imgs_file = '/home/luojiapeng/datasets/fddb/imgList.txt'
fddb_base_dir = '/home/luojiapeng/datasets/fddb'

#fddb_imgs_file = 'D:/Project/TinyFace/Dataset/FDDB/imgList.txt'
#fddb_base_dir = 'D:/Project/TinyFace/Dataset/FDDB'

from nets.cascadeOut import CascadeOut
net = CascadeOut()

model_dir = 'checkpoints/212-1211-m2'
output_dir = os.path.join(model_dir, 'fddb_eval_result')
params = {'thres': 0.01,
          'init_scale': 1}

def get_fddb_imgs():
    anno_file = open(fddb_imgs_file, 'r')
    fnames = []
    for line in anno_file.readlines():
        line = line.rstrip()
        if line:
            fnames.append(line)
    return fnames

def get_fddb():
    fnames = get_fddb_imgs()
    fnames = [os.path.join(fddb_base_dir, x)+'.jpg' for x in fnames]
    dataset = tf.data.Dataset.from_tensor_slices((fnames,))
    def fn(fname):
        raw_img = tf.read_file(fname)
        image = tf.image.decode_jpeg(raw_img, channels=3)
        image = tf.cast(image, tf.float32)
        return {'fname': fname, 'image': image}
    dataset = dataset.map(fn, num_parallel_calls=8)
    dataset.prefetch(None)
    return dataset

def save_fddb_result(outFile, fnames, result):
    with open(outFile,'w') as fid:
        i=0
        for fname in fnames:
            i += 1
            res = next(result)
            bboxes, scores = res['bboxes'], res['scores']
            assert len(bboxes) == len(scores)

            fid.write(fname+'\n')
            if bboxes is None:
                fid.write(str(1) + '\n')
                fid.write('%d %d %d %d %f\n' % (0, 0, 0, 0, 0.01))
                continue
            else:
                fid.write(str(len(bboxes)) + '\n')
                for _i in range(len(scores)):
                    s, b =scores[_i], bboxes[_i]
                    b=[int(np.round(x)) for x in b]
                    fid.write('%d %d %d %d %.3f\n' % (b[1], b[0], b[3] - b[1], b[2] - b[0], s))
            if i % 100 == 0 and i:
                print(i)
        fid.close()

def eval_on_fddb():
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    outFile = os.path.join(output_dir, 'detectBboxes.txt')
    runConfig = tf.estimator.RunConfig(model_dir=model_dir)
    classifier = tf.estimator.Estimator(model_fn=net.model_fn,
                                        params=params,
                                        config=runConfig)
    results = classifier.predict(get_fddb)
    fnames = get_fddb_imgs()
    save_fddb_result(outFile, fnames, results)

if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    ta = time.time()
    eval_on_fddb()
    tb = time.time()
    print('total_time: {}'.format(tb-ta))