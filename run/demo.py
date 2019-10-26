import tensorflow as tf
import os, json
from PIL import Image
import utils as ut
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from models.detector import Detector

cfg_path = 'base.json'
cfg = json.load(open(cfg_path))

img_dir = 'imgs'
out_dir = 'output'
ckpt_dir = 'checkpoints/lightweight'
anchor_scales = [16, 32]
anchor_strides = [8, 8]

eval_params = {'match_thres': 0.8,
               'init_scale': 1.,
               'pyramid_scale': 0.5,
               'max_output_size': 800,
               'enable_flip': True,
               'fix_size': False}

os.makedirs(out_dir, exist_ok=True)

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
        return image
    dataset = dataset.map(fn, num_parallel_calls=8)
    dataset = dataset.batch(1, drop_remainder=True)
    dataset.prefetch(None)
    return dataset

def save_result(fnames, result):
    i=0
    for fname in fnames:
        i += 1
        out_fname = os.path.split(fname)[-1]
        out_fname = os.path.join(out_dir, out_fname)
        res = next(result)
        boxes, scores = res['bboxes'], res['scores']
        boxes = boxes[0]
        scores = scores[0]
        im = Image.open(fname)
        im = ut.np_ops.draw_boxes(im, boxes, width=2)
        print('output: {}\t{}'.format(out_fname, len(boxes)))
        im.save(out_fname)

def detect():
    detector = Detector(anchor_scales=anchor_scales,
                        anchor_strides=anchor_strides,
                        eval_params=eval_params)
    classifier = tf.estimator.Estimator(model_fn=detector.model_fn,
                                        model_dir=ckpt_dir)
    results = classifier.predict(get_dataset, yield_single_examples=False)
    fnames = get_imgs()
    save_result(fnames, results)

if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    detect()
