
import os, io, re
import numpy as np
from PIL import Image
import scipy.io as sio
import utils as ut

import tensorflow as tf

__all__ = ['read_anno',
           'WiderTrain',
           'WiderTrainTfrecord',
           'WiderEval',
           'save_wider_result']

def read_anno(path, min_size, has_label=False):
    # annotation: float [ymin, xmin, ymax, xmax]
    annotation = dict()
    k = ''
    for line in open(path).readlines():
        line = line.rstrip()
        line_split = line.split()
        if re.match('.*.jpg', line):
            k = line
            annotation[k] = {'boxes':[], 'labels':[]}
            continue
        elif len(line_split) >= 4:
            box = np.array([float(x) for x in line_split])
            box = box[[1, 0, 3, 2]]
            if has_label:
                label = int(line_split[-1])
            else:
                label = 1
            # box must larger le min_size
            box[2] = abs(box[2])
            box[3] = abs(box[3])
            mask = np.sqrt(box[2] * box[3]) > min_size
            if mask:
                box[[2, 3]] = box[[0, 1]] + box[[2, 3]]
                annotation[k]['boxes'].append(box.tolist())
                annotation[k]['labels'].append(label)
    for k in list(annotation.keys()):
        if len(annotation[k]['boxes']) == 0:
            del annotation[k]
        else:
            annotation[k]['boxes'] = np.array(annotation[k]['boxes'])
            annotation[k]['labels'] = np.array(annotation[k]['labels'])
    return annotation

class WiderTrain():
    def __init__(self, img_dir, anno_path, min_size=0, has_label=False):
        self.imgs_dir = img_dir
        self.anno_dict = read_anno(anno_path, min_size, has_label)
        self.anno_keys = list(self.anno_dict.keys()) # the fname
        self.anno_values = list(self.anno_dict.values()) # it is the float box
        self.imgs_list = [] # list of io.BytesIO object
        self.loaded = False

    def __len__(self):
        return len(self.anno_dict)

    def __getitem__(self, index, transform_fn=None):
        if not self.loaded:
            self.load()
        im = Image.open(self.imgs_list[index]).convert('RGB')
        boxes = self.anno_values[index]
        if transform_fn is not None:
            im, boxes = transform_fn(im, boxes)
        return im, boxes

    def load(self):
        print('WIDERFACE loading...')
        for fname in self.anno_keys:
            img_path = os.path.join(self.imgs_dir, fname)
            with open(img_path, 'rb') as f:
                raw_data = f.read()
            raw_data = io.BytesIO(raw_data)
            self.imgs_list.append(raw_data)
        self.loaded = True
        print('WIDERFACE loaded.')

    def get_dataset(self,  transform_fn=None, image_shape=None, num_threads=24, batch_size=16):
        if not self.loaded:
            self.load()
        dataset = tf.data.Dataset.range(len(self))
        def _py_func(self, index):
            img, boxes = tf.py_func(func=lambda i: self.__getitem__(i, transform_fn), inp=[index],
                                    Tout=[tf.float32, tf.float32])
            img.set_shape([image_shape, image_shape, 3])
            boxes.set_shape([None, 4])
            return img, boxes
        dataset = dataset.map(_py_func, num_parallel_calls=num_threads)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(len(self)//2)
        dataset = dataset.padded_batch(batch_size, dataset.output_shapes, drop_remainder=True)
        dataset = dataset.prefetch(None)
        return dataset

class WiderTrainTfrecord:
    def __init__(self, tfrecord_path):
        self.tfrecord_path = tfrecord_path

    def get_dataset(self, transform_fn, batch_size, n_thread=16, shuffle_num=20000, prefetch_num=None):
        tfrecord_obj = ut.tf_ops.ImageBoxTfrecord()
        dataset = tfrecord_obj.get_dataset(self.tfrecord_path, n_thread=10)
        if transform_fn is not None:
            dataset = dataset.map(transform_fn, num_parallel_calls=n_thread)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(shuffle_num)
        dataset = dataset.padded_batch(batch_size=batch_size,
                                       padded_shapes=dataset.output_shapes,
                                       drop_remainder=True)
        dataset = dataset.prefetch(prefetch_num)
        return dataset


class WiderEval:
    def __init__(self, imgs_dir, anno_mat):
        self.imgs_dir = imgs_dir
        self.imgs = self._read_mat(anno_mat) # fnames
        self.imgs_list = [] # Pillow Image objects
        self.loaded = False

    def _read_mat(self, anno_mat):
        wider_face = sio.loadmat(anno_mat)
        event_list = wider_face['event_list']
        file_list = wider_face['file_list']
        del wider_face
        fnames = []
        for i, event in enumerate(event_list):
            event = event[0][0]
            for file in file_list[i][0]:
                file = file[0][0] + '.jpg'
                fnames.append(os.path.join(event, file))
        return fnames

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if not self.loaded:
            self.load()
        im = Image.open(self.imgs_list[index]).convert('RGB')
        im = np.array(im).astype(np.float32)
        return im

    def load(self):
        print('WIDERFACE loading...')
        for fname in self.imgs:
            img_path = os.path.join(self.imgs_dir, fname)
            raw_data = open(img_path,'rb').read()
            raw_data = io.BytesIO(raw_data)
            self.imgs_list.append(raw_data)
        self.loaded = True
        print('WIDERFACE loaded.')

    def get_dataset(self):
        if not self.loaded:
            self.load()
        dataset = tf.data.Dataset.range(len(self))
        dataset = dataset.map(self._py_func, num_parallel_calls=4)
        dataset = dataset.batch(1, drop_remainder=True)
        dataset = dataset.prefetch(50)
        return dataset

    def _py_func(self, index):
        img = tf.py_func(func=lambda i: self.__getitem__(i), inp=[index],
                                Tout=tf.float32)
        img.set_shape([None, None, 3])
        return img


def save_wider_result(output_dir, fnames, results):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    current_event = ''
    i=-1
    for res in results:
        i+=1
        fname = fnames[i]
        bboxes, scores = res['bboxes'], res['scores']
        assert len(bboxes) == len(scores)
        assert bboxes.shape[0]==1 and scores.shape[0]==1
        bboxes, scores = bboxes[0], scores[0]
        event = os.path.dirname(fname)
        if current_event != event:
            current_event = event
            save_path = os.path.join(output_dir, current_event)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            print('current path:', current_event)

        out_fname = fname.split('.')[0]
        out_fname = os.path.join(output_dir, out_fname + '.txt')
        fid = open(out_fname, 'w')
        fid.write(os.path.basename(fname) + '\n')
        if bboxes is None:
            fid.write(str(1) + '\n')
            fid.write('%.1f %.1f %.1f %.1f %.1f\n' % (0, 0, 0, 0, 0))
            continue
        else:
            fid.write(str(len(bboxes)) + '\n')
            for _i in range(len(scores)):
                s, b =scores[_i], bboxes[_i]
                fid.write('%.1f %.1f %.1f %.1f %.8f\n' % (b[1], b[0], b[3] - b[1], b[2] - b[0], s))

            fid.close()
            if i % 10 == 0 and i:
                print(i)
    if i+1 != len(fnames):
        raise Exception('length of fnames is not consistent with results')
