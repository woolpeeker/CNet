
import contextlib
import glob
import re

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.ops import gen_image_ops

__all__ = ['convert_bboxes_to_float', # boxes ops
           'convert_bboxes_to_int',
           'filter_zero_boxes',

           'img_shape', # image ops
           'image_pyramid',
           'nms',
           'nms_batch',

           'get_shape', # tensor ops
           'pad_axis',
           'batch_pad',
           'batch_mask',

           'set_arg_scope', # misc
           'lr_plateau_decay',

           'ImageBoxTfrecord', # tfrecords
           'inspect_vars_in_ckpt'
           ]


# **********************************************************************************
#                                  boxes ops
# **********************************************************************************
def convert_bboxes_to_float(bboxes, image_shape):
    image_shape = tf.cast(image_shape, tf.float32)
    bboxes = tf.cast(bboxes, tf.float32)
    bboxes = [bboxes[..., 0] / image_shape[0],
              bboxes[..., 1] / image_shape[1],
              bboxes[..., 2] / image_shape[0],
              bboxes[..., 3] / image_shape[1]]
    bboxes = tf.stack(bboxes,axis=-1)
    bboxes = tf.cast(bboxes, dtype=tf.float32)
    return bboxes

def convert_bboxes_to_int(bboxes, image_shape):
    image_shape = tf.to_float(image_shape)
    bboxes = tf.to_float(bboxes)
    bboxes = [bboxes[..., 0] * image_shape[0],
              bboxes[..., 1] * image_shape[1],
              bboxes[..., 2] * image_shape[0],
              bboxes[..., 3] * image_shape[1]]
    bboxes = tf.stack(bboxes, axis=-1)
    return bboxes

def filter_zero_boxes(boxes, size_thres=0):
    h = boxes[..., 2] - boxes[..., 0]
    w = boxes[..., 3] - boxes[..., 1]
    mask = tf.logical_and(h>size_thres, w>size_thres)
    boxes = tf.boolean_mask(boxes, mask)
    return boxes

def flip_boxes(boxes, image_shape=None):
    if image_shape is not None:
        boxes = convert_bboxes_to_float(boxes, image_shape)
    ymin = boxes[..., 0]
    xmin = boxes[..., 1]
    ymax = boxes[..., 2]
    xmax = boxes[..., 3]
    flipped = [ymin, 1-xmax, ymax, 1-xmin]
    flipped = tf.stack(flipped, axis=-1)
    if image_shape is not None:
        flipped = convert_bboxes_to_int(boxes, image_shape)
    return flipped


# **********************************************************************************
#                                  image ops
# **********************************************************************************
def img_shape(img, dtype=tf.int32, scope='img_shape'):
    with tf.name_scope(scope):
        with tf.control_dependencies([tf.assert_rank_in(img, (3, 4))]):
            result = tf.cond(pred = tf.equal(tf.rank(img),3),
                             true_fn=lambda: tf.shape(img)[:2],
                             false_fn=lambda: tf.shape(img)[1:3])
            return tf.cast(result, dtype)

def image_pyramid(image, scale=0.5, min_len=32, divisible=16, init_scale=1., scope='image_pyramid'):
    with tf.name_scope(scope):
        with tf.control_dependencies([tf.assert_rank(image, 4)]):
            image = tf.identity(image)
        batch_size = image.shape[0]
        im_arr=tf.TensorArray(tf.float32, size=0, dynamic_size=True, element_shape=[batch_size, None,None,3],
                              infer_shape=False, clear_after_read=False)
        divisible = tf.to_float(divisible)
        init_shape = init_scale*img_shape(image,tf.float32)
        init_shape = tf.ceil(init_shape / divisible) * divisible
        def body(i, im_arr, target_shape):
            target_shape = tf.to_float(target_shape)
            target_shape = tf.ceil(target_shape / divisible) * divisible
            scaled_im = tf.image.resize_bilinear(image,
                                                 tf.to_int32(target_shape),
                                                 align_corners=True)
            im_arr = im_arr.write(i,scaled_im)
            i += 1
            target_shape = target_shape * scale
            return i, im_arr, target_shape

        i, im_arr, _ = tf.while_loop(loop_vars=[0, im_arr, init_shape],
                                  cond=lambda i,im_arr,ts: tf.reduce_min(ts)>min_len,
                                  body=body,
                                  parallel_iterations=4)
    return im_arr


def nms(boxes,
        scores,
        max_output_size,
        nms_thres=0.5,
        score_thres=float('-inf'),
        scope='nms'):
    with tf.name_scope(scope):
        iou_threshold = tf.convert_to_tensor(nms_thres, name='iou_threshold')
        score_threshold = tf.convert_to_tensor(
            score_thres, name='score_threshold')
        score_mask = scores>score_threshold
        boxes = tf.boolean_mask(boxes, score_mask)
        scores = tf.boolean_mask(scores, score_mask)
        indices = tf.reshape(tf.where(score_mask), [-1])
        nms_indices = gen_image_ops.non_max_suppression_v2(boxes, scores, max_output_size,
                                                           iou_threshold)
        indices = tf.gather(indices, nms_indices)
        return indices


def nms_batch(batch_boxes,
              batch_scores,
              max_output_size,
              nms_thres=0.5,
              score_thres=float('-inf'),
              pad=True,
              scope='nms_batch'):
    with tf.name_scope(scope):
        batch_size = batch_boxes.shape[0].value
        nms_boxes = []
        nms_scores = []
        for i in range(batch_size):
            idx = nms(boxes=batch_boxes[i], scores=batch_scores[i],
                      max_output_size=max_output_size,
                      nms_thres=nms_thres,
                      score_thres=score_thres)
            boxes_i = tf.gather(batch_boxes[i], idx)
            scores_i = tf.gather(batch_scores[i], idx)
            nms_boxes.append(boxes_i)
            nms_scores.append(scores_i)
        if pad:
            nms_boxes = batch_pad(nms_boxes)
            nms_scores = batch_pad(nms_scores)
        return nms_boxes, nms_scores


# **********************************************************************************
#                                  tensors ops
# **********************************************************************************
def get_shape(x, rank=None):
    """Returns the dimensions of a Tensor as list of integers or scale tensors.

    Args:
      x: N-d Tensor;
      rank: Rank of the Tensor. If None, will try to guess it.
    Returns:
      A list of `[d1, d2, ..., dN]` corresponding to the dimensions of the
        input tensor.  Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape()
        if rank is None:
            static_shape = static_shape.as_list()
            rank = len(static_shape)
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def pad_axis(x, offset, size, axis=0, name=None):
    """Pad a tensor on an axis, with a given offset and output size.
    The tensor is padded with zero (i.e. CONSTANT mode). Note that the if the
    `size` is smaller than existing size + `offset`, the output tensor
    was the latter dimension.

    Args:
      x: Tensor to pad;
      offset: Offset to add on the dimension chosen;
      size: Final size of the dimension.
    Return:
      Padded tensor whose dimension on `axis` is `size`, or greater if
      the input vector was larger.
    """
    with tf.name_scope(name, 'pad_axis'):
        shape = get_shape(x)
        rank = len(shape)
        # Padding description.
        new_size = tf.maximum(size-offset-shape[axis], 0)
        pad1 = tf.stack([0]*axis + [offset] + [0]*(rank-axis-1))
        pad2 = tf.stack([0]*axis + [new_size] + [0]*(rank-axis-1))
        paddings = tf.stack([pad1, pad2], axis=1)
        x = tf.pad(x, paddings, mode='CONSTANT')
        # Reshape, to get fully defined shape if possible.
        # TODO: fix with tf.slice
        shape[axis] = size
        x = tf.reshape(x, tf.stack(shape))
        return x


def batch_pad(tensors, scope='batch_pad'):
    """
    pad a list of tensors to same shape and stack together
    :param tensors: a tf tensor
    :return: a batched tensor
    """
    with tf.name_scope(scope):
        shapes = [tf.shape(x)[0] for x in tensors]
        max_shape = tf.reduce_max(tf.stack(shapes))
        padded_tensors = [pad_axis(x, 0, max_shape) for x in tensors]
        batched_tensors = tf.stack(padded_tensors)
        return batched_tensors


def batch_mask(batch_tensor, mask, is_mask_batched=True, axis=None, scope='batch_mask'):
    with tf.name_scope(scope):
        result = []
        for i in range(batch_tensor.shape[0]):
            m = mask[i] if is_mask_batched else mask
            res = tf.boolean_mask(batch_tensor[i], m, axis=axis)
            result.append(res)
        return result


# **********************************************************************************
#                                  misc
# **********************************************************************************
@contextlib.contextmanager
def set_arg_scope(defaults):
    """Sets arg scope defaults for all items present in defaults.
    Args:
      defaults: dictionary/list of pairs, containing a mapping from
      function to a dictionary of default args.
    Yields:
      context manager where all defaults are set.
    """
    if hasattr(defaults, 'items'):
        items = list(defaults.items())
    else:
        items = defaults
    if not items:
        yield
    else:
        func, default_arg = items[0]
        with slim.arg_scope(func, **default_arg):
            with set_arg_scope(items[1:]):
                yield


def lr_plateau_decay(lr=0.01,decay=0.9999, min_lr=1e-6, loss=None, scope='lr_plateau_decay'):
    with tf.name_scope(scope):
        his_len = 10
        local_lr = tf.get_local_variable(name='local_lr', dtype=tf.float32,
                                         initializer=tf.constant(lr, dtype=tf.float32))
        loss_idx = tf.get_local_variable(name='loss_idx', dtype=tf.int32,
                                         initializer=tf.constant(1,dtype=tf.int32))
        his_loss = tf.get_local_variable(name='history_loss',dtype=tf.float32,
                                             initializer=tf.zeros([his_len])-1.0)
        if loss is None:
            loss = tf.losses.get_total_loss()
        def true_fn():
            update_history = tf.assign(his_loss[loss_idx], loss)
            with tf.control_dependencies([update_history]):
                update_idx = tf.assign(loss_idx, tf.mod(loss_idx + 1, his_len))
            with tf.control_dependencies([update_idx]):
                updated_lr = tf.cond(pred=loss>tf.reduce_mean(his_loss),
                                     true_fn=lambda: tf.assign(local_lr, local_lr*decay),
                                     false_fn=lambda: local_lr)
            lr = tf.maximum(updated_lr, min_lr)
            return lr
        lr = tf.cond(pred=tf.equal(tf.mod(tf.train.get_global_step(), 100), 0),
                     true_fn=true_fn,
                     false_fn=lambda: tf.identity(local_lr))
        tf.summary.scalar('lr_plateau_decay', lr)
    return lr


# **********************************************************************************
#                                  tfrecord
# **********************************************************************************
def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class ImageBoxTfrecord:
    def __init__(self):
        self.tf_writer = None
    def write(self, fname, samples):
        if self.tf_writer is None:
            self.tf_writer = tf.python_io.TFRecordWriter(fname)
        for sample in samples:
            example = self._convert_to_example(sample)
            self.tf_writer.write(example.SerializeToString())

    def _convert_to_example(self, sample):
        image = sample['image']
        boxes = sample['boxes']
        labels = sample['labels']
        if len(boxes) != len(labels):
            raise ValueError('length of boxes and labels is not equal.')
        boxes = boxes.reshape([-1]).tolist()
        labels = labels.tolist()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': bytes_feature(image),
            'boxes': float_feature(boxes),
            'labels': float_feature(labels)
        }))
        return example

    def get_dataset(self, fname, n_thread=8):
        fnames = glob.glob(fname)
        if not fnames:
            raise Exception('%s do not exist' % fnames)
        dataset = tf.data.TFRecordDataset(fnames, num_parallel_reads=n_thread)
        dataset = dataset.map(self.parser, num_parallel_calls=n_thread)
        return dataset

    def parser(self, record):
        keys_to_features = {'image': tf.FixedLenFeature((), tf.string),
                            'boxes': tf.VarLenFeature(dtype=tf.float32),
                            'labels': tf.VarLenFeature(dtype=tf.float32)}
        features = tf.parse_single_example(record, features=keys_to_features)
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image = tf.to_float(image)
        boxes = tf.sparse_tensor_to_dense(features['boxes'])
        boxes = tf.reshape(boxes, [-1, 4])
        labels = tf.sparse_tensor_to_dense(features['labels'])
        return {'image':image, 'boxes':boxes, 'labels':labels}

def convert_sample(sample):
    """convert list in sample to tensor
    :param sample: A dict of Tensor or List of tensor
    :return:
    a dict of tensor
    """
    result = {}
    for key,value in sample.items():
        if isinstance(value, list):
            for i,v in enumerate(value):
                result[key+'__%din%d'%(i,len(value))] = v
        else:
            result[key] = value
    return result

def reconvert_sample(sample):
    """reverse function of convert_sample"""
    result={}
    for key,value in sample.items():
        matchRes = re.match('(.*)__(\d+)in(\d+)$',key)
        if not matchRes:
            result[key]=value
        else:
            name, i, length = matchRes.group(1,2,3)
            i, length = int(i), int(length)
            result.setdefault(name,[None]*length)[i]=value
    return result


def inspect_vars_in_ckpt(file_name):
    varlist=[]
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
      varlist.append(key)
    return varlist