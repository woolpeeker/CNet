
import tensorflow as tf

ModeKeys = tf.estimator.ModeKeys

import numpy as np

from models.net import AlphaNet, HalfAlphaNet, PyramidFusedNet
from models.module import AlphaAnchor, SingleAnchor

import utils as ut

class Detector:
    def __init__(self, anchor_scales,
                 anchor_strides,
                 train_params=None,
                 eval_params=None):
        self.train_params = train_params
        self.eval_params = eval_params
        self.anchor_scales = anchor_scales
        self.anchor_strides = anchor_strides
        self.net = PyramidFusedNet(num_anchors=len(self.anchor_scales))
        self.create_anchor_fn = lambda: AlphaAnchor(image_shape=self.train_params['train_image_shape'],
                                                    anchor_scales=self.anchor_scales,
                                                    anchor_strides=self.anchor_strides,
                                                    cls_match_thres=self.train_params['cls_match_thres'],
                                                    cls_unmatch_thres=self.train_params['cls_unmatch_thres'],
                                                    reg_match_thres=self.train_params['reg_match_thres'])

    def model_fn(self, features, labels, mode):
        if mode == ModeKeys.PREDICT:
            return self.model_predict_fn(mode, features)
        else:
            raise Exception('Training code will release soon.')

    
    def _decode(self, anchor_obj, batch_codes, batch_scores, max_output_size=100, match_thres=0.5, scope='detector_decode'):
        with tf.name_scope(scope):
            batch_size = batch_codes[0].shape[0].value
            anchor_num = len(self.anchor_scales)
            reshaped_codes = [tf.reshape(batch_codes[i], [batch_size, -1, 4])
                              for i in range(anchor_num)]
            reshaped_scores = [tf.reshape(batch_scores[i], [batch_size, -1])
                               for i in range(anchor_num)]
            batch_boxes = anchor_obj.batch_decode(reshaped_codes)
            batch_scores = tf.concat(reshaped_scores, axis=1)
            batch_boxes, batch_scores = ut.tf_ops.nms_batch(batch_boxes, batch_scores,
                                                            max_output_size=max_output_size, nms_thres=0.4,
                                                            score_thres=match_thres, pad=True)
            return batch_boxes, batch_scores

    def loss_fn(self, ground_truth, predictions):
        cls_targets, cls_weights, reg_targets, reg_weights = ground_truth
        log_predicts, cls_predicts, reg_predicts = predictions
        with tf.name_scope('loss_fn'):
            cls_losses = []
            loc_losses = []
            for i in range(len(self.anchor_strides)):
                cls_t, cls_w = cls_targets[i], cls_weights[i]
                reg_t, reg_w = reg_targets[i], reg_weights[i]
                log_p, reg_p = log_predicts[i], reg_predicts[i]
                log_p = tf.reshape(log_p, [log_p.shape[0], -1, log_p.shape[-1]])
                cls_loss = tf.losses.sparse_softmax_cross_entropy(labels=cls_t, logits=log_p,
                                                                  weights=cls_w,
                                                                  reduction=tf.losses.Reduction.SUM)
                reg_p = tf.reshape(reg_p, [reg_p.shape[0], -1, reg_p.shape[-1]])
                loc_loss = tf.losses.mean_squared_error(labels=reg_t, predictions=reg_p,
                                                        weights=tf.expand_dims(reg_w, -1),
                                                        reduction=tf.losses.Reduction.SUM)
                cls_losses.append(cls_loss)
                loc_losses.append(loc_loss)
                tf.summary.scalar('cls_loss_%d' % i, cls_loss, family='loss')
                tf.summary.scalar('loc_loss_%d' % i, loc_loss, family='loss')
            cls_losses = sum(cls_losses)
            loc_losses = sum(loc_losses)
            total_loss = cls_losses + loc_losses
            tf.summary.scalar('clsLoss', cls_losses)
            tf.summary.scalar('locLoss', loc_losses)
            tf.summary.scalar('totalLoss', total_loss)
            return total_loss


    def model_predict_fn(self, mode, features):
        images = features
        match_thres = self.eval_params['match_thres']
        init_scale = self.eval_params['init_scale']
        pyramid_scale = self.eval_params['pyramid_scale']
        max_output_size = self.eval_params['max_output_size']
        enable_flip = self.eval_params['enable_flip']
        batch_size = images.shape[0]


        boxes, scores = self._predict_fn(images, match_thres, init_scale, pyramid_scale, max_output_size, batch_size)
        if enable_flip:
            flipped_images = tf.image.flip_left_right(images)
            flipped_boxes, flipped_scores = self._predict_fn(flipped_images, match_thres, init_scale, pyramid_scale, max_output_size, batch_size)
            flipped_boxes = ut.tf_ops.flip_boxes(flipped_boxes)

            bboxes_all = tf.concat([boxes, flipped_boxes], axis=1)
            scores_all = tf.concat([scores, flipped_scores], axis=1)
        else:
            bboxes_all = boxes
            scores_all = scores
        nms_boxes, nms_scores = ut.tf_ops.nms_batch(bboxes_all, scores_all,
                                                    max_output_size=max_output_size,
                                                    nms_thres=0.4,
                                                    score_thres=match_thres)
        nms_boxes = ut.tf_ops.convert_bboxes_to_int(nms_boxes, tf.shape(images)[1:3])
        samples = {'bboxes': nms_boxes, 'scores': nms_scores}
        return tf.estimator.EstimatorSpec(mode, samples)


    def _predict_fn(self, images, match_thres, init_scale, pyramid_scale, max_output_size, batch_size):
        im_arr = ut.tf_ops.image_pyramid(images, scale=pyramid_scale, min_len=16, divisible=16,
                                         init_scale=float(init_scale))

        def body(i, boxes_all, scores_all, im_arr=im_arr):
            im = im_arr.read(i)
            resized_shape = tf.to_int32(ut.tf_ops.img_shape(im) / 2)
            im_resized = tf.image.resize_bilinear(im, size=resized_shape, align_corners=True)
            anchor0 = SingleAnchor(image_shape=ut.tf_ops.img_shape(im),
                                   anchor_scale=16,
                                   anchor_stride=8)
            anchor1 = SingleAnchor(image_shape=ut.tf_ops.img_shape(im),
                                   anchor_scale=32,
                                   anchor_stride=8)

            def first_level():
                fem1, _ = self.net.fem(im)
                _, fem2 = self.net.fem(im_resized)
                feature = tf.concat([fem1, fem2], axis=-1)
                pred0 = self.net.dem(feature, 0, is_training=False)
                pred1 = self.net.dem(feature, 1, is_training=False)
                boxes0, scores0 = anchor0.batch_decode(pred0[2], pred0[1], max_out=max_output_size, thres=match_thres)
                boxes1, scores1 = anchor1.batch_decode(pred1[2], pred1[1], max_out=max_output_size, thres=match_thres)
                boxes = tf.concat([boxes0, boxes1], axis=1)
                scores = tf.concat([scores0, scores1], axis=1)
                return boxes, scores

            def tail_level():
                fem1, _ = self.net.fem(im)
                _, fem2 = self.net.fem(im_resized)
                feature = tf.concat([fem1, fem2], axis=-1)
                logit, cls, reg = self.net.dem(feature, 1)
                boxes, scores = anchor1.batch_decode(reg, cls, max_out=max_output_size, thres=match_thres)
                return boxes, scores

            boxes, scores = tf.cond(tf.equal(i, 0),
                                    true_fn=first_level,
                                    false_fn=tail_level)
            boxes_float = ut.tf_ops.convert_bboxes_to_float(boxes, ut.tf_ops.img_shape(im))
            boxes_all = tf.concat([boxes_all, boxes_float], axis=1)
            scores_all = tf.concat([scores_all, scores], axis=1)
            return i + 1, boxes_all, scores_all

        i = 0
        bboxes_all = tf.zeros([batch_size, 0, 4])
        scores_all = tf.zeros([batch_size, 0])
        shape_invariant = [tf.TensorShape([]),
                           tf.TensorShape([batch_size, None, 4]),
                           tf.TensorShape([batch_size, None])]
        i, bboxes_all, scores_all = \
            tf.while_loop(cond=lambda i, _s, _b: i + 1 < im_arr.size(),
                          loop_vars=[i, bboxes_all, scores_all],
                          body=body,
                          parallel_iterations=8,
                          shape_invariants=shape_invariant,
                          back_prop=False)
        nms_boxes, nms_scores = ut.tf_ops.nms_batch(bboxes_all, scores_all,
                                                    max_output_size=max_output_size,
                                                    nms_thres=0.4,
                                                    score_thres=match_thres)
        return nms_boxes, nms_scores

    def _predict_fn_fix_size(self, images, match_thres, init_scale, pyramid_scale, max_output_size, batch_size):
        im_arr = ut.tf_ops.image_pyramid(images, scale=pyramid_scale, min_len=32, divisible=8,
                                         init_scale=float(init_scale))
        fem1_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True, element_shape=[batch_size, None, None, 64],
                                  clear_after_read=False, infer_shape=False)
        fem2_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True, element_shape=[batch_size, None, None, 64],
                                  clear_after_read=False, infer_shape=False)

        def get_fems(i, fem1_arr, fem2_arr, im_arr=im_arr):
            im = im_arr.read(i)
            fem_a, fem_b = self.net.fem(im, is_training=False)
            fem1_arr = fem1_arr.write(i, fem_a)
            fem2_arr = fem2_arr.write(i, fem_b)
            return i + 1, fem1_arr, fem2_arr

        i, fem1_arr, fem2_arr = tf.while_loop(loop_vars=[0, fem1_arr, fem2_arr],
                                              cond=lambda i, _1, _2: i < im_arr.size(),
                                              body=get_fems,
                                              parallel_iterations=8,
                                              back_prop=False)

        def body(i, boxes_all, scores_all, im_arr=im_arr):
            im = im_arr.read(i)
            anchor0 = SingleAnchor(image_shape=ut.tf_ops.img_shape(im),
                                   anchor_scale=16,
                                   anchor_stride=8)
            anchor1 = SingleAnchor(image_shape=ut.tf_ops.img_shape(im),
                                   anchor_scale=32,
                                   anchor_stride=8)

            def first_level():
                fem1 = fem1_arr.read(i)
                fem2 = fem2_arr.read(i + 1)
                fem2 = tf.image.resize_bilinear(fem2, size=ut.tf_ops.img_shape(fem1), align_corners=True)
                feature = tf.concat([fem1, fem2], axis=-1)
                pred0 = self.net.dem(feature, 0, is_training=False)
                pred1 = self.net.dem(feature, 1, is_training=False)
                boxes0, scores0 = anchor0.batch_decode(pred0[2], pred0[1], max_out=max_output_size, thres=match_thres)
                boxes1, scores1 = anchor1.batch_decode(pred1[2], pred1[1], max_out=max_output_size, thres=match_thres)
                boxes = tf.concat([boxes0, boxes1], axis=1)
                scores = tf.concat([scores0, scores1], axis=1)
                return boxes, scores

            def tail_level():
                fem1 = fem1_arr.read(i)
                fem2 = fem2_arr.read(i + 1)
                feature = tf.concat([fem1, fem2], axis=-1)
                logit, cls, reg = self.net.dem(feature, 1)
                boxes, scores = anchor1.batch_decode(reg, cls, max_out=max_output_size, thres=match_thres)
                return boxes, scores

            boxes, scores = tf.cond(tf.equal(i, 0),
                                    true_fn=first_level,
                                    false_fn=tail_level)
            boxes_float = ut.tf_ops.convert_bboxes_to_float(boxes, ut.tf_ops.img_shape(im))
            boxes_all = tf.concat([boxes_all, boxes_float], axis=1)
            scores_all = tf.concat([scores_all, scores], axis=1)
            return i + 1, boxes_all, scores_all

        i = 0
        bboxes_all = tf.zeros([batch_size, 0, 4])
        scores_all = tf.zeros([batch_size, 0])
        shape_invariant = [tf.TensorShape([]),
                           tf.TensorShape([batch_size, None, 4]),
                           tf.TensorShape([batch_size, None])]
        i, bboxes_all, scores_all = \
            tf.while_loop(cond=lambda i, _s, _b: i + 1 < im_arr.size(),
                          loop_vars=[i, bboxes_all, scores_all],
                          body=body,
                          parallel_iterations=16,
                          shape_invariants=shape_invariant,
                          back_prop=False)
        nms_boxes, nms_scores = ut.tf_ops.nms_batch(bboxes_all, scores_all,
                                                    max_output_size=max_output_size,
                                                    nms_thres=0.4,
                                                    score_thres=match_thres)
        nms_boxes = ut.tf_ops.convert_bboxes_to_int(nms_boxes, tf.shape(images)[1:3])
        return nms_boxes, nms_scores

    def model_predict_fn_fix_size(self, mode, features):
        images = features
        match_thres = self.eval_params['match_thres']
        init_scale = self.eval_params['init_scale']
        pyramid_scale = self.eval_params['pyramid_scale']
        max_output_size = self.eval_params['max_output_size']
        batch_size = images.shape[0]

        im_arr = ut.tf_ops.image_pyramid(images, scale=pyramid_scale, min_len=32, divisible=16, init_scale=float(init_scale))
        fem1_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True, element_shape=[batch_size, None, None, 64],
                                  clear_after_read=False, infer_shape=False)
        fem2_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True, element_shape=[batch_size, None, None, 64],
                                  clear_after_read=False, infer_shape=False)

        def get_fems(i, fem1_arr, fem2_arr, im_arr=im_arr):
            im = im_arr.read(i)
            fem_a, fem_b = self.net.fem(im, is_training=False)
            fem1_arr = fem1_arr.write(i, fem_a)
            fem2_arr = fem2_arr.write(i, fem_b)
            return i + 1, fem1_arr, fem2_arr

        i, fem1_arr, fem2_arr = tf.while_loop(loop_vars=[0, fem1_arr, fem2_arr],
                                              cond=lambda i, _1, _2: i < im_arr.size(),
                                              body=get_fems,
                                              parallel_iterations=8,
                                              back_prop=False)
        def body(i, boxes_all, scores_all, im_arr=im_arr):
            im = im_arr.read(i)
            anchor0 = SingleAnchor(image_shape=ut.tf_ops.img_shape(im),
                                   anchor_scale=16,
                                   anchor_stride=8)
            anchor1 = SingleAnchor(image_shape=ut.tf_ops.img_shape(im),
                                   anchor_scale=32,
                                   anchor_stride=8)
            def first_level():
                fem1 = fem1_arr.read(i)
                fem2 = fem2_arr.read(i+1)
                fem2 = tf.image.resize_bilinear(fem2, size=ut.tf_ops.img_shape(fem1), align_corners=True)
                feature = tf.concat([fem1, fem2], axis=-1)
                pred0 = self.net.dem(feature, 0, is_training=False)
                pred1 = self.net.dem(feature, 1, is_training=False)
                boxes0, scores0 = anchor0.batch_decode(pred0[2], pred0[1], max_out=max_output_size, thres=match_thres)
                boxes1, scores1 = anchor1.batch_decode(pred1[2], pred1[1], max_out=max_output_size, thres=match_thres)
                boxes = tf.concat([boxes0, boxes1], axis=1)
                scores = tf.concat([scores0, scores1], axis=1)
                return boxes, scores
            def tail_level():
                fem1 = fem1_arr.read(i)
                fem2 = fem2_arr.read(i+1)
                feature = tf.concat([fem1, fem2], axis=-1)
                logit, cls, reg = self.net.dem(feature, 1)
                boxes, scores = anchor1.batch_decode(reg, cls, max_out=max_output_size, thres=match_thres)
                return boxes, scores

            boxes, scores = tf.cond(tf.equal(i, 0),
                                    true_fn=first_level,
                                    false_fn=tail_level)
            boxes_float = ut.tf_ops.convert_bboxes_to_float(boxes, ut.tf_ops.img_shape(im))
            boxes_all = tf.concat([boxes_all, boxes_float], axis=1)
            scores_all = tf.concat([scores_all, scores], axis=1)
            return i + 1, boxes_all, scores_all

        i = 0
        bboxes_all = tf.zeros([batch_size, 0, 4])
        scores_all = tf.zeros([batch_size, 0])
        shape_invariant = [tf.TensorShape([]),
                           tf.TensorShape([batch_size, None, 4]),
                           tf.TensorShape([batch_size, None])]
        i, bboxes_all, scores_all = \
            tf.while_loop(cond=lambda i, _s, _b: i+1 < im_arr.size(),
                          loop_vars=[i, bboxes_all, scores_all],
                          body=body,
                          parallel_iterations=16,
                          shape_invariants=shape_invariant,
                          back_prop=False)
        nms_boxes, nms_scores = ut.tf_ops.nms_batch(bboxes_all, scores_all,
                                                    max_output_size=max_output_size,
                                                    nms_thres=0.4,
                                                    score_thres=match_thres)
        nms_boxes = ut.tf_ops.convert_bboxes_to_int(nms_boxes, tf.shape(images)[1:3])
        samples = {'bboxes': nms_boxes, 'scores': nms_scores}
        return tf.estimator.EstimatorSpec(mode, samples)
