
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
        if mode == ModeKeys.TRAIN:
            return self.model_train_fn(mode, features, labels)
        elif mode == ModeKeys.PREDICT:
            return self.model_predict_fn(mode, features)

    def model_train_fn(self, mode, features, labels):
        features = ut.tf_ops.reconvert_sample(features)
        images = features['image']
        boxes = features['boxes']
        cls_t = features['cls_t']
        cls_w = features['cls_w']
        reg_t = features['reg_t']
        reg_w = features['reg_w']
        ground_truth = (cls_t, cls_w, reg_t, reg_w)
        images_shape = ut.tf_ops.img_shape(images)

        images = tf.image.random_saturation(images, lower=1 - 10. / 255, upper=1 + 10. / 255)
        images = tf.image.random_hue(images, 10. / 255)
        images = tf.image.random_brightness(images, 10. / 255)

        #logit, cls, reg
        predictions = self.net.train_fn(images)
        net_size = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        tf.logging.info('net_size: %d' % net_size)

        step = tf.train.get_or_create_global_step()
        loss = self.loss_fn(ground_truth, predictions)

        lr = self._adjust_learning_rate(lr=0.001)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        minimize_op = optimizer.minimize(loss, step)
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_op)

        # summary
        boxes_float = ut.tf_ops.convert_bboxes_to_float(boxes, images_shape)
        images_gt_bboxes = tf.image.draw_bounding_boxes(images, boxes_float)

        anchor_obj = self.create_anchor_fn()
        pred_bboxes, pred_scores  = self._decode(anchor_obj=anchor_obj,
                                                 batch_codes=predictions[2],
                                                 batch_scores=predictions[1])
        pred_bboxes = ut.tf_ops.convert_bboxes_to_float(pred_bboxes, images_shape)
        images_pred_bboxes = tf.image.draw_bounding_boxes(images, pred_bboxes)
        tf.summary.image('images_gt_bboxes', images_gt_bboxes)
        tf.summary.image('images_pred_bboxes', images_pred_bboxes)
        tf.summary.scalar('learning_rate', lr)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    def _adjust_learning_rate(self, lr):
        step = tf.train.get_or_create_global_step()
        decay_stage = int(2e5)
        first_stage = int(2e4)
        lr = tf.case([
            (step < first_stage, lambda: 0.0002),
            (step < decay_stage, lambda: lr),
            (step > 0,
             lambda: tf.train.exponential_decay(lr, global_step=step - decay_stage, decay_steps=10000, decay_rate=0.7))
        ])
        lr = tf.maximum(lr, 1e-7)
        return lr

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
        batch_size = images.shape[0]

        boxes, scores = self._predict_fn(images, match_thres, init_scale, pyramid_scale, max_output_size, batch_size)
        flipped_images = tf.image.flip_left_right(images)
        flipped_boxes, flipped_scores = self._predict_fn(flipped_images, match_thres, init_scale, pyramid_scale, max_output_size, batch_size)
        flipped_boxes = ut.tf_ops.flip_boxes(flipped_boxes)

        bboxes_all = tf.concat([boxes, flipped_boxes], axis=1)
        scores_all = tf.concat([scores, flipped_scores], axis=1)
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
                          parallel_iterations=16,
                          shape_invariants=shape_invariant,
                          back_prop=False)
        nms_boxes, nms_scores = ut.tf_ops.nms_batch(bboxes_all, scores_all,
                                                    max_output_size=max_output_size,
                                                    nms_thres=0.4,
                                                    score_thres=match_thres)
        return nms_boxes, nms_scores

    def model_predict_fn_bak(self, mode, features):
        images = features
        match_thres = self.eval_params['match_thres']
        init_scale = self.eval_params['init_scale']
        pyramid_scale = self.eval_params['pyramid_scale']
        max_output_size = self.eval_params['max_output_size']
        batch_size = images.shape[0]

        im_arr = ut.tf_ops.image_pyramid(images, scale=pyramid_scale, min_len=16, divisible=16, init_scale=float(init_scale))
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
                fem2 = tf.image.resize_bilinear(fem2, size=ut.tf_ops.img_shape(fem1), align_corners=True)
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


class RefineDetector:
    def __init__(self, anchor_scales,
                 anchor_strides,
                 train_params=None,
                 eval_params=None):
        self.train_params = train_params
        self.eval_params = eval_params
        self.anchor_scales = anchor_scales
        self.anchor_strides = anchor_strides
        self.base_net = AlphaNet(num_anchors=len(self.anchor_scales), scope='base_net')
        self.refine_net = AlphaNet(num_anchors=len(self.anchor_scales), scope='refine_net')
        self.create_anchor_fn = lambda: AlphaAnchor(image_shape=self.train_params['train_image_shape'],
                                                    anchor_scales=self.anchor_scales,
                                                    anchor_strides=self.anchor_strides,
                                                    cls_match_thres=self.train_params['cls_match_thres'],
                                                    cls_unmatch_thres=self.train_params['cls_unmatch_thres'],
                                                    reg_match_thres=self.train_params['reg_match_thres'])

    def model_fn(self, features, labels, mode):
        if mode == ModeKeys.TRAIN:
            return self.model_train_fn(mode, features, labels)
        elif mode == ModeKeys.PREDICT:
            return self.model_predict_fn(mode, features)

    def model_train_fn(self, mode, features, labels):
        features = ut.tf_ops.reconvert_sample(features)
        images = features['image']
        boxes = features['boxes']
        cls_t = features['cls_t']
        cls_w = features['cls_w']
        reg_t = features['reg_t']
        reg_w = features['reg_w']
        ground_truth = (cls_t, cls_w, reg_t, reg_w)
        images_shape = ut.tf_ops.img_shape(images)

        images = tf.image.random_saturation(images, lower=1 - 10. / 255, upper=1 + 10. / 255)
        images = tf.image.random_hue(images, 10. / 255)
        images = tf.image.random_brightness(images, 10. / 255)

        train_refine = self.train_params['refine']
        train_base = not train_refine

        #logit, cls, reg
        base_prediction = self.base_net(images, is_training=True)
        if train_base:
            loss = self.loss_fn(ground_truth, base_prediction)
        else:
            ref_prediction = self.refine_net(images, is_training=True)
            loss = self.refine_loss_fn(ground_truth, base_prediction, ref_prediction)

        net_size = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        tf.logging.info('net_size: %d' % net_size)

        step = tf.train.get_or_create_global_step()
        lr = self._adjust_learning_rate(lr=0.001)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        minimize_op = optimizer.minimize(loss, step)
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_op)

        # summary
        #boxes_float = ut.tf_ops.convert_bboxes_to_float(boxes, images_shape)
        #images_gt_bboxes = tf.image.draw_bounding_boxes(images, boxes_float)
        #anchor_obj = self.create_anchor_fn()
        #pred_bboxes, pred_scores = self._decode(anchor_obj=anchor_obj,
        #                                        batch_codes=base_prediction[2],
        #                                        batch_scores=base_prediction[1])
        #pred_bboxes = ut.tf_ops.convert_bboxes_to_float(pred_bboxes, images_shape)
        #images_pred_bboxes = tf.image.draw_bounding_boxes(images, pred_bboxes)
        #tf.summary.image('images_gt_bboxes', images_gt_bboxes)
        #tf.summary.image('images_pred_bboxes', images_pred_bboxes)
        tf.summary.scalar('learning_rate', lr)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    def _adjust_learning_rate(self, lr):
        step = tf.train.get_or_create_global_step()
        decay_stage = int(2e5)
        first_stage = int(2e4)
        lr = tf.case([
            (step < first_stage, lambda: 0.0002),
            (step < decay_stage, lambda: lr),
            (step > 0,
             lambda: tf.train.exponential_decay(lr, global_step=step - decay_stage, decay_steps=10000, decay_rate=0.7))
        ])
        lr = tf.maximum(lr, 1e-7)
        return lr

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
        log_predicts, cls_predicts, reg_predicts, *_ = predictions
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

    def refine_loss_fn(self, ground_truth, base_prediction, ref_prediction):
        cls_targets, cls_weights, reg_targets, reg_weights = ground_truth
        base_log_predicts, base_cls_predicts, *_= base_prediction
        ref_log_predicts, *_= ref_prediction
        with tf.name_scope('loss_fn'):
            refine_losses = []
            ref_P_count, ref_F_count = 0, 0
            for i in range(len(self.anchor_strides)):
                cls_t, cls_w = cls_targets[i], cls_weights[i]
                base_cls_p, base_log_p = base_cls_predicts[i], base_cls_predicts[i]
                ref_log_p = ref_log_predicts[i]
                base_cls_p = tf.reshape(base_cls_p, [base_cls_p.shape[0], -1])
                ref_log_p = tf.reshape(ref_log_p, [ref_log_p.shape[0], -1, ref_log_p.shape[-1]])

                loss_w = tf.logical_and(base_cls_p > 0.01, base_cls_p < 0.95)
                loss_w = tf.logical_and(loss_w, cls_w > 0.5)
                refine_loss = tf.losses.sparse_softmax_cross_entropy(labels=cls_t, logits=ref_log_p,
                                                                     weights=loss_w,
                                                                     reduction=tf.losses.Reduction.SUM)
                ref_P_count += tf.reduce_sum(tf.boolean_mask(cls_t, loss_w))
                ref_F_count += tf.reduce_sum(tf.boolean_mask(1 - cls_t, loss_w))
                refine_losses.append(refine_loss)
            refine_losses = sum(refine_losses)
            tf.summary.scalar('refine_loss', refine_losses)
            tf.summary.scalar('ref_P_count', ref_P_count)
            tf.summary.scalar('ref_F_count', ref_F_count)
        return refine_losses

    def model_predict_fn(self, mode, features):
        images = features
        match_thres = self.eval_params['match_thres']
        init_scale = self.eval_params['init_scale']
        pyramid_scale = self.eval_params['pyramid_scale']
        max_output_size = self.eval_params['max_output_size']
        use_refine = self.eval_params['use_refine']

        batch_size = images.shape[0]

        im_arr = ut.tf_ops.image_pyramid(images, scale=pyramid_scale, min_len=16, divisible=8, init_scale=float(init_scale))

        def body(i, boxes_all, scores_all, im_arr=im_arr):
            im = im_arr.read(i)
            anchor_obj = AlphaAnchor(image_shape=ut.tf_ops.img_shape(im),
                                     anchor_scales=self.anchor_scales,
                                     anchor_strides=self.anchor_strides)
            base_prediction = self.base_net(im)
            base_cls_list = base_prediction[1]
            if use_refine:
                refine_prediction = self.refine_net(im)
                refine_cls_list = refine_prediction[1]
                batch_scores = []
                for base_cls, refine_cls in zip(base_cls_list, refine_cls_list):
                    conditions = tf.logical_and(base_cls > 0.01, base_cls < 0.95)
                    scores = tf.where(conditions, refine_cls, base_cls)
                    batch_scores.append(scores)
            else:
                batch_scores = base_cls_list

            boxes, scores = self._decode(anchor_obj=anchor_obj,
                                         batch_codes=base_prediction[2],
                                         batch_scores=batch_scores,
                                         max_output_size=max_output_size,
                                         match_thres=match_thres)
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
            tf.while_loop(cond=lambda i, _s, _b: i < im_arr.size(),
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
