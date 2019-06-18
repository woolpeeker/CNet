
import tensorflow as tf
import tensorflow.contrib.slim as slim

import utils as ut

class AlphaNet:
    def __init__(self, num_anchors, scope='AlphaNet'):
        self.num_anchors = num_anchors
        self.scope = scope
    def __call__(self, inputs, is_training=False):
        logit_list = []
        cls_list = []
        reg_list = []

        arg_scopes={
            (slim.separable_conv2d,): {'kernel_size': 3,
                                       'depth_multiplier': 3,
                                       'normalizer_fn': slim.batch_norm,
                                       'activation_fn': tf.nn.leaky_relu},
            (slim.batch_norm,): {'is_training': is_training,
                                 'center': True,
                                 'scale': True}
        }

        with tf.name_scope(self.scope):
            with tf.variable_scope(self.scope):
                with ut.tf_ops.set_arg_scope(arg_scopes):
                    #FEM
                    net = slim.separable_conv2d(inputs, num_outputs=64, stride=2, depth_multiplier=32, scope='fem_conv0')
                    net = slim.separable_conv2d(net, num_outputs=128, stride=1, scope='fem_conv1')
                    net = slim.separable_conv2d(net, num_outputs=128, stride=2, scope='fem_conv2')
                    net = slim.separable_conv2d(net, num_outputs=128, stride=1, scope='fem_conv3')
                    fem_out = slim.separable_conv2d(net, num_outputs=128, stride=2, scope='fem_conv4')
                    #DEM
                    for i in range(self.num_anchors):
                        feature = slim.separable_conv2d(fem_out, num_outputs=64, scope='dem%d_log_conv0'%i)
                        logit = slim.separable_conv2d(feature, num_outputs=2, activation_fn=None, scope='dem%d_log_conv1'%i)
                        cls = slim.softmax(logit)[..., 1]

                        feature = slim.separable_conv2d(fem_out, num_outputs=64, scope='dem%d_reg_conv0'%i)
                        reg = slim.separable_conv2d(feature, num_outputs=4, activation_fn=None, scope='dem%d_reg_conv1'%i)
                        logit_list.append(logit)
                        cls_list.append(cls)
                        reg_list.append(reg)
        return logit_list, cls_list, reg_list

class HalfAlphaNet:
    def __init__(self, num_anchors, scope='HalfAlphaNet'):
        self.num_anchors = num_anchors
        self.scope = scope
    def __call__(self, inputs, is_training=False):
        logit_list = []
        cls_list = []
        reg_list = []

        arg_scopes={
            (slim.separable_conv2d,): {'kernel_size': 3,
                                       'depth_multiplier': 3,
                                       'normalizer_fn': slim.batch_norm,
                                       'activation_fn': tf.nn.leaky_relu},
            (slim.batch_norm,): {'is_training': is_training,
                                 'center': True,
                                 'scale': True}
        }

        with tf.name_scope(self.scope):
            with tf.variable_scope(self.scope):
                with ut.tf_ops.set_arg_scope(arg_scopes):
                    #FEM
                    net = slim.separable_conv2d(inputs, num_outputs=32, stride=2, depth_multiplier=32, scope='fem_conv0')
                    net = slim.separable_conv2d(net, num_outputs=64, stride=1, scope='fem_conv1')
                    net = slim.separable_conv2d(net, num_outputs=64, stride=2, scope='fem_conv2')
                    net = slim.separable_conv2d(net, num_outputs=64, stride=1, scope='fem_conv3')
                    fem_out = slim.separable_conv2d(net, num_outputs=64, stride=2, scope='fem_conv4')
                    #DEM
                    for i in range(self.num_anchors):
                        feature = slim.separable_conv2d(fem_out, num_outputs=32, scope='dem%d_log_conv0'%i)
                        logit = slim.separable_conv2d(feature, num_outputs=2, activation_fn=None, scope='dem%d_log_conv1'%i)
                        cls = slim.softmax(logit)[..., 1]

                        feature = slim.separable_conv2d(fem_out, num_outputs=32, scope='dem%d_reg_conv0'%i)
                        reg = slim.separable_conv2d(feature, num_outputs=4, activation_fn=None, scope='dem%d_reg_conv1'%i)
                        logit_list.append(logit)
                        cls_list.append(cls)
                        reg_list.append(reg)
        return logit_list, cls_list, reg_list


class PyramidFusedNet:
    def __init__(self, num_anchors, scope='PyramidFusedNet'):
        self.num_anchors = num_anchors
        self.scope = scope

    def get_arg_scopes(self, is_training):
        return {
            (slim.separable_conv2d,): {'kernel_size': 3,
                                       'depth_multiplier': 3,
                                       'normalizer_fn': slim.batch_norm,
                                       'activation_fn': tf.nn.leaky_relu},
            (slim.batch_norm,): {'is_training': is_training,
                                 'center': True,
                                 'scale': True}
        }

    def train_fn(self, inputs):
        fem1, _ = self.fem(inputs, is_training=True)
        size = tf.to_int32(tf.shape(inputs)[1:3] / 2)
        _, fem2 = self.fem(tf.image.resize_bilinear(inputs, size, True), is_training=True)
        feature = tf.concat([fem1, fem2], axis=-1)
        logit0, cls0, reg0 = self.dem(feature, anchor_num=0, is_training=True)
        logit1, cls1, reg1 = self.dem(feature, anchor_num=1, is_training=True)
        logit_list = [logit0, logit1]
        cls_list = [cls0, cls1]
        reg_list = [reg0, reg1]
        return logit_list, cls_list, reg_list

    def fem(self, inputs, is_training=False):
        arg_scope = self.get_arg_scopes(is_training)
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            with ut.tf_ops.set_arg_scope(arg_scope):
                # FEM
                net = slim.separable_conv2d(inputs, num_outputs=32, stride=2, depth_multiplier=32, scope='fem_conv0')
                net = slim.separable_conv2d(net, num_outputs=64, stride=1, scope='fem_conv1')
                net = slim.separable_conv2d(net, num_outputs=64, stride=2, scope='fem_conv2')
                b = slim.separable_conv2d(net, num_outputs=64, stride=1, scope='fem_conv3')
                a = slim.separable_conv2d(b, num_outputs=64, stride=2, scope='fem_conv4')
        return [a, b]

    def dem(self, feature, anchor_num, is_training=False):
        arg_scope = self.get_arg_scopes(is_training)
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            with ut.tf_ops.set_arg_scope(arg_scope):
                feature = slim.separable_conv2d(feature, num_outputs=32, scope='dem%d_log_conv0' % anchor_num)
                logit = slim.separable_conv2d(feature, num_outputs=2, activation_fn=None,
                                              scope='dem%d_log_conv1' % anchor_num)
                cls = slim.softmax(logit)[..., 1]

                feature = slim.separable_conv2d(feature, num_outputs=32, scope='dem%d_reg_conv0' % anchor_num)
                reg = slim.separable_conv2d(feature, num_outputs=4, activation_fn=None, scope='dem%d_reg_conv1' % anchor_num)
            return [logit, cls, reg]
