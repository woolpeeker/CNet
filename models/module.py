
import tensorflow as tf
from object_detection.core.box_list import BoxList
from object_detection.core import region_similarity_calculator as sim_calc
from object_detection.matchers.argmax_matcher import ArgMaxMatcher
from object_detection.box_coders.faster_rcnn_box_coder import FasterRcnnBoxCoder

from object_detection.core import box_coder as bcoder
from object_detection.core import box_list
from object_detection.core import matcher as mat
from object_detection.utils import shape_utils
from object_detection.core import standard_fields as fields

from utils import tf_ops

class AlphaAnchor:
    def __init__(self, image_shape, anchor_scales, anchor_strides,
                 cls_match_thres=None, cls_unmatch_thres=None, reg_match_thres=None):
        # match_thres used only for encoding
        self._similarity_calc = sim_calc.IouSimilarity()
        self._box_coder = FasterRcnnBoxCoder()
        if cls_match_thres is not None:
            cls_matcher = ArgMaxMatcher(matched_threshold=cls_match_thres,
                                        unmatched_threshold=cls_unmatch_thres)
            reg_matcher = ArgMaxMatcher(matched_threshold=reg_match_thres)
            self._target_assigner = FaceTargetAssigner(similarity_calc=self._similarity_calc,
                                                       cls_matcher=cls_matcher,
                                                       reg_matcher=reg_matcher,
                                                       box_coder=self._box_coder)

        self.anchor_scales = anchor_scales
        self.anchor_strides = anchor_strides
        self.anchors = []
        self.anchor_shapes = []

        if isinstance(image_shape, int):
            image_h, image_w = image_shape, image_shape
        elif isinstance(image_shape, tf.Tensor):
            image_h, image_w = image_shape[0], image_shape[1]
        else:
            raise ValueError('wrong image_shape value.')
        for scale, stride in zip(anchor_scales, anchor_strides):
            shape = tf.stack([tf.ceil(image_h / stride),
                               tf.ceil(image_w / stride)],
                               axis=0)
            shape = tf.to_int32(shape)
            anchor = self._get_anchors(shape, scale, stride)
            self.anchors.append(anchor)
            self.anchor_shapes.append(shape)


    def _get_anchors(self, shape, anchor_scale, anchor_stride):
        x, y = tf.meshgrid(tf.range(shape[1]),
                           tf.range(shape[0]))
        x, y = y * anchor_stride, x * anchor_stride
        x, y = tf.cast(y, tf.float32), tf.cast(x, tf.float32)
        ymin, ymax = y - anchor_scale / 2, y + anchor_scale / 2
        xmin, xmax = x - anchor_scale / 2, x + anchor_scale / 2
        anchors = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
        anchors = tf.cast(anchors, tf.float32)
        anchors = tf.reshape(anchors, [-1, 4])
        return BoxList(anchors)

    def encode(self, boxes):
        if isinstance(boxes, tf.Tensor):
            boxes = BoxList(boxes)
        cls_targets, cls_weights, reg_targets, reg_weights = [],[],[],[]
        for anchor in self.anchors:
            cls_target, cls_weight, reg_target, reg_weight, _, _ = \
                self._target_assigner.assign(anchors=anchor, groundtruth_boxes=boxes)
            cls_target = tf.squeeze(cls_target, -1)
            cls_target = tf.to_int32(cls_target)
            cls_weight = tf.squeeze(cls_weight, -1)
            cls_targets.append(cls_target)
            cls_weights.append(cls_weight)
            reg_targets.append(reg_target)
            reg_weights.append(reg_weight)
        return cls_targets, cls_weights, reg_targets, reg_weights

    def decode(self, codes):
        """
        :param codes: list of codes [a0_code, a1_code, ...]
        :return: deocded BoxList
        """
        decode_boxes = []
        for i in range(len(codes)):
            code = codes[i]
            anchor = self.anchors[i]
            box = self._box_coder.decode(code, anchor).get()
            decode_boxes.append(box)
        decode_boxes = tf.concat(decode_boxes, axis=0)
        decode_boxes = BoxList(decode_boxes)
        return decode_boxes

    def batch_decode(self, batch_codes):
        """
        :param batch_codes: list of batched codes [batched_a0_code, batched_a1_code, ...],
                            each tensor shape must be [batch_size, H*W, 4]
        :return: [batchsize, H*W, 4]
        """
        batch_boxes = []
        assert(len(batch_codes) == len(self.anchors))
        for i in range(len(batch_codes)):
            batch_code = batch_codes[i]
            anchor = self.anchors[i]
            batch_boxes_anchor_i = bcoder.batch_decode(batch_code, self._box_coder, anchor)
            batch_boxes.append(batch_boxes_anchor_i)
        batch_boxes = tf.concat(batch_boxes, axis=1)
        return batch_boxes

class SingleAnchor:
    def __init__(self, image_shape, anchor_scale, anchor_stride,
                 cls_match_thres=None, cls_unmatch_thres=None, reg_match_thres=None):
        # match_thres used only for encoding
        self._similarity_calc = sim_calc.IouSimilarity()
        self._box_coder = FasterRcnnBoxCoder()
        if cls_match_thres is not None:
            cls_matcher = ArgMaxMatcher(matched_threshold=cls_match_thres,
                                        unmatched_threshold=cls_unmatch_thres)
            reg_matcher = ArgMaxMatcher(matched_threshold=reg_match_thres)
            self._target_assigner = FaceTargetAssigner(similarity_calc=self._similarity_calc,
                                                       cls_matcher=cls_matcher,
                                                       reg_matcher=reg_matcher,
                                                       box_coder=self._box_coder)

        self.anchor_scale = anchor_scale
        self.anchor_stride = anchor_stride

        if isinstance(image_shape, int):
            image_h, image_w = image_shape, image_shape
        elif isinstance(image_shape, tf.Tensor):
            image_h, image_w = image_shape[0], image_shape[1]
        else:
            raise ValueError('wrong image_shape value.')
        shape = tf.stack([tf.ceil(image_h / self.anchor_stride),
                           tf.ceil(image_w / self.anchor_stride)],
                           axis=0)
        self.anchor_shape = tf.to_int32(shape)
        self.anchor = self._get_anchors(shape, self.anchor_scale, self.anchor_stride)


    def _get_anchors(self, shape, anchor_scale, anchor_stride):
        x, y = tf.meshgrid(tf.range(shape[1]),
                           tf.range(shape[0]))
        x, y = y * anchor_stride, x * anchor_stride
        x, y = tf.cast(y, tf.float32), tf.cast(x, tf.float32)
        ymin, ymax = y - anchor_scale / 2, y + anchor_scale / 2
        xmin, xmax = x - anchor_scale / 2, x + anchor_scale / 2
        anchors = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
        anchors = tf.cast(anchors, tf.float32)
        anchors = tf.reshape(anchors, [-1, 4])
        return BoxList(anchors)

    def encode(self, boxes):
        if isinstance(boxes, tf.Tensor):
            boxes = BoxList(boxes)
        cls_targets, cls_weights, reg_targets, reg_weights = [],[],[],[]
        cls_target, cls_weight, reg_target, reg_weight, _, _ = \
            self._target_assigner.assign(anchors=self.anchor, groundtruth_boxes=boxes)
        cls_target = tf.squeeze(cls_target, -1)
        cls_target = tf.to_int32(cls_target)
        cls_weight = tf.squeeze(cls_weight, -1)
        cls_targets.append(cls_target)
        cls_weights.append(cls_weight)
        reg_targets.append(reg_target)
        reg_weights.append(reg_weight)
        return cls_targets, cls_weights, reg_targets, reg_weights

    def decode(self, code):
        """
        :param codes: list of codes [a0_code, a1_code, ...]
        :return: deocded BoxList
        """
        boxes = self._box_coder.decode(code, self.anchor).get()
        return boxes

    def batch_decode(self, batch_code, batch_score, max_out, thres):
        """
        :param batch_codes: list of batched codes [batched_a0_code, batched_a1_code, ...],
                            each tensor shape must be [batch_size, H*W, 4]
        :param batch_score: list of batched scores
        :param max_out: max output
        :param thres: threshold
        :return: [batchsize, *, 4]
        """
        with tf.name_scope('batch_decode'):
            anchor = self.anchor
            code_rank_assert = tf.assert_equal(tf.rank(batch_code), 4)
            score_rank_assert = tf.assert_equal(tf.rank(batch_score), 3)
            with tf.control_dependencies([code_rank_assert, score_rank_assert]):
                c_shape = batch_code.shape
                s_shape = batch_score.shape
                batch_code = tf.reshape(batch_code, [c_shape[0], -1, c_shape[3]])
                batch_score = tf.reshape(batch_score, [s_shape[0], -1])
                batch_boxes = bcoder.batch_decode(batch_code, self._box_coder, anchor)
                batch_boxes, batch_scores = tf_ops.nms_batch(batch_boxes, batch_score,
                                                            max_output_size=max_out, nms_thres=0.4,
                                                            score_thres=thres, pad=True)
        return batch_boxes, batch_scores


'''
class BetaSimilarity(sim_calc.RegionSimilarityCalculator):
    def __init__(self):
        super(BetaSimilarity, self).__init__()
        self.distance_thres = 0.5
        self.scale_thres = 0.25
    def _compare(self, boxlist1, boxlist2):
        # in targe assign, boxlist1 is gtboxes, boxlist2 is anchorboxes
        ymin1, xmin1, ymax1, xmax1 = tf.split(boxlist1.get(), 4, axis=1)
        yc1, xc1 = (ymin1+ymax1)/2, (xmin1+xmax1)/2
        h1, w1 = ymax1-ymin1, xmax1-xmin1
        ymin2, xmin2, ymax2, xmax2 = tf.split(boxlist2.get(), 4, axis=1)
        h2, w2 = ymax2-ymin2, xmax2-xmin2
        yc2, xc2 = (ymin2+ymax2)/2, (xmin2+xmax2)/2
        all_pair_dist_y = abs(yc1 - tf.transpose(yc2)) / tf.transpose(h2)
        all_pair_dist_x = abs(xc1 - tf.transpose(xc2)) / tf.transpose(w2)
        all_pair_scale_h = abs(h1 - tf.transpose(h2)) / tf.transpose(h2)
        all_pair_scale_w = abs(w1 - tf.transpose(w2)) / tf.transpose(w2)
        
        dist_mask = tf.logical_and(all_pair_dist_y < self.distance_thres,
                                   all_pair_dist_x < self.distance_thres)
        scale_mask = tf.logical_or(all_pair_scale_h < self.scale_thres,
                                    all_pair_scale_w < self.scale_thres)
        mask = tf.logical_and(dist_mask, scale_mask)
        mask = tf.to_float(mask)
        return mask
'''


class FaceTargetAssigner(object):
    def __init__(self,
                 similarity_calc,
                 cls_matcher,
                 reg_matcher,
                 box_coder,
                 negative_class_weight=1.0):
        """Construct Object Detection Target Assigner.

        Args:
          similarity_calc: a RegionSimilarityCalculator
          matcher: an object_detection.core.Matcher used to match groundtruth to
            anchors.
          box_coder: an object_detection.core.BoxCoder used to encode matching
            groundtruth boxes with respect to anchors.
          negative_class_weight: classification weight to be associated to negative
            anchors (default: 1.0). The weight must be in [0., 1.].

        Raises:
          ValueError: if similarity_calc is not a RegionSimilarityCalculator or
            if matcher is not a Matcher or if box_coder is not a BoxCoder
        """
        if not isinstance(similarity_calc, sim_calc.RegionSimilarityCalculator):
            raise ValueError('similarity_calc must be a RegionSimilarityCalculator')
        if not isinstance(cls_matcher, mat.Matcher):
            raise ValueError('cls_matcher must be a Matcher')
        if not isinstance(reg_matcher, mat.Matcher):
            raise ValueError('reg_matcher must be a Matcher')
        if not isinstance(box_coder, bcoder.BoxCoder):
            raise ValueError('box_coder must be a BoxCoder')
        self._similarity_calc = similarity_calc
        self._cls_matcher = cls_matcher
        self._reg_matcher = reg_matcher
        self._box_coder = box_coder
        self._negative_class_weight = negative_class_weight

    @property
    def box_coder(self):
        return self._box_coder

    # TODO(rathodv): move labels, scores, and weights to groundtruth_boxes fields.
    def assign(self,
               anchors,
               groundtruth_boxes,
               groundtruth_weights=None):
        """Assign classification and regression targets to each anchor.

        For a given set of anchors and groundtruth detections, match anchors
        to groundtruth_boxes and assign classification and regression targets to
        each anchor as well as weights based on the resulting match (specifying,
        e.g., which anchors should not contribute to training loss).

        Anchors that are not matched to anything are given a classification target
        of self._unmatched_cls_target which can be specified via the constructor.

        Args:
          anchors: a BoxList representing N anchors
          groundtruth_boxes: a BoxList representing M groundtruth boxes
          groundtruth_weights: a float tensor of shape [M] indicating the weight to
            assign to all anchors match to a particular groundtruth box. The weights
            must be in [0., 1.]. If None, all weights are set to 1. Generally no
            groundtruth boxes with zero weight match to any anchors as matchers are
            aware of groundtruth weights. Additionally, `cls_weights` and
            `reg_weights` are calculated using groundtruth weights as an added
            safety.

        Returns:
          cls_targets: a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k],
            where the subshape [d_1, ..., d_k] is compatible with groundtruth_labels
            which has shape [num_gt_boxes, d_1, d_2, ... d_k].
          cls_weights: a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k],
            representing weights for each element in cls_targets.
          reg_targets: a float32 tensor with shape [num_anchors, box_code_dimension]
          reg_weights: a float32 tensor with shape [num_anchors]
          match: a matcher.Match object encoding the match between anchors and
            groundtruth boxes, with rows corresponding to groundtruth boxes
            and columns corresponding to anchors.

        Raises:
          ValueError: if anchors or groundtruth_boxes are not of type
            box_list.BoxList
        """
        if not isinstance(anchors, box_list.BoxList):
            raise ValueError('anchors must be an BoxList')
        if not isinstance(groundtruth_boxes, box_list.BoxList):
            raise ValueError('groundtruth_boxes must be an BoxList')

        unmatched_class_label = tf.constant([0], tf.float32)
        groundtruth_labels = tf.ones(tf.expand_dims(groundtruth_boxes.num_boxes(),
                                                    0))
        groundtruth_labels = tf.expand_dims(groundtruth_labels, -1)

        unmatched_shape_assert = shape_utils.assert_shape_equal(
            shape_utils.combined_static_and_dynamic_shape(groundtruth_labels)[1:],
            shape_utils.combined_static_and_dynamic_shape(unmatched_class_label))
        labels_and_box_shapes_assert = shape_utils.assert_shape_equal(
            shape_utils.combined_static_and_dynamic_shape(
                groundtruth_labels)[:1],
            shape_utils.combined_static_and_dynamic_shape(
                groundtruth_boxes.get())[:1])

        if groundtruth_weights is None:
            num_gt_boxes = groundtruth_boxes.num_boxes_static()
            if not num_gt_boxes:
                num_gt_boxes = groundtruth_boxes.num_boxes()
            groundtruth_weights = tf.ones([num_gt_boxes], dtype=tf.float32)

        with tf.control_dependencies(
                [unmatched_shape_assert, labels_and_box_shapes_assert]):
            match_quality_matrix = self._similarity_calc.compare(groundtruth_boxes,
                                                                 anchors)
            cls_match = self._cls_matcher.match(match_quality_matrix)

            reg_match = self._reg_matcher.match(match_quality_matrix)
            reg_targets = self._create_regression_targets(anchors,
                                                          groundtruth_boxes,
                                                          reg_match)
            cls_targets = self._create_classification_targets(groundtruth_labels,
                                                              unmatched_class_label,
                                                              cls_match)
            reg_weights = self._create_regression_weights(reg_match, groundtruth_weights)

            cls_weights = self._create_classification_weights(cls_match,
                                                              groundtruth_weights)
            # convert cls_weights from per-anchor to per-class.
            class_label_shape = tf.shape(cls_targets)[1:]
            weights_shape = tf.shape(cls_weights)
            weights_multiple = tf.concat(
                [tf.ones_like(weights_shape), class_label_shape],
                axis=0)
            for _ in range(len(cls_targets.get_shape()[1:])):
                cls_weights = tf.expand_dims(cls_weights, -1)
            cls_weights = tf.tile(cls_weights, weights_multiple)

        num_anchors = anchors.num_boxes_static()
        if num_anchors is not None:
            reg_targets = self._reset_target_shape(reg_targets, num_anchors)
            cls_targets = self._reset_target_shape(cls_targets, num_anchors)
            reg_weights = self._reset_target_shape(reg_weights, num_anchors)
            cls_weights = self._reset_target_shape(cls_weights, num_anchors)

        return cls_targets, cls_weights, reg_targets, reg_weights, cls_match, reg_match

    def _reset_target_shape(self, target, num_anchors):
        """Sets the static shape of the target.

        Args:
          target: the target tensor. Its first dimension will be overwritten.
          num_anchors: the number of anchors, which is used to override the target's
            first dimension.

        Returns:
          A tensor with the shape info filled in.
        """
        target_shape = target.get_shape().as_list()
        target_shape[0] = num_anchors
        target.set_shape(target_shape)
        return target

    def _create_regression_targets(self, anchors, groundtruth_boxes, match):
        """Returns a regression target for each anchor.

        Args:
          anchors: a BoxList representing N anchors
          groundtruth_boxes: a BoxList representing M groundtruth_boxes
          match: a matcher.Match object

        Returns:
          reg_targets: a float32 tensor with shape [N, box_code_dimension]
        """
        matched_gt_boxes = match.gather_based_on_match(
            groundtruth_boxes.get(),
            unmatched_value=tf.zeros(4),
            ignored_value=tf.zeros(4))
        matched_gt_boxlist = box_list.BoxList(matched_gt_boxes)
        if groundtruth_boxes.has_field(fields.BoxListFields.keypoints):
            groundtruth_keypoints = groundtruth_boxes.get_field(
                fields.BoxListFields.keypoints)
            matched_keypoints = match.gather_based_on_match(
                groundtruth_keypoints,
                unmatched_value=tf.zeros(groundtruth_keypoints.get_shape()[1:]),
                ignored_value=tf.zeros(groundtruth_keypoints.get_shape()[1:]))
            matched_gt_boxlist.add_field(fields.BoxListFields.keypoints,
                                         matched_keypoints)
        matched_reg_targets = self._box_coder.encode(matched_gt_boxlist, anchors)
        match_results_shape = shape_utils.combined_static_and_dynamic_shape(
            match.match_results)

        # Zero out the unmatched and ignored regression targets.
        unmatched_ignored_reg_targets = tf.tile(
            self._default_regression_target(), [match_results_shape[0], 1])
        matched_anchors_mask = match.matched_column_indicator()
        reg_targets = tf.where(matched_anchors_mask,
                               matched_reg_targets,
                               unmatched_ignored_reg_targets)
        return reg_targets

    def _default_regression_target(self):
        """Returns the default target for anchors to regress to.

        Default regression targets are set to zero (though in
        this implementation what these targets are set to should
        not matter as the regression weight of any box set to
        regress to the default target is zero).

        Returns:
          default_target: a float32 tensor with shape [1, box_code_dimension]
        """
        return tf.constant([self._box_coder.code_size * [0]], tf.float32)

    def _create_classification_targets(self, groundtruth_labels,
                                       unmatched_class_label, match):
        """Create classification targets for each anchor.

        Assign a classification target of for each anchor to the matching
        groundtruth label that is provided by match.  Anchors that are not matched
        to anything are given the target self._unmatched_cls_target

        Args:
          groundtruth_labels:  a tensor of shape [num_gt_boxes, d_1, ... d_k]
            with labels for each of the ground_truth boxes. The subshape
            [d_1, ... d_k] can be empty (corresponding to scalar labels).
          unmatched_class_label: a float32 tensor with shape [d_1, d_2, ..., d_k]
            which is consistent with the classification target for each
            anchor (and can be empty for scalar targets).  This shape must thus be
            compatible with the groundtruth labels that are passed to the "assign"
            function (which have shape [num_gt_boxes, d_1, d_2, ..., d_k]).
          match: a matcher.Match object that provides a matching between anchors
            and groundtruth boxes.

        Returns:
          a float32 tensor with shape [num_anchors, d_1, d_2 ... d_k], where the
          subshape [d_1, ..., d_k] is compatible with groundtruth_labels which has
          shape [num_gt_boxes, d_1, d_2, ... d_k].
        """
        return match.gather_based_on_match(
            groundtruth_labels,
            unmatched_value=unmatched_class_label,
            ignored_value=unmatched_class_label)

    def _create_regression_weights(self, match, groundtruth_weights):
        """Set regression weight for each anchor.

        Only positive anchors are set to contribute to the regression loss, so this
        method returns a weight of 1 for every positive anchor and 0 for every
        negative anchor.

        Args:
          match: a matcher.Match object that provides a matching between anchors
            and groundtruth boxes.
          groundtruth_weights: a float tensor of shape [M] indicating the weight to
            assign to all anchors match to a particular groundtruth box.

        Returns:
          a float32 tensor with shape [num_anchors] representing regression weights.
        """
        return match.gather_based_on_match(
            groundtruth_weights, ignored_value=0., unmatched_value=0.)

    def _create_classification_weights(self,
                                       match,
                                       groundtruth_weights):
        """Create classification weights for each anchor.

        Positive (matched) anchors are associated with a weight of
        positive_class_weight and negative (unmatched) anchors are associated with
        a weight of negative_class_weight. When anchors are ignored, weights are set
        to zero. By default, both positive/negative weights are set to 1.0,
        but they can be adjusted to handle class imbalance (which is almost always
        the case in object detection).

        Args:
          match: a matcher.Match object that provides a matching between anchors
            and groundtruth boxes.
          groundtruth_weights: a float tensor of shape [M] indicating the weight to
            assign to all anchors match to a particular groundtruth box.

        Returns:
          a float32 tensor with shape [num_anchors] representing classification
          weights.
        """
        return match.gather_based_on_match(
            groundtruth_weights,
            ignored_value=0.,
            unmatched_value=self._negative_class_weight)

    def get_box_coder(self):
        """Get BoxCoder of this TargetAssigner.

        Returns:
          BoxCoder object.
        """
        return self._box_coder

