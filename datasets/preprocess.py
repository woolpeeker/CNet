
import utils as ut

import PIL
from PIL import Image
import numpy as np
import random

__all__ = ['select_crop_face']

def select_crop_face(img, boxes, labels, out_shape, sel_box_idx, to_size):
    '''random crop the image, the output image must contain the selected box,
    and resize the image to make the box to_size. crop fisrtly, resize second.
    :param im: PIL.Image
    :param boxes: int [[ymin, xmin, ymax, xmax],...]
    :param out_shape: a scalar
    :param sel_box: [ymin, xmin, ymax, xmax]
    :param to_size: a scalar
    :return: (cropped_image, cropped_boxes)
    '''
    if len(boxes)==0:
        raise ValueError('boxes is empty.')
    sel_box = boxes[sel_box_idx]
    original_sel_box_size = np.sqrt((sel_box[2]-sel_box[0]) * (sel_box[3]-sel_box[1]))
    ratio = to_size / original_sel_box_size
    ratio_h = ratio * random.uniform(0.85, 1.15)
    ratio_w = ratio * random.uniform(0.85, 1.15)
    resized_h = int(ratio_h * img.size[1])
    resized_w = int(ratio_w * img.size[0])
    resized_img = img.resize((resized_w, resized_h), Image.BILINEAR)
    resized_boxes = ut.np_ops.resize_bboxes((ratio_h,ratio_w), boxes)
    sel_box = resized_boxes[sel_box_idx]

    dst_h = out_shape
    dst_w = out_shape
    dst_y_interval = (np.maximum(0, sel_box[2] - dst_h),
                      np.minimum(sel_box[0], resized_h - dst_h))
    dst_x_interval = (np.maximum(0, sel_box[3] - dst_w),
                      np.minimum(sel_box[1], resized_w - dst_w))
    dst_y = np.random.uniform(low=min(dst_y_interval),
                              high=max(dst_y_interval))
    dst_x = np.random.uniform(low=min(dst_x_interval),
                              high=max(dst_x_interval))
    cropped_img, cropped_boxes = ut.np_ops.crop_image(resized_img, [dst_y, dst_x, dst_y+dst_h, dst_x+dst_w], resized_boxes)
    mask = ut.np_ops.bboxes_filter_center(cropped_boxes, (dst_h, dst_w))
    cropped_boxes = cropped_boxes[mask]
    cropped_labels = labels[mask]
    if len(cropped_boxes)==0:
        cropped_boxes = np.zeros([1,4])
        cropped_labels = np.zeros([1])
    return cropped_img, cropped_boxes, cropped_labels
