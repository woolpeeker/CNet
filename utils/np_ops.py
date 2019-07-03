import numpy as np
from PIL import Image, ImageOps, ImageDraw

__all__ = ['convert_bboxes_to_float',
           'convert_bboxes_to_int',
           'bboxes_filter_center',
           'crop_bboxes',
           'resize_bboxes',
           'pad_image',
           'crop_image',
           'draw_boxes',
           'intersection',
           'iou']

def convert_bboxes_to_float(bboxes, image_shape):
    '''
    :param bboxes: int type boxes [ymin, xmin, ymax, ymin]
    :param image_shape: [height, width]
    :return: float bboxes
    '''
    bboxes = [bboxes[..., 0] / image_shape[0],
              bboxes[..., 1] / image_shape[1],
              bboxes[..., 2] / image_shape[0],
              bboxes[..., 3] / image_shape[1]]
    bboxes = np.stack(bboxes,axis=-1)
    return bboxes

def convert_bboxes_to_int(bboxes, image_shape):
    bboxes = [bboxes[..., 0] * image_shape[0],
              bboxes[..., 1] * image_shape[1],
              bboxes[..., 2] * image_shape[0],
              bboxes[..., 3] * image_shape[1]]
    bboxes = np.stack(bboxes, axis=-1)
    return bboxes

def bboxes_filter_center(bboxes, image_shape):
    """Filter out bounding boxes whose center are not in
    the rectangle [0, 0, 1, 1] + margins. The margin Tensor
    can be used to enforce or loosen this condition.
    :param bboxes: int format boxes
    :param image_shape: [h,w]
    Return:
      mask: a logical numpy array
    """
    cy = (bboxes[..., 0] + bboxes[..., 2]) / 2.
    cx = (bboxes[..., 1] + bboxes[..., 3]) / 2.
    mask = cy > 0
    mask = np.logical_and(mask, cx > 0)
    mask = np.logical_and(mask, cy < image_shape[0])
    mask = np.logical_and(mask, cx < image_shape[1])
    return mask

def crop_bboxes(bbox_ref, bboxes):
    """Transform bounding boxes based on a reference bounding box,
    Useful for updating a collection of boxes after cropping an image.
    :param bbox_ref, bboxes: int format boxes [ymin, xmin, ymax, xmax]
    """
    v = np.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
    bboxes = bboxes - v
    return bboxes

def resize_bboxes(ratios, bboxes):
    """calibrate the bboxes after the image was resized.
    :param ratios: (ratio_h, ratio_w)
    :param bboxes: int format bboxes
    :return: int format bboxes
    """
    ymin = bboxes[..., 0] * ratios[0]
    xmin = bboxes[..., 1] * ratios[1]
    ymax = bboxes[..., 2] * ratios[0]
    xmax = bboxes[..., 3] * ratios[1]
    bboxes = np.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes

def pad_image(img, boxes, pad_shape):
    '''
    pad the image to pad_shape.
    if the a side of img is bigger than pad_shape, then do nothing on the side.
    :param img: Pillow Image
    :param boxes: int boxes
    :param pad_shape: (height, width)
    :return: (padded_img, padded_boxes)
    '''
    img_w, img_h = img.shape
    if img_h<pad_shape[0] or img_w<pad_shape[1]:
        delta_h = max(0, pad_shape[0]-img_h)
        delta_w = max(0, pad_shape[1]-img_w)
        padding = (delta_h // 2, delta_w // 2, delta_h - (delta_h // 2), delta_w - (delta_w // 2))
        padded_img = ImageOps.expand(img, padding)
        boxes[0] += padding[0]
        boxes[1] += padding[1]
        return padded_img, boxes
    else:
        return img, boxes

def crop_image(img, crop_box, boxes):
    '''crop the image
    :param img: Pillow Image
    :param crop_box: int [ymin, xmin, ymax, xmax]
    :param boxes: int
    :return: (cropped_img, cropeed_boxes)
    '''
    cropped_img = img.crop([crop_box[1],
                            crop_box[0],
                            crop_box[3],
                            crop_box[2]])
    cropped_boxes = crop_bboxes(crop_box, boxes)
    return cropped_img, cropped_boxes

def draw_boxes(img, boxes, color='green', width=3):
    '''
    draw the boxes in the img
    :param img: Pillow Image or numpy
    :param boxes: boxes, [[ymax, xmax, ymin, xmin]...]
    :param color: color
    :return: Image drawed boxes
    '''
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')
    elif not isinstance(img, Image.Image):
        raise ValueError("image must be a Image or ndarray.")
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle([box[1], box[0], box[3], box[2]], outline=color, width=width)
    return img


def intersection(boxes1, boxes2):
    """
    :param boxes1: numpy.ndarray [num, 4], each column is ymin, xmin, ymax, xmax
    :param boxes2: same as boxes1
    :return: numpy.ndarray [num1, num2]
    """
    assert(boxes1.shape[1]==4 and boxes2.shape[1]==4)
    ymin1, xmin1, ymax1, xmax1 = np.split(boxes1, 4, axis=1)
    ymin2, xmin2, ymax2, xmax2 = np.split(boxes2, 4, axis=1)
    all_pairs_min_ymax = np.minimum(ymax1, ymax2.reshape(-1))
    all_pairs_max_ymin = np.maximum(ymin1, ymin2.reshape(-1))
    intersect_heights = np.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = np.minimum(xmax1, xmax2.reshape(-1))
    all_pairs_max_xmin = np.maximum(xmin1, xmin2.reshape(-1))
    intersect_widths = np.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def iou(boxes1, boxes2):
    """
    :param boxes1: numpy.ndarray [num, 4], each column is ymin, xmin, ymax, xmax
    :param boxes2: same as boxes1
    :return: numpy.ndarray [num1, num2]
    """
    intersections = intersection(boxes1, boxes2)
    areas1 = (boxes1[:,2]-boxes1[:,0]) * (boxes1[:,3]-boxes1[:,1])
    areas2 = (boxes2[:,2]-boxes2[:,0]) * (boxes2[:,3]-boxes2[:,1])
    unions = areas1.reshape([-1, 1]) + areas2.reshape([1, -1]) - intersections
    ious = intersections / unions
    return ious
