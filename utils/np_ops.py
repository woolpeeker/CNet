import numpy as np
from PIL import Image, ImageOps, ImageDraw

__all__ = ['convert_bboxes_to_float',
           'convert_bboxes_to_int',
           'bboxes_filter_center',
           'crop_bboxes',
           'resize_bboxes',
           'pad_image',
           'crop_image',
           'draw_boxes']

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
