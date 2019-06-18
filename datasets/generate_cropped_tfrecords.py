import numpy as np
import multiprocessing as mp
import PIL.Image as Image
import os, random, json, io
from datasets.preprocess import select_crop_face
from datasets.widerface import WiderTrain
import utils as ut

cfg_path = 'base.json'
cfg = json.load(open(cfg_path))

anchor_scales = [16, 32]
cropped_shape = 320
repeat = 100
outPath = 'wider_train_cropped%d_scaleTo%s_repeat%d.tfrecord' % \
          (cropped_shape,
           '.'.join([str(x) for x in anchor_scales]),
           repeat)
wider_train = WiderTrain(cfg['wider_train']['image_dir'],
                         anno_path=cfg['wider_train']['txt_path'],
                         min_size=cfg['wider_train']['min_face_size'])

def process_single_anno(fname, boxes):
    image = Image.open(fname)
    sel_box_idx = random.randrange(0, len(boxes))
    size = np.sqrt(np.product(boxes[sel_box_idx, 2:] - boxes[sel_box_idx, :2]))
    to_size = np.array(anchor_scales)
    to_size = to_size[to_size <= size * 1.5]
    to_size = [*to_size, size]
    to_size = random.choice(to_size)
    to_size = random.randint(round(to_size * 0.7), round(to_size * 1.3))
    cropped_image, cropped_boxes = select_crop_face(image, boxes, cropped_shape, sel_box_idx, to_size)
    boxes_mask = np.amax(cropped_boxes[:, 2:] - cropped_boxes[:, :2], axis=1) > 10
    cropped_boxes = cropped_boxes[boxes_mask]

    cropped_boxes = np.array(cropped_boxes).astype(np.float32)
    labels = np.ones(len(cropped_boxes))
    imgByteArr = io.BytesIO()
    cropped_image.save(imgByteArr, format='JPEG')
    cropped_image = imgByteArr.getvalue()

    return {'image':cropped_image,
            'boxes':cropped_boxes,
            'labels':labels}

def work_fn(anno):
    fname = anno[0]
    bboxes = anno[1][:, :4]
    samples = process_single_anno(os.path.join(cfg['wider_train']['image_dir'], fname), bboxes)
    return samples


def main():
    tfrecordObj = ut.tf_ops.ImageBoxTfrecord()
    data = list(wider_train.anno_dict.items())
    pool = mp.Pool(processes=16)
    for i in range(repeat):
        print('%d epochs' % i)
        samples = pool.map(work_fn, data, 300)
        samples = list(samples)
        tfrecordObj.write(outPath, samples)

if __name__ == '__main__':
    main()

