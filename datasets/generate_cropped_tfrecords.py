import numpy as np
import multiprocessing as mp
import PIL.Image as Image
import os, random, json, io
from datasets.preprocess import select_crop_face
from datasets.widerface import read_anno
import utils as ut

cfg_path = 'base.json'
cfg = json.load(open(cfg_path))

anchor_scales = [16]
repeat = 100
outPath = 'dsfd_wider_train_cropped%d_scaleTo%s_repeat%d.tfrecord' % \
          (cfg['train']['image_shape'],
           '.'.join([str(x) for x in anchor_scales]),
           repeat)
annotations = read_anno(cfg['dsfd']['txt_path'],
                        cfg['wider_train']['min_face_size'],
                        has_label=True)
annotations = list(annotations.items())

def process_single_anno(sample):
    image = sample['image']
    boxes = sample['boxes']
    labels = sample['labels']
    sel_box_idx = random.randrange(0, len(boxes))
    size = np.sqrt(np.product(boxes[sel_box_idx, 2:4] - boxes[sel_box_idx, 0:2]))
    to_size = np.array(anchor_scales)
    to_size = to_size[to_size <= size * 1.5]
    to_size = [*to_size, size]
    to_size = random.choice(to_size)
    to_size = random.randint(round(to_size * 0.7), round(to_size * 1.3))
    cropped_image, cropped_boxes, cropped_labels = select_crop_face(image, boxes, labels, cfg['train']['image_shape'], sel_box_idx, to_size)
    mask = np.amax(cropped_boxes[:, 2:4] - cropped_boxes[:, 0:2], axis=1) > 10
    cropped_boxes = cropped_boxes[mask]
    cropped_labels = cropped_labels[mask]

    cropped_boxes = np.array(cropped_boxes).astype(np.float32)
    cropped_labels = np.array(cropped_labels).astype(np.int32)
    imgByteArr = io.BytesIO()
    cropped_image.save(imgByteArr, format='JPEG')
    cropped_image = imgByteArr.getvalue()

    return {'image':cropped_image,
            'boxes':cropped_boxes,
            'labels':cropped_labels}

def work_fn(anno):
    image = Image.open(os.path.join(cfg['wider_train']['image_dir'], anno[0]))
    sample = {'image': image,
              'boxes': anno[1]['boxes'],
              'labels': anno[1]['labels']}
    samples = process_single_anno(sample)
    return samples


def main():
    tfrecordObj = ut.tf_ops.ImageBoxTfrecord()
    pool = mp.Pool(processes=12)
    for i in range(repeat):
        print('%d epochs' % i)
        samples = pool.map(work_fn, annotations, 500)
        samples = list(samples)
        tfrecordObj.write(outPath, samples)

if __name__ == '__main__':
    main()
