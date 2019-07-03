
import os, json
import numpy as np
from utils import np_ops
from datasets.widerface import read_anno

cfg = json.load(open('base.json'))

detect_result_dir = '/home/luojiapeng/project/CNet/datasets/dsfd_wider_train'
origin_anno_txt = cfg['wider_train']['txt_path']
output_txt = cfg['dsfd']['txt_path']
detect_result_thres = 0.7


def read_detect_result():
    detect_dict= dict()
    for event in os.listdir(detect_result_dir):
        event_dir = os.path.join(detect_result_dir, event)
        for txt in os.listdir(event_dir):
            if not txt.endswith('.txt'): continue
            key = event+'/'+txt[:-4]+'.jpg'
            txt_path = os.path.join(event_dir, txt)
            boxes = _read_single_detect(txt_path)
            boxes = np.array(boxes)
            detect_dict[key] = boxes
    return detect_dict


def _read_single_detect(path):
    lines = open(path).readlines()
    lines = [l.rstrip() for l in lines]
    boxes_num = int(lines[1])
    boxes = []
    for i in range(2, 2+boxes_num):
        xmin, ymin, w, h, s = [float(x) for x in lines[i].split()]
        xmax, ymax = xmin+w, ymin+h
        boxes.append([ymin, xmin, ymax, xmax, s])
    return boxes


def write_new_txt(origin_dict, detect_dict, output_txt):
    fid = open(output_txt, 'w')
    for fname in origin_dict.keys():
        boxes1 = detect_dict[fname]
        boxes2 = origin_dict[fname]['boxes']
        boxes1, score1 = boxes1[:, :4], boxes1[:, 4]
        boxes1 = boxes1[score1 > detect_result_thres]
        if len(boxes1) == 0:
            iou = np.zeros(len(boxes2))
        else:
            iou = np_ops.iou(boxes1, boxes2)
            iou = np.max(iou, axis=0)
        fid.write(fname+'\n')
        fid.write(str(len(boxes2))+'\n')
        for i, box in enumerate(boxes2.tolist()):
            ymin, xmin, ymax, xmax = box[0], box[1], box[2], box[3]
            w = xmax - xmin
            h = ymax - ymin
            tag = 1 if iou[i]>0.5 else 0
            fid.write('%d %d %d %d %d\n'%(xmin, ymin, w, h, tag))
    fid.close()

def run():
    origin_dict = read_anno(origin_anno_txt, 5, has_label=False)
    detect_dict = read_detect_result()
    write_new_txt(origin_dict, detect_dict, output_txt)

if __name__ == '__main__':
     run()