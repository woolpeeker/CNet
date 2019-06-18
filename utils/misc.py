
import os, re, glob

__all__ = ['get_global_dict',
           'add_suffix_ckpt_path']

_global_dict = dict()
def get_global_dict():
    return _global_dict


def add_suffix_ckpt_path(path='./ckpt', num=None):
    path = path[:-1] if path[-1] in ['/','\\'] else path
    if num is not None:
        return path+'_'+str(num)
    max_num=-1
    for p in glob.glob(path+'_*'):
        if os.path.isdir(p):
            match=re.search(path+'_(\\d+)',p)
            if match:
                num=int(match.group(1))
                max_num=num if num>max_num else max_num
    max_num=max_num+1
    result_path=path+'_%d' % max_num
    return result_path