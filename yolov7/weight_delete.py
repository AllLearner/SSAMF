import os

import numpy as np

def weight_deletd(path1, path2):
    names = os.listdir('C:\\Users\\user5\\project\\data\\shu\\txt')
    for name in names:
        txt_path1 = path1 + '\\' + name
        txt_path2 = path2 + '\\' + name

        if os.path.exists(txt_path1):
            with open(txt_path2, 'w+') as f2:
                with open(txt_path1, 'r') as f1:
                    for line1 in f1.readlines():
                        line1 = line1.strip('\n')
                        line1 = line1[:-5]
                        f2.write(line1)
                        f2.write('\n')

            with open(txt_path1, 'w') as f1:
                with open(txt_path2, 'r') as f2:
                    for line1 in f2.readlines():
                        f1.write(line1)

if __name__ == '__main__':
    path1 = 'C:\\Users\\user5\\project\\yolov7\\runs7\\txt_cache'
    path2 = 'C:\\Users\\user5\\project\\yolov7\\runs7\\txt'
    weight_deletd(path1, path2)