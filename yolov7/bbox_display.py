import random
import cv2
from matplotlib import pyplot as plt
import cv2
import os
import numpy as np


def draw_box_in_single_image(image_path, txt_path1, txt_path2):
    
    image1 = cv2.imread(image_path)
    image2 = cv2.imread(image_path)

    def read_list(txt_path):
        pos = []
        with open(txt_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()  
                if not lines:
                    break
                    pass
                p_tmp = [float(i) for i in lines.split(' ')]
                pos.append(p_tmp)  
                # Efield.append(E_tmp)
                pass
        return pos

    # txt转换为box
    def convert(size, box):
        xmin = (box[1]-box[3]/2.)*size[1]
        xmax = (box[1]+box[3]/2.)*size[1]
        ymin = (box[2]-box[4]/2.)*size[0]
        ymax = (box[2]+box[4]/2.)*size[0]
        box = (int(xmin), int(ymin), int(xmax), int(ymax))
        return box

    pos1 = read_list(txt_path1)
    pos2 = read_list(txt_path2)
    tl = int((image1.shape[0]+image1.shape[1])/2) + 1
    lf = max(tl-1, 1)
    for i in range(len(pos1)):
        label = str(int(pos1[i][0]))
        box = convert(image1.shape, pos1[i])
        image1 = cv2.rectangle(image1, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(image1, label, (box[0], box[1]-2), 0, tl, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        pass
    
    for i in range(len(pos2)):
        label = str(int(pos2[i][0]))
        box = convert(image2.shape, pos2[i])
        image2 = cv2.rectangle(image2, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(image2, label, (box[0], box[1]-2), 0, tl, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
        pass

    imgs = np.hstack([image1, image2])

    return imgs
    #cv2.imshow("images", imgs)
    #cv2.waitKey(0)

def cv2showimgs(scale, imglist, order):

    allimgs = imglist.copy()
    for i, img in enumerate(allimgs):
        if np.ndim(img) == 2:
            allimgs[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        allimgs[i] = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
    w, h = allimgs[0].shape[1], allimgs[0].shape[0]

    sub = int(order[0] * order[1] - len(imglist))
    if sub > 0:
        for s in range(sub):
            allimgs.append(np.zeros_like(allimgs[0]))
    elif sub < 0:
        allimgs = allimgs[:sub]
    imgblank = np.zeros((h * order[0], w * order[1], 3), np.uint8)
    for i in range(order[0]):
        for j in range(order[1]):
            imgblank[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = allimgs[i * order[1] + j]
    return imgblank


if __name__=='__main__':
    image_path = 'C:\\Users\\user5\\project\\data\\shu\\JPG'
    txt_path1 = 'C:\\Users\\user5\\project\\data\\shu\\txt'
    txt_path2 = 'C:\\Users\\user5\\project\\data\\shu\\image_txt_updata3'

    NUM = 9
    i = 0
    j = 0
    list = random.sample(range(0, 21083), NUM)
    List = sorted(list)
    print(List)
    img = []
    txt1 = []
    txt2 = []
    for filename in os.listdir(image_path):
        if j == NUM:
            break
        if i == List[j]:
            img.append(os.path.join(image_path, filename))
            (filename, extension) = os.path.splitext(filename)
            filename = filename + '.txt'
            txt1.append(os.path.join(txt_path1, filename))
            txt2.append(os.path.join(txt_path2, filename))
            j = j + 1
        else:
            i = i + 1
    print(img)

    images = []
    for i in range(NUM):
        images.append(draw_box_in_single_image(img[i], txt1[i], txt2[i]))
    #draw_box_in_single_image(image_path, txt_path1, txt_path2)

    show = cv2showimgs(scale=0.6, imglist=images, order=(3, 3))
    # cv2.namedWindow('images', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('images', show)
    cv2.imwrite(os.path.join('C:\\Users\\user5\\project\\yolov7\\runs3\\result', '9.jpg'), show)
    cv2.waitKey(0)
