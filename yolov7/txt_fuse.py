import os

import numpy as np

def compute_iou(box10, box20, wh=True):
    """
    compute the iou of two boxes.
    Args:
            box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
            wh: the format of coordinate.
    Return:
            iou: iou of box1 and box2.
    """
    box1 = []
    box2 = []
    for item in box10:
        box1.append(item * 512)
    for item in box20:
        box2.append(item * 512)

    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))  
    iou = inter_area / (area1 + area2 - inter_area + 1e-6) 

    return iou

def txt_fuse0(path1, path2, path3, IOU):
    names = os.listdir('C:\\Users\\user5\\project\\data\\shu\\txt')
    for name in names:
        txt_path1 = path1 + '\\' + name
        txt_path2 = path2 + '\\' + name
        txt_path3 = path3 + '\\' + name

        if os.path.exists(txt_path1):
            with open(txt_path3, 'w+') as f3:
                with open(txt_path1, 'r') as f1:
                    for line1 in f1.readlines():
                        line1 = line1.strip('\n')  
                        box1 = line1[2:].split()
                        box1 = list(map(float, box1))
                        box1 = box1[0:4]
                        #print(box1)
                        overlap = False
                        with open(txt_path2, 'r') as f2:
                            for line2 in f2.readlines():
                                line2 = line2.strip('\n')  
                                box2 = line2[2:].split()
                                box2 = list(map(float, box2))
                                box2 = box2[0:4]
                                #print(box2)
                                iou = compute_iou(box1, box2, wh=True)
                                #print(iou)
                                if iou > IOU:
                                    overlap = True
                                    break
                            if overlap is False:
                                line1 = line1 + ' ' + format(1.00, '.2f')
                                f3.write(line1)
                                f3.write('\n')
                            else:
                                line1 = line1 + ' ' + format(0.80, '.2f')
                                f3.write(line1)
                                f3.write('\n')

            with open(txt_path2, 'r') as f2:
                for line1 in f2.readlines():
                    line1 = line1.strip('\n') 
                    box1 = line1[2:].split()
                    box1 = list(map(float, box1))
                    box1 = box1[0:4]
                    #print(box1)
                    overlap = False
                    with open(txt_path3, 'r+') as f3:
                        for line2 in f3.readlines():
                            line2 = line2.strip('\n')  
                            box2 = line2[2:].split()
                            box2 = list(map(float, box2))
                            box2 = box2[0:4]
                            #print(box2)
                            iou = compute_iou(box1, box2, wh=True)
                            #print(iou)
                            if iou > IOU:
                                overlap = True
                                break
                        if overlap is False:
                            line1 = line1 + ' ' + format(0.75, '.2f')
                            f3.write(line1)
                            f3.write('\n')

            with open(txt_path2, 'w') as f2:
                with open(txt_path3, 'r') as f3:
                    for line1 in f3.readlines():
                            f2.write(line1)

        else:
            with open(txt_path3, 'w+') as f3:
                with open(txt_path2, 'r') as f1:
                    for line1 in f1.readlines():
                        line1 = line1.strip('\n')
                        line1 = line1 + ' ' + format(0.75, '.2f')
                        f3.write(line1)
                        f3.write('\n')

            with open(txt_path2, 'w') as f2:
                with open(txt_path3, 'r') as f3:
                    for line1 in f3.readlines():
                            f2.write(line1)

    print('txt fuse finished !!!')


def txt_fuse(path1, path2, path3, IOU):
    names = os.listdir('C:\\Users\\user5\\project\\data\\shu\\txt')
    for name in names:
        txt_path1 = path1 + '\\' + name
        txt_path2 = path2 + '\\' + name
        txt_path3 = path3 + '\\' + name

        if os.path.exists(txt_path1):
            with open(txt_path3, 'w+') as f3:
                with open(txt_path1, 'r') as f1:
                    for line1 in f1.readlines():
                        line1 = line1.strip('\n')  
                        box1 = line1[2:].split()
                        box1 = list(map(float, box1))
                        box1 = box1[0:4]
                        # print(box1)
                        overlap = False
                        with open(txt_path2, 'r') as f2:
                            for line2 in f2.readlines():
                                line2 = line2.strip('\n')  
                                box2 = line2[2:].split()
                                box2 = list(map(float, box2))
                                box2 = box2[0:4]
                                # print(box2)
                                iou = compute_iou(box1, box2, wh=True)
                                # print(iou)
                                if iou > IOU:
                                    overlap = True
                                    break
                            if overlap is False:
                                line1 = line1 + ' ' + format(1.00, '.2f')
                                f3.write(line1)
                                f3.write('\n')
                            else:
                                line1 = line1 + ' ' + format(0.80, '.2f')
                                f3.write(line1)
                                f3.write('\n')

            with open(txt_path2, 'r') as f2:
                for line1 in f2.readlines():
                    line1 = line1.strip('\n')  
                    box1 = line1[2:].split()
                    box1 = list(map(float, box1))
                    box1 = box1[0:4]
                    # print(box1)
                    overlap = False
                    with open(txt_path3, 'r+') as f3:
                        for line2 in f3.readlines():
                            line2 = line2.strip('\n')  
                            box2 = line2[2:].split()
                            box2 = list(map(float, box2))
                            box2 = box2[0:4]
                            # print(box2)
                            iou = compute_iou(box1, box2, wh=True)
                            # print(iou)
                            if iou > IOU:
                                overlap = True
                                break
                        if overlap is False:
                            line1 = line1[:-4] + format(0.75, '.2f')
                            f3.write(line1)
                            f3.write('\n')

            with open(txt_path2, 'w') as f2:
                with open(txt_path3, 'r') as f3:
                    for line1 in f3.readlines():
                        f2.write(line1)

    print('txt fuse finished !!!')

if __name__ == '__main__':
    box1 = [0.2607421875, 0.361328125, 0.16015625, 0.189453125]
    box2 = [0.269531, 0.361328, 0.171875, 0.183594]
    #IOU = compute_iou(box1, box2, wh=True)
    detect_labels_path = 'C:\\Users\\user5\\project\\yolov7\\runs2\\detect\\stage\\labels'
    train_label_path = 'C:\\Users\\user5\\project\\data\\shu\\image_txt'
    txt_fuse0(detect_labels_path, train_label_path, 0.15)
