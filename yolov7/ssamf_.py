import os
import shutil

import ruamel.yaml
from txt_fuse import compute_iou
from image_divide import randomSplit
import datetime

from train_pack import train_pack
from detect_pack import detect_pack


def change_data(path, data1, data2):
    file_name = path
    config, ind, bsi = ruamel.yaml.util.load_yaml_guess_indent(open(file_name))
    config['train'] = data1
    config['val'] = data2

    yaml = ruamel.yaml.YAML()
    yaml.indent(mapping=ind, sequence=ind, offset=bsi)
    with open(path, 'w') as fp:
        yaml.dump(config, fp)


def label_statistics(label_path):
    line_count = 0
    for filename in os.listdir(label_path):
        if filename.endswith(".txt"):
            path = os.path.join(label_path, filename)
            fp = open(path, 'r+', encoding='utf-8')
            for line in fp.readlines():
                if not line.split():
                    line.strip()
                    continue
                else:
                    line_count += 1
                fp.close()
    return line_count

if __name__ == '__main__':
    starttime = datetime.datetime.now()
    result_path = 'runs'
    device = '9'
    train_weight_path = 'yolov7.pt'
    epoch = 50

    source_path = 'C:\\Users\\user5\\project\\data\\shu\\JPG'
    conf_thres = 0.30
    iou_thres = 0.45

    work_path = os.getcwd()
    project1 = result_path + '/train'
    project2 = result_path + '/detect'
    train_result_path0 = work_path + '\\' + result_path + '\\train\\stage'
    labels_path0 = work_path + '\\' + result_path + '\\detect\\stage'
    train_label_path = 'C:\\Users\\user5\\project\\data\\shu\\image_txt_updata'

    label_num_path = work_path + '\\' + result_path + '\\label_num.txt'

    with open(label_num_path, 'a') as f:
        f.write(str(starttime))
        f.write('\n')
        label_num = label_statistics(train_label_path)
        f.write(str(label_num))
        f.write('\n')

    for i in range(6):
        # train and train path updata
        train_pack(train_weight_path, epoch, device, project1)
        if i == 0:
            train_weight_path = train_result_path0
        else:
            train_weight_path = train_result_path0 + str(i + 1)
        train_weight_path = train_weight_path + '\\weights\\best.pt'

        # detect
        detect_pack(train_weight_path, source_path, conf_thres, iou_thres, device, project2)
        if i == 0:
            labels_path = labels_path0
        else:
            labels_path = labels_path0 + str(i + 1)
        detect_labels_path = labels_path + '\\labels'

        detect_JPG_path = labels_path + '\\JPG'
        os.makedirs(detect_JPG_path)
        for filename in os.listdir(detect_labels_path):
            if filename.endswith(".txt"):
                jpgname = filename.replace('.txt', '.jpg')
                shutil.copy(source_path + '\\' + jpgname, detect_JPG_path + '\\' + jpgname)

        for filename in os.listdir(detect_JPG_path):
            shutil.copy(detect_JPG_path + '\\' + filename, detect_labels_path + '\\' + filename)


        train_path, label_path = randomSplit(detect_JPG_path, detect_labels_path, labels_path, labels_path,
                                             random_state=i + 1)

        # yaml path updata
        yamlPath = work_path + '\\shu.yaml'
        change_data(yamlPath, train_path, label_path)

        with open(label_num_path, 'a') as f:
            label_num = label_statistics(detect_labels_path)
            f.write(str(label_num))
            f.write('\n')
            endtime = datetime.datetime.now()
            times = endtime - starttime
            f.write(str(times))
            f.write('\n')

    endtime = datetime.datetime.now()
    with open(label_num_path, 'a') as f:
        f.write(str(endtime))
        f.write('\n')




