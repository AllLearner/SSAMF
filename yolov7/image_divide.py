import os
from sklearn import model_selection

def randomSplit(file,path_all,train_txt,val_txt,random_state): 
        sample = []
        train_txt = train_txt + '\\train.txt'
        val_txt = val_txt + '\\test.txt'
        trainSet = open(train_txt, 'w')  
        valSet = open(val_txt, 'w')    
        for Image in os.listdir(file):
                sample.append(Image)
        sample_train, sample_test = model_selection.train_test_split(sample, test_size=0.1, random_state=random_state)
        for i in sample_train:
                path = os.path.join(path_all, i)
                trainSet.write(path)
                trainSet.write('\n') 
        for i in sample_test:
                path = os.path.join(path_all, i)
                valSet.write(path)
                valSet.write('\n')

        return train_txt, val_txt

if __name__=='__main__':
    path_jpg = 'C:\\Users\\user5\\project\\data\\shu\\JPG'
    path_all = 'C:\\Users\\user5\\project\\data\\shu\\image_txt_updata3'
    train_txt = 'C:\\Users\\user5\\project\\data\\shu'
    val_txt = 'C:\\Users\\user5\\project\\data\\shu'
    print(randomSplit(path_jpg, path_all, train_txt, val_txt, random_state=24))