import scipy.io as scio
import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import re


# dataFile = './data/deep_lesion/train/temp'
# datapath = os.listdir(dataFile)
# for i in datapath:
#     #print(os.path.splitext(i)[1])
#     #print(i)
#     if os.path.splitext(i)[0] != '.mat' and os.path.splitext(i)[0] != 'gt':
#         print(os.path.splitext(i)[0])
#         datapath = os.path.join(dataFile,f"{os.path.splitext(i)[0]}.mat")
#print(datapath)
# print(os.path.splitext(datapath[1])[0])
#data = scio.loadmat(dataFile)
# image = data['image']

# print(image.shape[0])

# fo = open("tmp", "w")
# fo.write(str(data))
 
# # 关闭打开的文件
# fo.close()
#print(data)

def takeSize():
    dataFile = './data/deep_lesion/test'
    datapath = os.listdir(dataFile)
    for i in datapath:
        if os.path.splitext(i)[0] != '.mat' and os.path.splitext(i)[0] != 'gt':
            print(os.path.splitext(i)[0])
            datapath = os.path.join(dataFile,f"{os.path.splitext(i)[0]}.mat")
    #path = os.listdir(ata_dict)
    data = scio.loadmat(datapath)
    image = data['image']
    #print(image)
    return image.shape


def get_err():
    outpath = './run/deep_lesion/err'
    dataFile1 = './run/deep_lesion/ori'
    dataFile2 = './run/deep_lesion/test'
    datapath = os.listdir(dataFile1)
    datapath2 = os.listdir(dataFile2)
    for i in datapath:
        for j in datapath2:
            if re.match(os.path.splitext(i)[0],j).group() != None:
                img = cv2.imread(i)
                img2 = cv2.imread(j)
                img2 = cv2.resize(img2,(img.shape[1],img.shape[0]))
                err = cv2.absdiff(img,img2)     #差值的绝对值
                cv2.imwrite(outpath, err)
                continue
# image = image * 255
# new = Image.fromarray(image.astype(np.uint8))
# new = Image.fromarray(image.astype(np.uint8)).convert('RGB')
# plt.imshow(image,cmap=plt.cm.gray,interpolation='nearest')
# new.show()

# print(image.shape)
# plt.imshow(image)
# plt.show
