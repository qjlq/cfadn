import scipy.io as scio
#import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
dataFile = './data/deep_lesion/train/temp'
datapath = os.listdir(dataFile)
for i in datapath:
    #print(os.path.splitext(i)[1])
    #print(i)
    if os.path.splitext(i)[0] != '.mat' and os.path.splitext(i)[0] != 'gt':
        print(os.path.splitext(i)[0])
        datapath = os.path.join(dataFile,f"{os.path.splitext(i)[0]}.mat")
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
    dataFile = './data/deep_lesion/train/temp'
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

# image = image * 255
# new = Image.fromarray(image.astype(np.uint8))
# new = Image.fromarray(image.astype(np.uint8)).convert('RGB')
# plt.imshow(image,cmap=plt.cm.gray,interpolation='nearest')
# new.show()

# print(image.shape)
# plt.imshow(image)
# plt.show
