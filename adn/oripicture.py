import scipy.io as scio
import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import re

def takeSize():
    dataFile = './data/deep_lesion/test'
    datapath = os.listdir(dataFile)
    for i in datapath:
        if os.path.splitext(i)[0] != '.mat' and os.path.splitext(i)[0] != 'gt':
            print(os.path.splitext(i)[0])
            datapath = os.path.join(dataFile,f"{os.path.splitext(i)[0]}.mat")
    data = scio.loadmat(datapath)
    image = data['image']
    return image.shape


def get_err(out_dir):
    outpath = './runs/deep_lesion/err'
    dataFile1 = './runs/deep_lesion/ori'
    dataFile2 = os.path.join('./runs/deep_lesion/',out_dir)
    datapath = os.listdir(dataFile1)
    datapath2 = os.listdir(dataFile2)
    for i in datapath:
        for j in datapath2:
            if re.match(os.path.splitext(i)[0],j) != None:
                pimg = os.path.join(dataFile1,i)
                pimg2 = os.path.join(dataFile2,j)
                img = cv2.imread(pimg)
                img2 = cv2.imread(pimg2)
                img2 = cv2.resize(img2,(img.shape[1],img.shape[0]))
                err = cv2.absdiff(img,img2)     #差值的绝对值
                errPath = os.path.join(outpath,f"{os.path.splitext(i)[0]}.png")
                cv2.imwrite(errPath, err)
                continue

