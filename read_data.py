import cv2, os, glob, re, time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

start = time.time()
def splitstring(word):
    label = np.zeros(22);
    x = re.split(" ", word)
    if(x[0] != 'nonface'):
        label[0] = 1
        label[int(x[0])] = 1
    return label

# read several image
img_dir = "21pose" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*jpg')
files = glob.glob(data_path)

img_dir = "21pose" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*png')
files_png = glob.glob(data_path)

data = []
label = []

for f1 in files:
    image = cv2.imread(f1)
    base = os.path.basename(f1)
    base = os.path.splitext(base)
    # title = splitstring(base[0])
    title = splitstring(base[0])
    # print(title)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    layer = image
    layer = cv2.resize(layer,(100,100))

    height = layer.shape[0]
    width = layer.shape[1]
    pad = 16

    new_layer = np.ones([height + 2*pad, width + 2*pad])
    new_layer += 128
    for i in range(height):
        for j in range(width):
            new_layer[i + pad][j + pad] = layer[i][j]

    # plt.imshow(new_layer, cmap='gray')
    # plt.show()
    # cv2.waitKey(1000)

    data.append(new_layer)
    label.append(title)

for f1 in files_png:
    image = cv2.imread(f1)
    base = os.path.basename(f1)
    base = os.path.splitext(base)
    # title = splitstring(base[0])
    title = splitstring(base[0])
    # print(title)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    layer = image
    layer = cv2.resize(layer,(100,100))

    height = layer.shape[0]
    width = layer.shape[1]
    pad = 16

    new_layer = np.ones([height + 2*pad, width + 2*pad])
    new_layer += 128
    for i in range(height):
        for j in range(width):
            new_layer[i + pad][j + pad] = layer[i][j]

    data.append(new_layer)
    label.append(title)

data = np.array(data)
label = np.array(label)
end = time.time()
print("read data complete " , round(end-start,2) , "s, total : ", len(data))
# cv2.imshow('data',data[0])
# cv2.waitKey(1000)