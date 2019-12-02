# import the necessary packages
from helpers import pyramid, sliding_window
from PIL import Image, ImageDraw
import new_filters as nf
import NumPyCNN as numpycnn
import argparse, time, cv2, pickle, os, glob
import numpy as np

start = time.time()
def sigmoid(x):
    return 1/(1+np.exp(-x))

def averagePosition(myList):
    return np.mean(myList, axis=0)

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the image")
# args = vars(ap.parse_args())

# Load Weight and Bias
weights = pickle.load(open("syn0.pickle", "rb"))
weights2 = pickle.load(open("syn1.pickle", "rb"))
bias = pickle.load(open("bias.pickle", "rb"))
bias2 = pickle.load(open("bias2.pickle", "rb"))

# load the image and define the window width and height
img_dir = "slidingperson2" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*jpg')
files = glob.glob(data_path)
gambar = 1

for f1 in files:
    print(gambar)
    gambar += 1
    image = cv2.imread(f1)
    base = os.path.basename(f1)
    base = os.path.splitext(base)
    # title = splitstring(base[0])
    title = base[0]

    # image = cv2.imread(args["image"])
    (winW, winH) = (131, 131)
    # resized = cv2.resize(image,(1294,1097))
    resized = image
    bounding_box = []
    scala = 1
    # loop over the image pyramid
    # for resized in pyramid(image, scale=1.25):
        # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=20, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
        # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
        # WINDOW
        # Start Classifier
        ravel_input = []
        window = cv2.resize(window,(100,100))
        window = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)

        # adding padding 16
        height = window.shape[0]
        width = window.shape[1]
        pad = 16

        new_layer = np.ones([height + 2*pad, width + 2*pad])
        new_layer += 128
        for i in range(height):
            for j in range(width):
                new_layer[i + pad][j + pad] = window[i][j]

        inputs = ( (new_layer - np.amin(new_layer) ) * 1 ) / ( np.amax(new_layer) - np.amin(new_layer) )

        l1_feature_map = numpycnn.conv(inputs, nf.filter1)
        l1_feature_map_relu_pool = numpycnn.pooling(l1_feature_map, 2, 2)

        l1_feature_map_i = numpycnn.conv(inputs, nf.filter1_i)
        l1_feature_map_relu_pool_i = numpycnn.pooling(l1_feature_map_i, 2, 2)

        magnitude = np.sqrt((l1_feature_map_relu_pool.T ** 2) + (l1_feature_map_relu_pool_i.T ** 2))
        phase = np.arctan(l1_feature_map_relu_pool_i.T / l1_feature_map_relu_pool.T)

        # Normalize 0 to 1
        magnitude = ( (magnitude - np.amin(magnitude) ) * 1 ) / ( np.amax(magnitude) - np.amin(magnitude) )
        # magnitude -= 1
        phase = ( (phase - np.amin(phase) ) * 1 ) / ( np.amax(phase) - np.amin(phase) )
        # phase -= 1

        for in1, conv1 in enumerate(magnitude):
            # print(np.amax(conv1), np.amin(conv1))
            # cv2.imwrite('magnitude'+str(in1)+'.jpg', conv1 * 255)
            ravel_input.append(conv1)

        for in1, conv1 in enumerate(phase):
            # print(np.amax(conv1), np.amin(conv1))
            # cv2.imwrite('phase'+str(in1)+'.jpg', conv1 * 255)
            ravel_input.append(conv1)

        ravel_input = np.array([np.array(ravel_input).ravel()])
        # print(np.amin(ravel_input), np.amax(ravel_input), np.mean(ravel_input))

        # feedforward step1
        l1 = sigmoid(np.dot(ravel_input, weights) + bias)
        z = sigmoid(np.dot(l1, weights2) + bias2)

        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        # if(z[0][0] >= 0.85):
        # cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 255), 4)
        # bounding_box.append(np.array([x*scala,y*scala,(x+winW)*scala, (y+winH)*scala,z]))
        bounding_box.append([x*scala,y*scala,(x+winW)*scala, (y+winH)*scala,z[0][0]])
        # print(bounding_box)
            # print(bounding_box)
        cv2.imshow("window", clone)
        cv2.waitKey(10)

        current = time.time()
        print("duration : ",round(current - start,2)," s",z)
    scala = scala * 1.25

        # since we do not have a classifier, we'll just draw the window
    im = Image.open(f1)
    d = ImageDraw.Draw(im)

    map = Image.new('RGB', (384, 288), (0, 0, 0))
    drawMap = ImageDraw.Draw(map)

    for bbox in bounding_box:
        print(int(bbox[4] * 255))
        conf = int(bbox[4] * 255)
        drawMap.point(((bbox[0] + bbox[2])/2, (bbox[1]+bbox[3])/2), fill=(conf,conf,conf))
        if(bbox[4] >= 0.95): #nanti tingkatkan ke 0.95
            d.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), fill=None, outline=(255, 255, 255))

    # bounding the largest confidence score
    sortedconfmap = sorted(bounding_box,key=lambda l:l[4], reverse=True)
    if(len(sortedconfmap)>=1): #red
        avg = averagePosition(sortedconfmap[:1])
        d.rectangle((sortedconfmap[0][0], sortedconfmap[0][1], sortedconfmap[0][2], sortedconfmap[0][3]), fill=None, outline=(255,0,0))
    if(len(sortedconfmap)>=2): #green
        avg = averagePosition(sortedconfmap[:2])
        d.rectangle((sortedconfmap[1][0], sortedconfmap[1][1], sortedconfmap[1][2], sortedconfmap[1][3]), fill=None, outline=(0,255,0))
    if(len(sortedconfmap)>=3): #blue
        avg = averagePosition(sortedconfmap[:3])
        d.rectangle((sortedconfmap[2][0], sortedconfmap[2][1], sortedconfmap[2][2], sortedconfmap[2][3]), fill=None, outline=(0,0,255))
    if(len(sortedconfmap)>=4): #purple
        avg = averagePosition(sortedconfmap[:4])
        d.rectangle((sortedconfmap[3][0], sortedconfmap[3][1], sortedconfmap[3][2], sortedconfmap[3][3]), fill=None, outline=(255,0,255))
    if(len(sortedconfmap)>=5): #cyan
        avg = averagePosition(sortedconfmap[:5])
        d.rectangle((sortedconfmap[4][0], sortedconfmap[4][1], sortedconfmap[4][2], sortedconfmap[4][3]), fill=None, outline=(0,255,255))
    # nanti kuning untuk average position
    d.rectangle((avg[0], avg[1], avg[2], avg[3]), fill=None, outline=(255,255,0))

    im.save("slidingperson2/slidingmaps/"+title+".jpg")
    map.save("slidingperson2/slidingmaps/map2d "+title+".jpg")

    pickle_out = open("slidingperson2/slidingmaps/picklemap2d "+title+".pickle", "wb")
    pickle.dump(sortedconfmap, pickle_out)

exit()
