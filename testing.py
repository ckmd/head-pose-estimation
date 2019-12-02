import NumPyCNN as numpycnn
import numpy as np
import pandas as pd
import time, pickle, function, read_data_test, cv2, pose
import new_filters as nf
from PIL import Image, ImageFont, ImageDraw

def tanh(x):
    return ( 2 / 1 + np.exp(-2*x)) - 1

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

# feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
# labels = np.array([[1,0,0,1,1]])
# labels = labels.reshape(5,1)

start = time.time()

data = read_data_test.data
labels = read_data_test.label
feature_set = data

# print(feature_set.shape, labels.shape, feature_set[0])
# exit()

np.random.seed(42)

weights = pickle.load(open("syn0.pickle", "rb"))
weights2 = pickle.load(open("syn1.pickle", "rb"))
bias = pickle.load(open("bias.pickle", "rb"))
bias2 = pickle.load(open("bias2.pickle", "rb"))

epoch = 1 * len(feature_set)
# epoch = 100
cocok = 0
i = 0
for j in range(epoch):
    ravel_input = []
    ri = np.random.randint(len(feature_set))
    ri = i
    i += 1
    inputs = feature_set[ri] / 255
    # inputs = ( (inputs - np.amin(inputs) ) * 1 ) / ( np.amax(inputs) - np.amin(inputs) )

    # print(feature_set.shape, np.array([feature_set[1]]).shape)
    
    # convolutional 1
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
    real = np.argmax(labels[ri][1:22])+1
    if(z[0][0] > 0.5 and labels[ri][0] > 0.5) or (z[0][0] < 0.5 and labels[ri][0] < 0.5):
        if(z[0][0] > 0.5):
            print("face found ")
            poses = pose.poseplusmin(z[0][1:22].tolist())
            # Draw Pose est on image
            # text = "x:",poses[0],"\n y:",poses[1]
            cv2.putText(inputs,"x:"+str(poses[0])+" y:"+str(poses[1]),(10,127), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255),1)
            cv2.imwrite('21pose-test/poseresult/'+str(real)+'.jpg', inputs*255)
        cocok += 1
    else:
        cv2.imwrite('21pose-test/poseresult/'+str(real)+'.jpg', inputs*255)

    print("Accuraccy : ",cocok/(j+1)*100,'%')
end = time.time()
print("execution time ", end - start)