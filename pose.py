import numpy as np

confidencesample = [
    0.0, 0.0, 0.0, 0.001, 0.0, 0.0 , 0.0,
    0.0, 0.0, 0.02, 0.9, 0.0002, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 0.0,
]

def poseplusmin(confidence):
    # print(len(confidence))
    # 21 map dengan nilai [x,y] masing  masing
    posemap = [
        [45,30], [30,30], [15,30], [0,30], [-15,30], [-30,30], [-45,30],
        [45,0],  [30,0],  [15,0],  [0,0],  [-15,0],  [-30,0],  [-45,0],
        [45,-30],[30,-30],[15,-30],[0,-30],[-15,-30],[-30,-30],[-45,-30],
    ]

    predpose = np.argmax(confidence)+1
    # predpose = np.argmax(z[0][1:21])+1
    predxplus = predpose + 1
    predxmin = predpose - 1
    predyplus = predpose + 7
    predymin = predpose - 7

    # Rules untuk area sekitar highest confidence
    if(predpose % 7 == 0):
        predxplus = None
        predxmin = predpose - 1
    if(predpose % 7 == 1):
        predxplus = predpose + 1;
        predxmin = None
    if(predpose < 8):
        predymin = None
        predyplus = predpose + 7
    if(predpose > 14):
        predymin = predpose - 7
        predyplus = None

    # Rumus untuk menghitung nilai x dan y
    xmin, xmax, ymin, ymax = 0,0,0,0

    if(predxmin!=None):
        xmin = 15 * confidence[predxmin-1] / (confidence[predpose-1] + confidence[predxmin-1])
    if(predxplus!=None):
        xmax = 15 * confidence[predxplus-1] / (confidence[predpose-1] + confidence[predxplus-1])
    if(predymin!=None):
        ymin = 30 * confidence[predymin-1] / (confidence[predpose-1] + confidence[predymin-1])
    if(predyplus!=None):
        ymax = 30 * confidence[predyplus-1] / (confidence[predpose-1] + confidence[predyplus-1])

    if(xmin < xmax):
        x = posemap[predpose-1][0] - xmax
    else:
        x = posemap[predpose-1][0] + xmin
    if(ymin < ymax):
        y = posemap[predpose-1][1] - ymax
    else:
        y = posemap[predpose-1][1] + ymin

    return round(x,3),round(y,3)

# poseplusmin(confidencesample)