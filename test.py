import numpy as np
import cv2
import time
from mynet import alexnet
import sys




if len(sys.argv) < 2:
    print(sys.argv[0],"<epoches>")
    sys.exit(0)


WIDTH = 80
HEIGHT = 60

LR = 1e-3
EPOCHS = int(sys.argv[1])
MODEL_NAME = 'geometry-dash-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2', EPOCHS)


train_data = np.load("./data2.npy")[:10]

#print(train_data[0][0].reshape(-1, WIDTH, HEIGHT,1)[0])
X = np.array([i[0] for i in train_data]).reshape(-1, WIDTH, HEIGHT, 1)





model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

for data in train_data:
    pre = model.predict(data[0].reshape(-1,WIDTH, HEIGHT, 1))
    print("############")
    print("Prediction:", pre)
    print("Data:", data[1])

