import numpy as np
from mynet_softmax import alexnet
from collections import Counter

WIDTH = 80
HEIGHT = 60
LR = 1e-5
EPOCHS = 6
MODEL_NAME = 'geometry-dash-{}-{}-{}-epochs_softmax.model'.format(LR, 'alexnetv2', EPOCHS)


model = alexnet(WIDTH, HEIGHT, LR)

train_data = np.load('./data2.npy')



train = train_data[:-500]
test = train_data[-500:]


X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
Y = [i[1] for i in train]

print(Counter(map(lambda x:str(x),Y)))

test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = [i[1] for i in test]

model.fit({'input':X}, {'targets':Y}, n_epoch = EPOCHS,
        validation_set=({'input':test_x}, {'targets':test_y}),
        snapshot_step=500, show_metric=True,
        run_id=MODEL_NAME)

# tenserboard --logdir=<log file>


model.save(MODEL_NAME)


train_data = np.load("./data2.npy")[:10]
 
#print(train_data[0][0].reshape(-1, WIDTH, HEIGHT,1)[0])
X = np.array([i[0] for i in train_data]).reshape(-1, WIDTH, HEIGHT, 1)
#model = alexnet(WIDTH, HEIGHT, LR)
#model.load(MODEL_NAME)
print([1,0],"jump")
for data in train_data:
    pre = model.predict(data[0].reshape(-1,WIDTH, HEIGHT, 1))
    print("############")
    print("Prediction:", pre)
    print("Data:", data[1])



