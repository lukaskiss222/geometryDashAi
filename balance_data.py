import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('./data.npy')
df = pd.DataFrame(train_data)
print(len(train_data))
print(df.head())
print(Counter(df[1].apply(str)))

for data in train_data:
    img = data[0]
    img = cv2.resize(img, (800,600)) 
    choice = data[1]
    cv2.imshow('test', img)
    print(choice)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

zero = []
ones = []

np.random.shuffle(train_data)
print(len(train_data))
df = pd.DataFrame(train_data)
print(df.head())
print(Counter(df[1].apply(str)))

for data in train_data:
    img = data[0] 
    choice = data[1]
    

    if choice == [1]:
        ones.append([img, [1,0]])
    elif choice == [0]:
        zero.append([img, [0,1]])
    else:
        print(choice)
        print("no matches!!!!!!!!!!!!!!!!!")
        
print("Zeros:", len(zero))
print("ones",len(ones))

zero = zero[:len(ones)]
ones = ones[:len(zero)]

final_data = zero + ones

shuffle(final_data)
print("###################################")
print(len(final_data))
df = pd.DataFrame(final_data)
print(df.head())
print(Counter(df[1].apply(str)))
np.save('data2.npy', final_data)


