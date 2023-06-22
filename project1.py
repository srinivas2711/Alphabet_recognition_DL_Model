#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# In[2]:


data = pd.read_csv(r"C:\Users\Desktop\A_Z Handwritten Data\A_Z Handwritten Data.csv").astype('float32')
#print(data.head(10))
print(data.shape[1])
print(data['0'])


# In[4]:


X = data.drop('0',axis = 1)
y = data['0']


# In[4]:


print(0==0.0)


# In[5]:


train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2)

train_x = np.reshape(train_x.values, (train_x.shape[0], 28,28))
test_x = np.reshape(test_x.values, (test_x.shape[0], 28,28))

print("Train data shape: ", train_x.shape)
print("Test data shape: ", test_x.shape)


# In[6]:


print(y.shape)


# In[6]:


word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}


# In[8]:


import numpy as np

x = 5
x_array = np.int0(x)
print(x_array)
print(type(x_array))


# In[10]:


print(train_x[10:10])


# In[7]:


shuff = shuffle(train_x[:100])

fig, ax = plt.subplots(3,3, figsize = (10,10))
axes = ax.flatten()

for i in range(9):
    _, shu = cv2.threshold(shuff[i], 30, 200, cv2.THRESH_BINARY)
    axes[i].imshow(np.reshape(shuff[i], (28,28)), cmap="Greys")
plt.show()


# In[8]:


train_X = train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1)
print("New shape of train data: ", train_X.shape)

test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2],1)
print("New shape of train data: ", test_X.shape)


# In[9]:


train_ydata = to_categorical(train_y, num_classes = 26, dtype='int')
print(train_yOHE[1])
print("New shape of train labels: ", train_ydat.shape)

test_ydata = to_categorical(test_y, num_classes = 26, dtype='int')
print("New shape of test labels: ", test_ydata.shape)


# In[12]:


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(64,activation ="relu"))
model.add(Dense(128,activation ="relu"))

model.add(Dense(26,activation ="softmax"))


# In[13]:


model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_X, train_yOHE, epochs=1,  validation_data = (test_X,test_yOHE))


# In[16]:


a=model.predict(r'C:\Users\srinivasa.moorthy\OneDrive - DISYS\Desktop\one.png')
print(a)


# In[ ]:


model.summary()
model.save(r'model_hand.h5')


# In[ ]:


import cv2
# Read the image
image = cv2.imread(r'C:\Users\Desktop\e.jpg')
# Display the image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[23]:


img = cv2.imread(r'C:\Users\srinivasa.moorthy\OneDrive - DISYS\Desktop\img1.png')
img_copy = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (400,440))


# In[24]:


img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
_, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

img_final = cv2.resize(img_thresh, (28,28))
img_final =np.reshape(img_final, (1,28,28,1))


# In[25]:


img_pred = word_dict[np.argmax(model.predict(img_final))]

cv2.putText(img, "Mymodel: ", (20,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color = (0,0,0))
cv2.putText(img, "Alphabet prediction: " + img_pred, (20,410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color = (0,0,0))
cv2.imshow('Image for recognition: ', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




