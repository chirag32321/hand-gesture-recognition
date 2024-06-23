#!/usr/bin/env python
# coding: utf-8

# # Task - 4: Develop a hand gesture recognition model that can accurately identify and classify different hand gestures from image or video data, enabling intuitive human-computer interaction and gesture-based control systems.
# 
# Dataset: Dataset :-  https://www.kaggle.com/gti-upm/leapgestrecog
# 
# 
# ## Content
#  
# The database is composed by 10 different hand-gestures that were performed by 10 different sunjects (5 men and 5 women)

# In[89]:


import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os 
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns

from keras.layers import Conv2D, Activation, MaxPool2D, Dense, Flatten, Dropout
# from keras.utils import to_categorical

import warnings
warnings.filterwarnings("ignore")


# In[65]:


# pip install keras


# In[66]:


# !pip install tensorflow


# In[67]:


Categories = ["01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb", "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"]
img_size = 50

data_path = "dataset/leapGestRecog/leapGestRecog"


# ### Load the Data

# In[68]:


# Loading the images and their class(0 - 9)
image_data = []
for dr in os.listdir(data_path):
    for category in Categories:
        class_index = Categories.index(category)
        path = os.path.join(data_path, dr, category)
        for img in os.listdir(path):
            try:
                # print("hi")
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                image_data.append([cv2.resize(img_arr, (img_size, img_size)), class_index])
            except Exception as e:
                pass



# In[69]:


image_data[0]


# In[70]:


# Shuffle the input data
import random
random.shuffle(image_data)


# In[71]:


input_data = []
label = []
for X, y in image_data:
    input_data.append(X)
    label.append(y)


# In[72]:


label[:10]


# In[73]:


plt.figure(1, figsize=(10,10))
for i in range(1,10):
    plt.subplot(3,3,i)
    plt.imshow(image_data[i][0], cmap = "hot")
    plt.xticks([])
    plt.yticks([])
    plt.title(Categories[label[i]][3:])


# In[74]:


# Mormalizing the data
input_data = np.array(input_data)
label = np.array(label)
input_data = input_data/255.0
input_data.shape


# In[75]:


import tensorflow as tf


# In[76]:


# One hot Encoding
label = tf.keras.utils.to_categorical(label)


# In[77]:


label = label.astype("i1")


# In[78]:


label[0]


# In[79]:


# Reshaping the data
input_data.shape = (-1,img_size, img_size, 1)


# In[80]:


# Splitting the input_data to train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(input_data, label, test_size=0.3, random_state=0)


# ### The Model

# In[81]:


model = keras.models.Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=(img_size, img_size, 1)))
model.add(Activation("relu"))

model.add(Conv2D(filters=32, kernel_size=(3,3)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters=64, kernel_size=(3,3)))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy",
             optimizer = "rmsprop",
             metrics = ["accuracy"])


# In[ ]:





# In[82]:


model.fit(X_train, y_train, epochs=7, batch_size=32, validation_data=(X_test, y_test))


# In[83]:


print(model.summary())


# In[84]:


plt.plot(model.history.history["loss"])
plt.plot(model.history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend(["train", "test"])
plt.show()


# In[85]:


plt.plot(model.history.history["accuracy"])
plt.plot(model.history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["train", "test"])
plt.show()


# In[86]:


# Calculate loss and accuracy on test data

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test accuracy: {:2.2f}%".format(test_accuracy*100))


# ### Confusion Matrix

# In[87]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
cat = [c[3:] for c in Categories]
plt.figure(figsize=(10,10))
cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
sns.heatmap(cm, annot=True, xticklabels=cat, yticklabels=cat)
plt.plot()


# In[ ]:




