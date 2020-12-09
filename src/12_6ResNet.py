#!/usr/bin/env python
# coding: utf-8

# In[77]:


import pandas as pd
import numpy as np 
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential 
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense 
from keras import applications 
from keras.utils.np_utils import to_categorical 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import math 
import datetime
import time
#from tensorflow.keras.layers import Conv2D, Flatten,Dense, MaxPool2D,BatchNormalization, GlobalAveragePooling2D

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

#from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.models import Sequential 
#from tensorflow.keras.models import Model
import h5py


# In[78]:


#Default dimensions we found online
img_width, img_height = 224, 224 
 
#Create a bottleneck file
top_model_weights_path = r'C:\Users\ccastano\Desktop\data.h5'
# loading up our datasets
train_data_dir = r'C:\Users\ccastano\Desktop\basedata\training'
validation_data_dir = r'C:\Users\ccastano\Desktop\basedata\validation'
test_data_dir = r'C:\Users\ccastano\Desktop\basedata\testing'
 
# number of epochs to train top model 
epochs = 7 #this has been changed after multiple model run 
# batch size used by flow_from_directory and predict_generator 
batch_size = 50 


# In[80]:


#Loading ResNet50 model
ResNet50 = ResNet50(include_top=False, weights='imagenet',input_shape=(224,224,3))
datagen = ImageDataGenerator(rescale=1. / 255) 
#needed to create the bottleneck .npy files


# In[81]:


#__this can take an hour and half to run so only run it once. 
#once the npy files have been created, no need to run again. Convert this cell to a code cell to run.__
start = datetime.datetime.now()
 
generator = datagen.flow_from_directory( 
    train_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode=None, 
    shuffle=False) 
 
nb_train_samples = len(generator.filenames) 
num_classes = len(generator.class_indices) 
 
predict_size_train = int(math.ceil(nb_train_samples / batch_size)) 
 
bottleneck_features_train = ResNet50.predict_generator(generator, predict_size_train) 
 
np.save('bottleneck_features_train1.npy', bottleneck_features_train)
end= datetime.datetime.now()
elapsed= end-start
print ('Time:' , elapsed)


# In[82]:


#training data
generator_top = datagen.flow_from_directory( 
   train_data_dir, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=False) 
 
nb_train_samples = len(generator_top.filenames) 
num_classes = len(generator_top.class_indices) 
 
# load the bottleneck features saved earlier 
train_data = np.load('bottleneck_features_train1.npy') 
 
# get the class labels for the training data, in the original order 
train_labels = generator_top.classes 
 
# convert the training labels to categorical vectors 
train_labels = to_categorical(train_labels, num_classes=num_classes)


# In[83]:


#__this can take an hour and half to run so only run it once. 
#once the npy files have been created, no need to run again. Convert this cell to a code cell to run.__
#VAlidation###################################
start = datetime.datetime.now()
 
Validgenerator = datagen.flow_from_directory( 
    validation_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode=None, 
    shuffle=False) 
 
nb_valid_samples = len(Validgenerator.filenames) 
num_classes = len(Validgenerator.class_indices) 
 
predict_size_validation = int(math.ceil(nb_valid_samples / batch_size)) 
 
bottleneck_features_validation = ResNet50.predict_generator(Validgenerator, predict_size_validation) 
 
np.save('bottleneck_features_validation.npy', bottleneck_features_validation)
end= datetime.datetime.now()
elapsed= end-start
print ('Time:' , elapsed)


# In[100]:


#Validation 
valid_generator_top = datagen.flow_from_directory( 
   validation_data_dir, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=False) 
 
nb_valid_samples = len(valid_generator_top.filenames) 
num_classes = len(valid_generator_top.class_indices) 
 
# load the bottleneck features saved earlier 
valid_data = np.load('bottleneck_features_validation.npy') 
 
# get the class labels for the training data, in the original order 
valid_labels = valid_generator_top.classes 
 
# convert the training labels to categorical vectors 
valid_labels = to_categorical(valid_labels, num_classes=num_classes)


# In[106]:


#__this can take an hour and half to run so only run it once. 
#once the npy files have been created, no need to run again. Convert this cell to a code cell to run.__
#Testing###################################**********************************************************
start = datetime.datetime.now()
 
testinggenerator = datagen.flow_from_directory( 
    test_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode=None, 
    shuffle=False) 
 
nb_test_samples = len(testinggenerator.filenames) 
num_classes = len(testinggenerator.class_indices) 
 
predict_size_test = int(math.ceil(nb_test_samples / batch_size)) 
 
bottleneck_features_testing = ResNet50.predict_generator(testinggenerator, predict_size_test) 
 
np.save('bottleneck_features_testing.npy', bottleneck_features_testing)
end= datetime.datetime.now()
elapsed= end-start
print ('Time:' , elapsed)


# In[109]:


#Testing
testing_generator_top = datagen.flow_from_directory( 
   test_data_dir, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=False) 
 
nb_train_samples = len(testing_generator_top.filenames) 
num_classes = len(testing_generator_top.class_indices) 
 
# load the bottleneck features saved earlier 
test_data = np.load('bottleneck_features_testing.npy') 
 
# get the class labels for the training data, in the original order 
testing_labels = testing_generator_top.classes 
 
# convert the training labels to categorical vectors 
testing_labels = to_categorical(testing_labels, num_classes=num_classes)


# In[117]:


#CNN 

start = datetime.datetime.now()
model = Sequential() 
model.add(Flatten(input_shape=train_data.shape[1:])) 


model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.5)) 
model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.3)) 
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
   optimizer=optimizers.RMSprop(lr=1e-4),
   metrics=['acc'])
history = model.fit(train_data, train_labels, 
   epochs=200,
   batch_size=batch_size,validation_data=(valid_data, valid_labels)
   )


# In[118]:


model.save_weights(top_model_weights_path)
(eval_loss, eval_accuracy) = model.evaluate( 
    valid_data, valid_labels, batch_size=batch_size,     verbose=1)
print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100)) 
print("[INFO] Loss: {}".format(eval_loss)) 
end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)


# In[104]:


#DEBUG**********************************************************************************************
#print (valid_data.shape[1:])
#generator.class_indices
#x,y=valid_generator_top.next()
#x.shape
print(train_labels.shape)
print(train_data.shape)
print(valid_data.shape)
print(valid_labels.shape)
#print(valid_labels.reshape(44169,3))


# In[119]:


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()
plt.show()


# In[120]:


#Overall model Accuracy

model.evaluate(test_data,testing_labels)


# In[126]:


preds=np.round(model.predict(test_data),0)
print('rounded test_labels',preds)


# In[127]:


fake_news=['misleading','satire_parody', 'true']
classification_metrics=metrics.classification_report(testing_labels,preds,target_names=fake_news)
print(classification_metrics)


# In[128]:


#Since our data is in dummy format we put the numpy array into a dataframe and call idxmax axis=1 to return the column
# label of the maximum value thus creating a categorical variable
#Basically, flipping a dummy variable back to it’s categorical variable
categorical_test_labels = pd.DataFrame(testing_labels).idxmax(axis=1)
categorical_preds = pd.DataFrame(preds).idxmax(axis=1)
confusion_matrix= confusion_matrix(categorical_test_labels, categorical_preds)

#To get better visual of the confusion matrix:
def plot_confusion_matrix(cm, classes,
   normalize=False,
   title='Confusion matrix',
   cmap=plt.cm.Blues):
 
#Add Normalization Option
 #‘’’prints pretty confusion metric with normalization option ‘’’
   if normalize:
     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
     print("Normalized confusion matrix")
   else:
     print('Confusion matrix, without normalization')
 
# print(cm)
 
   plt.imshow(cm, interpolation='nearest', cmap=cmap)
   plt.title(title)
   plt.colorbar()
   tick_marks = np.arange(len(classes))
   plt.xticks(tick_marks, classes, rotation=45)
   plt.yticks(tick_marks, classes)
 
   fmt = '.2f' if normalize else 'd'
   thresh = cm.max() / 2.
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center', color="white" if cm[i, j] > thresh else "black")
 
   plt.tight_layout()
   plt.ylabel('True label')
   plt.xlabel('Predicted label') 


# In[125]:


plot_confusion_matrix(confusion_matrix,['misleading','satire_parody', 'true'],normalize=True)


# In[130]:





# In[ ]:




