#!/usr/bin/env python
# coding: utf-8

# In[19]:


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


# In[21]:


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


# In[22]:


#Loading ResNet50 model
ResNet50 = ResNet50(include_top=False, weights='imagenet',input_shape=(224,224,3))
#datagen = ImageDataGenerator(rescale=1. / 255) 

datagen=ImageDataGenerator(rescale=1./255,
                          )
#needed to create the bottleneck .npy files


# In[23]:


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
 
np.save('bottleneck_features_train3.npy', bottleneck_features_train)
end= datetime.datetime.now()
elapsed= end-start
print ('Time:' , elapsed)


# In[24]:


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
train_data = np.load('bottleneck_features_train3.npy') 
 
# get the class labels for the training data, in the original order 
train_labels = generator_top.classes 
 
# convert the training labels to categorical vectors 
train_labels = to_categorical(train_labels, num_classes=num_classes)


# In[25]:


datagen2=ImageDataGenerator(rescale=1./255)
                          


# In[26]:


#__this can take an hour and half to run so only run it once. 
#once the npy files have been created, no need to run again. Convert this cell to a code cell to run.__
#VAlidation###################################
start = datetime.datetime.now()
 
Validgenerator = datagen2.flow_from_directory( 
    validation_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode=None, 
    shuffle=False) 
 
nb_valid_samples = len(Validgenerator.filenames) 
num_classes = len(Validgenerator.class_indices) 
 
predict_size_validation = int(math.ceil(nb_valid_samples / batch_size)) 
 
bottleneck_features_validation = ResNet50.predict_generator(Validgenerator, predict_size_validation) 
 
np.save('bottleneck_features_validation3.npy', bottleneck_features_validation)
end= datetime.datetime.now()
elapsed= end-start
print ('Time:' , elapsed)#RUNNNNNNNNNNNNNNNNNN


# In[27]:


#Validation 
valid_generator_top = datagen2.flow_from_directory( 
   validation_data_dir, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=False) 
 
nb_valid_samples = len(valid_generator_top.filenames) 
num_classes = len(valid_generator_top.class_indices) 
 
# load the bottleneck features saved earlier 
valid_data = np.load('bottleneck_features_validation3.npy') 
 
# get the class labels for the training data, in the original order 
valid_labels = valid_generator_top.classes 
 
# convert the training labels to categorical vectors 
valid_labels = to_categorical(valid_labels, num_classes=num_classes)


# In[28]:


#__this can take an hour and half to run so only run it once. 
#once the npy files have been created, no need to run again. Convert this cell to a code cell to run.__
#Testing###################################**********************************************************
start = datetime.datetime.now()
 
testinggenerator = datagen2.flow_from_directory( 
    test_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode=None, 
    shuffle=False) 
 
nb_test_samples = len(testinggenerator.filenames) 
num_classes = len(testinggenerator.class_indices) 
 
predict_size_test = int(math.ceil(nb_test_samples / batch_size)) 
 
bottleneck_features_testing = ResNet50.predict_generator(testinggenerator, predict_size_test) 
 
np.save('bottleneck_features_testing3.npy', bottleneck_features_testing)
end= datetime.datetime.now()
elapsed= end-start
print ('Time:' , elapsed)


# In[29]:


#Testing
testing_generator_top = datagen2.flow_from_directory( 
   test_data_dir, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=False) 
 
nb_train_samples = len(testing_generator_top.filenames) 
num_classes = len(testing_generator_top.class_indices) 
 
# load the bottleneck features saved earlier 
test_data = np.load('bottleneck_features_testing3.npy') 
 
# get the class labels for the training data, in the original order 
testing_labels = testing_generator_top.classes 
 
# convert the training labels to categorical vectors 
testing_labels = to_categorical(testing_labels, num_classes=num_classes)


# In[30]:


from tensorflow.keras import layers
from tensorflow.keras import activations
#data_augmentation =Sequential([model.add(layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical")])
                               
#model = Sequential()
#model.add(layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"))


# In[31]:


start = datetime.datetime.now()
model = Sequential() 
model.add(Flatten(input_shape=train_data.shape[1:])) 


model.add(Dense(100))
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(Dropout(0.5)) 
model.add(Dense(50))
model.add(keras.layers.LeakyReLU(alpha=0.3))
model.add(Dropout(0.3)) 
model.add(Dense(num_classes, activation='softmax'))


# In[32]:


model.summary()


# In[33]:


from keras.callbacks import ModelCheckpoint


# In[34]:


checkpoint=ModelCheckpoint(r'C:\Users\ccastano\Desktop\checkpoint.h5',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')


# In[35]:


model.compile(loss='categorical_crossentropy',
   optimizer=optimizers.RMSprop(lr=1e-4),
   metrics=['acc'])
history = model.fit(train_data, train_labels, 
   epochs=100,
   batch_size=batch_size,validation_data=(valid_data, valid_labels),callbacks=[checkpoint],verbose=1
   )


# In[36]:


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


# In[37]:


from keras.preprocessing.image import ImageDataGenerator
datagen1=ImageDataGenerator(featurewise_center=False,
                            samplewise_center=False,
                            samplewise_std_normalization=False,
                            zca_whitening=False,
                            rotation_range=45,
                            width_shift_range=0.2,
                        
                            height_shift_range=0.2,
                            horizontal_flip=True,
                        
                           )
#datagen1.fit(valid_generator_top)
augmented_model=model


# In[42]:


from keras.callbacks import ModelCheckpoint
augmented_checkpoint=ModelCheckpoint(r'C:\Users\ccastano\Desktop\augmented_best1.h5',
                                                    monitor='val_loss',
                                                    verbose=0,
                                                    save_best_only=True,
                                                    mode='auto')


# In[43]:


augmented_model.compile(loss='categorical_crossentropy',
   optimizer=optimizers.RMSprop(lr=1e-4),
   metrics=['acc'])


# In[ ]:


augemented_model_details=augmented_model.fit_generator(datagen1.flow(train_data, train_labels,
   batch_size=32),steps_per_epoch=len(train_data)/32,
                                                      epochs=50,validation_data=(valid_data, valid_labels),callbacks=[augmented_checkpoint],verbose=1)


# In[42]:


model.evaluate(test_data,testing_labels)


# In[43]:


preds=np.round(model.predict(test_data),0)
print('rounded test_labels',preds)


# In[44]:


fake_news=['misleading','satire_parody', 'true']
classification_metrics=metrics.classification_report(testing_labels,preds,target_names=fake_news)
print(classification_metrics)


# In[45]:


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


# In[46]:


plot_confusion_matrix(confusion_matrix,['misleading','satire_parody', 'true'],normalize=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[47]:


import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import pandas as pd
import glob,os
from os import listdir
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras import models


# In[48]:


folderPath=r'C:/Users/ccastano/Desktop/TSV_Files/multimodal_train_clean_sort/'
filepaths = [f for f in listdir(folderPath) if f.endswith('.tsv')]


filepaths=[ folderPath+s for s in filepaths]
dfs = [] 
for f in filepaths:
    df = pd.read_csv(f, sep='\t')
    dfs.append(df)
df_train = pd.concat(dfs, ignore_index=True)
print(df_train.shape)


# In[49]:


folderPath=r'C:/Users/ccastano/Desktop/TSV_Files/multimodal_validate_clean_sort/'
filepaths = [f for f in listdir(folderPath) if f.endswith('.tsv')]


filepaths=[ folderPath+s for s in filepaths]
dfs = [] 
for f in filepaths:
    df = pd.read_csv(f, sep='\t')
    dfs.append(df)
df_valid = pd.concat(dfs, ignore_index=True)
print(df_valid.shape)


# In[50]:


train=ImageDataGenerator(rescale=1./255)
valid=ImageDataGenerator(rescale=1./255)


# In[51]:


train_dataset=train.flow_from_directory(r'C:/Users/ccastano/Desktop/basedata/training',target_size=(224,224),batch_size=50,class_mode='categorical')


# In[52]:


train_text=[]
train_labels2=[]
for f in train_dataset.filenames:
    
    f=(f[f.index('\\')+1:f.index('.')])
    train_text.append(df_train[df_train.id==f].clean_title.tolist())
    train_labels2.append(df_train[df_train.id==f]["3_way_label"].tolist())
    
    
    
    
len(train_text)
len(train_labels)


# In[53]:


train_labels2=[]
for f in train_dataset.filenames:
    f=(f[f.index('\\')+1:f.index('.')])
    train_labels2.append((df_train[df_train.id==f]["3_way_label"].tolist()))


# In[54]:


validation_train_dataset=train.flow_from_directory(r'C:/Users/ccastano/Desktop/basedata/validation',target_size=(224,224),batch_size=50,class_mode='categorical')


# In[55]:


validate_text=[]
validate_labels2=[]
for f in validation_train_dataset.filenames:
    ff=f
    f=(f[f.index('\\')+1:f.index('.')])
     
     
        
    validate_text.append(df_valid[df_valid.id==f].clean_title.tolist())
    validate_labels2.append(df_valid[df_valid.id==f]["3_way_label"].tolist())
    
    
    
    
    
len(validate_text)
len(validate_labels2)


# In[56]:


#tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(train_text+validate_text)
sequences = tokenizer.texts_to_sequences(train_text)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

seq_train = pad_sequences(sequences, maxlen=1000)
sequences = tokenizer.texts_to_sequences(validate_text)
seq_valid=pad_sequences(sequences, maxlen=1000)
 
#print(pad_sequences(sequences, maxlen=10000))
#print(sequences)
#print(seq_train)
#print(seq_valid)
print(train_text)


# In[57]:


embeddings_index={}
f=open(os.path.join(r'C:/Users/ccastano/Desktop/glove.6B.100d.txt'), encoding='utf-8')
for line in f: 
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:],dtype='float32')
    embeddings_index[word]=coefs
f.close()


print('Found %s word vectors.' % len(embeddings_index))


num_words = min(100000, len(word_index))+1
print(num_words)
embedding_matrix = np.zeros((num_words, 100))
print(embedding_matrix.shape)
for word, i in word_index.items():
    if i >= 100000:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[58]:


train_labelss = to_categorical(np.asarray(train_labels2))
valid_labelss = to_categorical(np.asarray(validate_labels2))
print('Shape of label tensor:', valid_labelss.shape)
print('Shape of label tensor:', train_labelss.shape)


# In[59]:


train_labelss


# In[60]:


embedding_layer = Embedding(100000,
                            100,
                            weights=[embedding_matrix],
                            input_length=1000,
                            trainable=False)

print('Training model.')


# In[61]:


import tensorflow as tf
import keras.backend.tensorflow_backend as tfback
from tensorflow.keras.models import load_model

print("tf.__version__ is", tf.__version__)
print("tf.keras.__version__ is:", tf.keras.__version__)

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
tfback._get_available_gpus = _get_available_gpus 


# In[62]:


embedding_layer = Embedding(48497,
                            100,
                            weights=[embedding_matrix],
                            input_length=1000,
                            trainable=False)

print('Training model.')



# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(1000,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu',data_format = 'channels_first')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu',data_format = 'channels_first')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)

#preds = Dense(len(train_labelss)+len(valid_labelss), activation='softmax')(x)
preds = Dense(3, activation='softmax')(x)


model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


 

model.fit(seq_train, train_labelss,
          batch_size=128,
          epochs=10,
          validation_data=(seq_valid, valid_labelss))


# In[71]:


import tensorflow as tf
import numpy as np
#training_padded = np.array(train_dataset)
#training_labels = np.array(train_labelss)
#testing_padded = np.array(validation_train_dataset)
#testing_labels = np.array(valid_labelss)

# Image
input_1 = tf.keras.layers.Input(shape=(train_data.shape[1:]))
conv2d_1 = tf.keras.layers.Conv2D(64, kernel_size=3,
                                  activation=tf.keras.activations.relu)(input_1)

# Second conv layer :
conv2d_2 = tf.keras.layers.Conv2D(32, kernel_size=3,
                                  activation=tf.keras.activations.relu)(conv2d_1)

# Flatten layer :
flatten = tf.keras.layers.Flatten()(conv2d_2)

# The other input
input_2 = tf.keras.layers.Input(shape=(1000,))
dense_2 = tf.keras.layers.Dense(5, activation=tf.keras.activations.relu)(input_2)

# Concatenate
concat = tf.keras.layers.Concatenate()([flatten, dense_2])

n_classes = 3
# output layer
output = tf.keras.layers.Dense(units=n_classes,
                               activation=tf.keras.activations.softmax)(concat)

full_model = tf.keras.Model(inputs=[input_1, input_2], outputs=[output])

print(full_model.summary())
full_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['acc'])
#history = model.fit(train_data, train_labels, 
  # epochs=100,
   #batch_size=batch_size,validation_data=(valid_data, valid_labels),callbacks=[checkpoint],verbose=1
   #


# To train
history = full_model.fit([train_data,seq_train], train_labels, epochs=50, validation_data=([valid_data,seq_valid],valid_labels ) )


# In[ ]:




