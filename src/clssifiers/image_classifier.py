import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image


def train_images():
    train = ImageDataGenerator(rescale=1 / 150)
    valid = ImageDataGenerator(rescale=1 / 150)

    train_dataset = train.flow_from_directory(r'C:\\Users\\ccastano\\Desktop\\basedata\\training',
                                              target_size=(200, 200),
                                              batch_size=3, class_mode='categorical')
    validation_train_dataset = train.flow_from_directory(r'C:\\Users\\ccastano\\Desktop\\basedata\\validation',
                                                         target_size=(200, 200), batch_size=3, class_mode='categorical')
    train_dataset.class_indices()

    model = tf.keras.models.Sequential(
        [tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
         tf.keras.layers.MaxPool2D(2, 2),
         tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
         tf.keras.layers.MaxPool2D(2, 2),

         tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(200, 200, 3)),
         tf.keras.layers.MaxPool2D(2, 2),

         tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(200, 200, 3)),
         tf.keras.layers.MaxPool2D(2, 2),
         tf.keras.layers.Conv2D(256, (3, 3), activation='relu', input_shape=(200, 200, 3)),
         tf.keras.layers.MaxPool2D(2, 2),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(700, activation='relu'),

         tf.keras.layers.Dense(3, activation='sigmoid'),

         ])

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
                  metrics=['accuracy'])

    model_fit = model.fit(train_dataset, steps_per_epoch=5,
                          epochs=100,
                          validation_data=validation_train_dataset)
    return "Done"


if __name__ == '__main__':
    train_images()
