import tensorflow as tf
import keras
import time
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAveragePooling2D,Flatten
from tensorflow.keras import  Model
import csv
import os
import glob
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
import  shutil
import matplotlib.pyplot as plt
import  numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from  utility import *
from   deeplearingmodel import  cloth



if __name__ == "__main__":
    test_data_path = 'C:\\Users\\Daniel Samuel\\Desktop\\cloth_recommendation\\test'
    train_data_path = 'C:\\Users\\Daniel Samuel\\Desktop\\cloth_recommendation\\train'
    val_data_path = 'C:\\Users\\Daniel Samuel\\Desktop\\cloth_recommendation\\val'
    batch_size = 64
    train_generator, val_generator, test_generator = create_generators(batch_size ,train_data_path,val_data_path,test_data_path)
    nbr_classes = train_generator.num_classes
    Train = True
    if Train:
        path_to_save_model = "./model"
        chk_saver = ModelCheckpoint(path_to_save_model,
                                     monitor  = 'val_accuracy',
                                     mode = 'max',
                                     save_best_only = True,
                                    save_freq = 'epoch',
                                      verbose = 1  )

        model = cloth(nbr_classes)
        optimizer = tf.keras.optimizers.Adam(
                                 learning_rate=0.01)
        model.compile(optimizer ='adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
        model.fit(train_generator,
                  epochs = 30,
                  batch_size=batch_size,
                  validation_data = val_generator,
                  callbacks= [chk_saver] )

        # ts = int(time.time())
        # file_path = f"tf-models/img_classifier/{ts}/"
        # model.save(filepath=file_path, save_format='tf')

    Test = True
    if Test:
        model =tf.keras.models.load_model('./model')
        model.summary()
        print("evaluating Validation set:")
        model.evaluate(val_generator)
        print("evaluating test set : ")
        model.evaluate(test_generator)








