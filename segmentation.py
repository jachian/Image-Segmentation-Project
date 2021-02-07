# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 17:23:53 2020

@author: HUB HUB
"""
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

from keras import layers
from keras import models

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

import keras_metrics as km

from keras_segmentation.models.model_utils import get_segmentation_model
from segmentation_models import Unet
from segmentation_models import FPN
from segmentation_models import Linknet
from segmentation_models import PSPNet

from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score

from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Cropping2D
from keras.layers import Concatenate
from keras.layers import concatenate
from keras.layers import Input
from keras.layers.core import Dense

import glob
from PIL import Image

from keras.models import model_from_json
import os

#import segmentation models
# from keras_segmentation.models.unet import vgg_unet, fcn_8, fcn_32, fcn_8_vgg, fcn_32_vgg, fcn_8_resnet50,
# fcn_32_resnet50, fcn_8_mobilenet, fcn_32_mobilenet, pspnet, vgg_pspnet, resnet50_pspnet, unet_mini, unet,
# resnet50_unet, mobilenet_unet, segnet, vgg_segnet, resnet50_segnet, mobilenet_segnet

import numpy as np


train_images = 'D:/Data Science/segmentation project/AppleA_Train'
train_segmentation = 'D:/Data Science/segmentation project/AppleALabels_Train'

apple_a = 'D:/Data Science/segmentation project/AppleA'
apple_a_labels = 'D:/Data Science/segmentation project/AppleA_Labels_1'

apple_b = 'D:/Data Science/segmentation project/AppleB_1'
apple_b_labels = 'D:/Data Science/segmentation project/AppleB_Labels_1'

peach = 'D:/Data Science/segmentation project/Peach_1'
peach_labels = 'D:/Data Science/segmentation project/PeachLabels_1'

pear = 'D:/Data Science/segmentation project/Pear_1'
pear_labels = 'D:/Data Science/segmentation project/PearLabels_2'



#define data generators   #######################################
train_datagen = ImageDataGenerator()
train_seg_datagen = ImageDataGenerator()

apple_datagen = ImageDataGenerator()
apple_seg  = ImageDataGenerator()

peach_datagen = ImageDataGenerator()
peach_seg  = ImageDataGenerator()

pear_datagen = ImageDataGenerator()
pear_seg  = ImageDataGenerator()

################################
#train_datagen.fit(train_images, seed = 1)
#train_seg_datagen.fit(train_segmentation, seed = 1)

#val_datagen.fit(val_images, seed = 1)
#val_segmentation.fit(val_segmentation, seed = 1)


# Assign images to generators #######################################
train_generator = train_datagen.flow_from_directory(train_images,
                                                    #target_size = (320,320),
                                                    color_mode= 'rgb',
                                                    #batch_size=4,
                                                    class_mode= 'categorical')

train_senerator = train_seg_datagen.flow_from_directory(train_segmentation,
                                                            #target_size = (320,320),
                                                            #batch_size=4,
                                                            color_mode= 'rgb',
                                                            class_mode= 'categorical')


apple_generator = apple_datagen.flow_from_directory(apple_a, 
                                                #batch_size=4)
                                                #target_size = (320,320),
                                                color_mode= 'rgb',
                                                class_mode= 'categorical')

apple_senerator = apple_seg.flow_from_directory(apple_a_labels,
                                                        #batch_size=4)
                                                        #target_size = (320,320),
                                                        color_mode= 'rgb',
                                                        class_mode= 'categorical')


peach_generator = peach_datagen.flow_from_directory(peach, 
                                                #batch_size=4)
                                                #target_size = (320,320),
                                                color_mode= 'rgb',
                                                class_mode= 'categorical')

peach_senerator = peach_seg.flow_from_directory(peach_labels,
                                                        #batch_size=4)
                                                        #target_size = (320,320),
                                                        color_mode= 'rgb',
                                                        class_mode= 'categorical')


pear_generator = pear_datagen.flow_from_directory(pear, 
                                                #batch_size=4)
                                                #target_size = (320,320),
                                                color_mode= 'rgb',
                                                class_mode= 'categorical')

pear_senerator = pear_seg.flow_from_directory(pear_labels,
                                                        #batch_size=4)
                                                        #target_size = (320,320),
                                                        color_mode= 'rgb',
                                                        class_mode= 'categorical')


def custom_model():
    
    #COCO width == 640 height == 483 #--- may be a problem images are of variable width and height.
    # input_shape = (input_height, input_width, 3)
    
    # defining encoder layers
    
    conv1 = layers.Conv2D(32, (3,3), activation = 'relu', padding = 'same', input_shape = (None, None, 3))
    pool1 = layers.MaxPooling2D((2,2))
    
    conv2 = layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same')   #intermediate output of encoder
    pool2 = layers.MaxPooling2D((2,2))      #final output of encoder
    
    
    #defining the decoder layers  #################################################
    conv3 = layers.Conv2D(128, (3,3), activation = 'relu', padding = 'same')
    
    #up1 = layers.UpSampling2D((2,2))
    conv4 = layers.Conv2D(64, (3,3), activation = 'relu', padding = 'same')
    
    # up2 = layers.UpSampling2D((2,2))
    conv5 = layers.Conv2D(32, (3,3), activation = 'relu', padding = 'same')
    
    out = layers.Conv2D(182, (1,1), padding = 'same')
    

    
    # now we build the model by combining layers ###################################
    my_model = models.Sequential()
    
    my_model.add(conv1)
    my_model.add(layers.Dropout(0.2))
    my_model.add(pool1)
    
    my_model.add(conv2)
    my_model.add(layers.Dropout(0.2))
    my_model.add(pool2)
    
    my_model.add(conv3)
    my_model.add(layers.Dropout(0.2))
    my_model.add(layers.UpSampling2D((2,2)))
    # my_model.add(np.concatenate([layers.UpSampling2D(2,2), conv2], axis = -1))
    
    my_model.add(conv4)
    my_model.add(layers.Dropout(0.2))
    my_model.add(layers.UpSampling2D((2,2)))
    # my_model.add(np.concatenate([layers.UpSampling2D(2,2), conv1], axis = -1))
    
    my_model.add(conv5)
    my_model.add(layers.Dropout(0.2))
    
    my_model.add(out)
    
    #compile the model
    my_model.compile(loss = 'binary_crossentropy',
                     optimizer = optimizers.RMSprop(learning_rate =1e-4),
                     metrics = ['accuracy'])
    
    return my_model



def custom_model1():
    
    #COCO width == 640 height == 483 #--- may be a problem images are of variable width and height.
    # input_shape = (input_height, input_width, 3)
    
    # defining encoder layers
    # shape = [None, 2],
    
    input_1 = Input(shape = (3, 300, 300))
    # input_1 = tf.placeholder(shape = (None, None, 3), dtype = tf.float32)
    
    conv1 = Conv2D(32, (3,3), activation = 'relu', padding = 'same', data_format="channels_first")(input_1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling2D((2,2), dim_ordering="th")(conv1)
    
    conv2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(pool1) #intermediate output of encoder
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling2D((2,2), dim_ordering="th")(conv2)      #final output of encoder
    
    
    #defining the decoder layers  #################################################
    conv3 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(conv3)
    
    # up1 = concatenate([UpSampling2D((2,2))(conv3), conv2], axis = -1)
    
    #up_conv3 = UpSampling2D(size=(2, 2), data_format="channels_last")(conv3)
    #ch, cw = get_crop_shape(conv2, up_conv3)
    #crop_conv2 = Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv2)
    #up1   = concatenate([up_conv3, crop_conv2], axis = -1)
    
    up1 = UpSampling2D((2,2))(conv3)
    
    conv4 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(conv4)
    
    up2 = UpSampling2D((2,2))(conv4)
    
    conv5 = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(conv5)
    
    out = Conv2D(3, (1,1), padding = 'same')(conv5)
    
    model = get_segmentation_model(input_1, out)   # this will build the model.
    
    return model







#def train_models():
    #custom = custom_model()   
    
    #big_train_gen = zip(train_generator, train_seg_generator)
    #big_val_gen = zip(val_generator, val_seg_generator)
    
    # big_train_gen = ((pair for pair in zip(train_generator, train_seg_generator)))
    # big_val_gen = ((pair for pair in zip(val_generator, val_seg_generator)))
    
    #history = custom.fit_generator(big_train_gen, 
                                   #epochs=10, steps_per_epoch = 1,
                                   #validation_data = big_val_gen, #use_multiprocessing=True,
                                   #validation_steps = 1)
    
    #history = custom.fit((train_generator, train_seg_generator),
                         #epochs = 10,
                         #validation_data = (val_generator, val_seg_generator))
    
    #return history


#Try converting them to numpy arrays first.


#def train_models1():
    
    #from keras_segmentation.models.unet import vgg_unet

    #model = vgg_unet(n_classes=1 ,  input_height=300, input_width=300)
    
    #history = model.train(train_images = 'D:/Data Science/segmentation project/train_images/train_images',
                          #train_annotations = 'D:/Data Science/segmentation project/train_segmentation/train_segmentation',
                         # checkpoints_path = 'D:/Data Science/segmentation project/models',
                          #epochs = 5)
    
    #return history

#def unet_model():
    
    #print("training unet >>>> ")
    
    #model = Unet('resnet34', input_shape=(None, None, 3), encoder_weights = 'imagenet')
    #model.compile('Adam', loss='categorical_crossentropy',metrics=[km.binary_precision(), km.binary_recall()]) # metrics=['acc'])
     
    #return model



def unet_model():
    
    print("training unet >>>> ")
    
    model = Unet('resnet34', encoder_weights = 'imagenet')
    model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
     
    return model


def fpn_model():
    
    print("training unet >>>> ")
    
    model = FPN('resnet34', encoder_weights = 'imagenet')
    model.compile('Adam', loss=bce_jaccard_loss, metrics=['acc']) # metrics=[iou_score]) # metrics=['acc'])
     
    return model


def linknet_model():
    
    print("training unet >>>> ")
    
    model = Linknet('resnet34', encoder_weights = 'imagenet')
    model.compile('Adam', loss=bce_jaccard_loss, metrics=['acc']) # metrics=[iou_score]) #metrics=['acc'])
     
    return model


def psp_model():
    
    print("training unet >>>> ")
    
    model = PSPNet('resnet34', input_shape=(None, None, 3), encoder_weights = 'imagenet')
    model.compile('Adam', loss=bce_jaccard_loss, metrics=['acc']) # metrics=[iou_score]) #metrics=['acc'])
     
    return model


def train_all():
    
    unet = unet_model()
    unet_trained = train_model(unet, train_generator, train_senerator, apple_generator, apple_senerator)
    save_model(unet_trained, 'unet_t')
    
    fpn = fpn_model()
    fpn_trained = train_model(fpn, train_generator, train_senerator, apple_generator, apple_senerator)
    save_model(fpn_trained, 'fpn_t')
    
    linknet = linknet_model()
    linknet_trained = train_model(linknet, train_generator, train_senerator, apple_generator, apple_senerator)
    save_model(linknet_trained, 'linknet_t')
    
    psp = psp_model()
    psp_trained = train_model(psp, train_generator, train_senerator, apple_generator, apple_senerator)
    save_model(psp_trained, 'psp_t')
    
    #plot some results
    plot_training(unet_trained, 'Unet')
    plot_training(fpn_trained, 'FPN')
    plot_training(linknet_trained, 'linkNet')
    plot_training(psp_trained, 'PSP')
    
    return [unet_trained, unet_trained, linknet_trained, psp_trained]

    
    
    

def plot_training(model, name):
    
    import matplotlib.pyplot as plt
    
    # summarize history for accuracy
    plt.plot(model.history['iou_score'])
    plt.plot(model.history['val_iou_score'])
    plt.title(name + ' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()
    
    # summarize history for loss
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title(name + ' loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()


    
    
    
    
    
    
    
# shape (2, 4, 256, 256, 3)

#def train_models2(model):
    
    #from segmentation_models import get_preprocessing
    
    # BACKBONE = 'resnet34'
    # preprocess_input = get_preprocessing('resnet34')
    
    
    #train_gen = pair_generator(train_generator, train_seg_generator)
    #val_gen = pair_generator(val_generator, val_seg_generator)
    

    
    # fit model
    #history = model.fit_generator(train_gen,        # y = train_seg_generator,  #batch_size=16,
                        #epochs=10,
                        #steps_per_epoch = 2,
                        #validation_data= val_gen,
                        #validation_steps = val_generator.samples)
    
    # history = model.train(train_images = 'D:/Data Science/segmentation project/train_images/train_images',
                          # train_annotations = 'D:/Data Science/segmentation project/train_segmentation/train_segmentation',
                          # checkpoints_path = 'D:/Data Science/segmentation project/models',
                          # epochs = 5)
    
    #train_im = load_images('D:/Data Science/segmentation project/train_images/train_images/*')
   # train_mask_im = load_images('D:/Data Science/segmentation project/train_segmentation/train_segmentation/*')
    
    #val_im = load_images('D:/Data Science/segmentation project/val_images/val_images/*')
    #val_mask_im = load_images('D:/Data Science/segmentation project/val_segmentation/val_segmentation/*')    
    
    # fit model
    #history = model.fit(x = train_im,
              #y = train_mask_im,
              #batch_size=16,
              #epochs=10,
              #validation_data=(val_im, val_mask_im))
    
    #return history

# ValueError: Error when checking input: expected data to have 4 dimensions, but got array with shape (4, 1)
    
def train_model(model, train_x, train_y, val_x, val_y):
    
    train_gen = pair_generator(train_x, train_y)
    val_gen = pair_generator(val_x, val_y)    

    #train_gen = pair_generator(train_generator, train_seg_generator)
    #val_gen = pair_generator(val_generator, val_seg_generator)   
    
    # fit model
    history = model.fit_generator(train_gen,        # y = train_seg_generator,  #batch_size=16,
                                  epochs=5,
                                  steps_per_epoch = 2,
                                  validation_data= val_gen,
                                  validation_steps = train_x.samples, verbose = 1)    
    
    return history


def pair_generator(image_gen, mask_gen):
    big_train_gen = (pair for pair in zip(image_gen, mask_gen))
    for (img, mask) in big_train_gen:
        # print("Type of Image == " +str(type(img)))
        yield (img[0], mask[0])
        

#def pair_generator(image_gen, mask_gen):
    #big_train_gen = zip(image_gen, mask_gen)
    #for (img, mask) in big_train_gen:
        #yield np.array(img), np.array(mask)


#def train_models4(model):
    
    #train_gen = pair_generator1(train_generator, train_seg_generator)
    #val_gen = pair_generator1(val_generator, val_seg_generator)    

    #train_gen = pair_generator(train_generator, train_seg_generator)
    #val_gen = pair_generator(val_generator, val_seg_generator)   
    
    # fit model
    #history = model.fit_generator(batch_generator(train_generator, train_seg_generator),        # y = train_seg_generator,  #batch_size=16,
                                  #epochs=10,
                                  #steps_per_epoch = 1,
                                  #validation_data = batch_generator(val_generator, val_seg_generator),
                                  #validation_steps = val_generator.samples)    
    
    #return history
        

#def batch_generator(X_gen,Y_gen):
    #while True:
        #yield (pair for pair in (X_gen.next(), Y_gen.next()))
    

        

#def preprocess_input(generator):
    
    #image_list = []
    
    #for img in generator:
        #print("type of image == " +str(type(img)))
        
        #image_list.append(img)
    
    #print("converting to numpy")
    
    #images = np.array(image_list)
    
    #return images


#def load_images(folder_path):
    
    #images = []
    #for f in glob.iglob(folder_path):
        #images.append(np.asarray(Image.open(f)))
        #print("processing image.....")
        
    #images = np.array(images)
    
    #return images

    
    
    

#write your own preprocess function and use the fit function.

# big_train_gen = zip(train_generator, train_seg_generator)
# big_val_gen = zip(val_generator, val_seg_generator)
# (1, 4, 256, 256, 3)

    
#train_images = 'D:/Data Science/segmentation project/train_images'
#train_segmentation = 'D:/Data Science/segmentation project/train_segmentation'

#val_images = 'D:/Data Science/segmentation project/val_images'
#val_segmentation = 'D:/Data Science/segmentation project/val_segmentation'

# ValueError: could not broadcast input array from 
# shape (4,300,300,3) into shape (4)
    
def save_model(model, name):
    model.save(name + '.h5')  # creates a HDF5 file 'my_model.h5'
    
    print("...Model saved....")


def load_model(name):
    from keras.models import load_model
    # returns a compiled model
    # identical to the previous one
    model = load_model(name)
    
    return model

    
    
    
#def save_model(model, name):
    # serialize model to JSON
    #model_json = model.to_json() 
                 #with open(name + ".json", "w") as json_file:
                 #json_file.write(model_json)
    # serialize weights to HDF5
    #model.save_weights("model.h5")
    #print("Saved model to disk")


##json_file = open(name, 'r')
    #loaded_model_json = json_file.read()
    #json_file.close()
    ##loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    #loaded_model.load_weights("model.h5")
    #print("Loaded model from disk")
    
    #return loaded_model




    
    
    