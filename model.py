import argparse

import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, MaxPooling3D, ELU
from keras.models import Sequential, model_from_json
from sklearn.model_selection import train_test_split
import json



# This method is for displaying the image from a file for debugging purpose
def getImage(file1, debug=False, plot=False):
    if (debug):
        print(file1)
    img = cv2.imread(file1)
    correctedimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if (plot):
        plotImage(correctedimg, debug)

    return correctedimg


# This method is for displaying the image from a file for debugging purpose
def plotImage(image, debug=False, title=""):
    if (debug):
        print(image.shape)
        print(image)
    plt.imshow(image)
    plt.figtext(0, 0, title)
    plt.show()


# Simple normalization , dividing the data by 255 and setting the mean to 0
def normalizeImage(image):
    image = image / 255.
    image -= 0.5
    return image


def getnormalizeddata(filelocations,yoriginaldata, debug=False):
    xnormalized = []
    for filename in filelocations:
        image = getImage(filename)
        resize = resizeimage(image, 4)
        normalizedimage = normalizeImage(resize)
        xnormalized.append(normalizedimage)
        if debug and len(xnormalized) % 100 == 0:
            print("Loaded " + str(len(xnormalized)) + "images")

    ydata = yoriginaldata
    return xnormalized, ydata


def resizeimage(image, factor):
    return cv2.resize(image, (int(image.shape[0] / factor), int(image.shape[1] / factor)))


def flipimage(image):
    flippedimage = cv2.flip(image, 1)
    return flippedimage


def augmentdata(xnormalizeddata, ydata):
    datalen = len(ydata)
    i = 0
    while i < datalen:
        currentx = xnormalizeddata[i]
        flippedimage = flipimage(currentx)
        xnormalizeddata.append(flippedimage)
        ydata = ydata.append(pd.Series(ydata[i] * -1))
        i = i + 1
    return xnormalizeddata, ydata


def getFinalData(imagefiles,yoriginaldata, augment=True):
    xnormalized, ydata = getnormalizeddata(imagefiles,yoriginaldata, debug=True)
    if (augment):
        xfinal, yfinal = augmentdata(xnormalized, ydata)
    else:
        xfinal = xnormalized
        yfinal = ydata
    return np.asarray(xfinal), np.asarray(yfinal)


def generate_data(xdata,ydata,batch_size=128,pr_threshold=0.5):
    while 1:
        batch_x_data = []
        batch_y_data = []
        for i in range(batch_size):
            rand_i = np.random.randint(len(xdata))
            keep_pr = 0
            while keep_pr==0:
                rand_i = np.random.randint(len(xdata))
                if abs(ydata[rand_i]) < .1:
                    pr_val = np.random.uniform()
                    if pr_val>pr_threshold:
                        keep_pr = 1
                else:
                    keep_pr = 1

            batch_x_data.append(xdata[rand_i])
            batch_y_data.append(ydata[rand_i])
        yield np.asarray(batch_x_data),np.asarray(batch_y_data)



def trainmodel(X_final,Y_final, batch_size=128, nb_epoch=20, kernel_size=(5, 5), nb_filters=32,model=None):
    checkpoint = ModelCheckpoint("model.h5", monitor='loss', verbose=1, save_best_only=False, mode='max')
    callbacks_list = [checkpoint]
    if model is None:
        model = Sequential()
        model.add(Convolution2D(24, kernel_size[0], kernel_size[1],
                                border_mode='same',subsample=(2,2),init="he_normal",input_shape=(80, 40, 3,)))
        model.add(ELU())

        model.add(Convolution2D(36, kernel_size[0], kernel_size[1],
                                border_mode='same',subsample=(1,1),init="he_normal"))
        model.add(ELU())
        model.add(Convolution2D(48, kernel_size[0], kernel_size[1],
                                border_mode='same',subsample=(1,1),init="he_normal"))
        model.add(ELU())

        model.add(Convolution2D(64, 3, 3,
                                border_mode='same',subsample=(1,1),init="he_normal"))
        model.add(ELU())

        model.add(Convolution2D(64, 3, 3,
                                border_mode='same',subsample=(1,1),init="he_normal"))
        model.add(ELU())
        model.add(Flatten())
        model.add(ELU())
        model.add(Dense(1164,init="he_normal"))
        model.add(ELU())
        model.add(Dense(100,init="he_normal"))
        model.add(ELU())
        model.add(Dense(50,init="he_normal"))
        model.add(ELU())
        model.add(Dense(10,init="he_normal"))
        model.add(ELU())
        model.add(Dense(1,init="he_normal"))

        model.compile(loss='mean_squared_error', optimizer='adam')
    else:
        print("Loaded model from file")

    print(model.summary())
    history = model.fit_generator(generate_data(X_final,Y_final,pr_threshold=0.5),batch_size,nb_epoch*2,callbacks=callbacks_list)
    history = model.fit_generator(generate_data(X_final,Y_final,pr_threshold=0.8),batch_size,nb_epoch,callbacks=callbacks_list)
    history = model.fit_generator(generate_data(X_final,Y_final,pr_threshold=0.0),batch_size,nb_epoch,callbacks=callbacks_list)
    history = model.fit_generator(generate_data(X_final,Y_final,pr_threshold=0.1),batch_size,nb_epoch,callbacks=callbacks_list)
    history = model.fit_generator(generate_data(X_final,Y_final,pr_threshold=0.2),batch_size,nb_epoch,callbacks=callbacks_list)
    return model



def save_model(model, filename="model.h5",model_arch="model.json"):
    model.save(filepath=filename)
    model_json = model.to_json()
    with open(model_arch, 'w') as f:
        json.dump(model_json, f)
    print("Saved model to filename="+filename + ", and model_arch="+model_arch)




if __name__ == '__main__':
    dataframe = pd.read_csv("udacity_data/data/driving_log.csv", delim_whitespace=False, header=None)
    centerimages = (dataframe[:][0])
    yoriginal = (dataframe[:][3])
    xfinal, yfinal = getFinalData(centerimages,yoriginal)

    # xgen,ygen = next(generate_data(xfinal,yfinal))
    # print(ygen)

    # X_train, X_valid, y_train, y_valid = train_test_split(
    #     xfinal,
    #     yfinal,
    #     test_size=0.05,
    #     random_state=832289)

    model = None
    if os.path.isfile("model.json"):
        with open('model.json', 'r') as jfile:
            if jfile is not None:
                model = model_from_json(json.load(jfile))
                model.compile("adam", "mse")
                weights_file = "model.h5"
                model.load_weights(weights_file)
                print("Loading model from file")

    model = trainmodel(xfinal, yfinal, batch_size=1280*10,nb_epoch=5 ,model=model)

    save_model(model)

